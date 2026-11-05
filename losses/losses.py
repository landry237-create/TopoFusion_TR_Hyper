# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Objectif composite de TopoFusion-TR Hyper.

L = L_sem + l_fusion L_fusion + l_geom L_geom + l_top L_top + l_rob L_rob + l_orth L_orth + l_rec L_rec + l_var L_var

Chaque terme a un rôle précis et peut être retiré par une variante. La perte temporelle d'origine
est retirée : elle traitait des tokens de bruit comme une suite temporelle, ce qui n'a aucun sens.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from typing import Any, Dict, Optional, Sequence, Tuple  # Importe les types des signatures.

import numpy as np  # Fournit les indices des arêtes critiques.
import torch  # Fournit les tenseurs.
import torch.nn.functional as F  # Fournit l'entropie croisée et la normalisation.

from models.latent_bank import orthogonality_penalty  # Importe la pénalité d'orthogonalité.
from models.multimodal_framework import VariantSpec  # Importe la description des variantes.
from models.topology import critical_edges, landscape_l2_distance  # Importe les outils topologiques.


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:  # Calcule les distances par paires.
    """Retourne la matrice des distances euclidiennes, sans gradient infini sur la diagonale."""
    carres = (x.unsqueeze(0) - x.unsqueeze(1)).pow(2).sum(dim=-1)  # Calcule les carrés des distances.
    identite = torch.eye(x.size(0), device=x.device, dtype=x.dtype)  # Prépare la diagonale.
    return torch.sqrt(carres + identite + 1e-12) * (1.0 - identite)  # Annule la diagonale sans dériver la racine en zéro.


def _triangle_superieur(matrice: torch.Tensor) -> torch.Tensor:  # Extrait les paires distinctes.
    """Extrait le triangle supérieur strict, pour ne pas compter la diagonale ni les doublons."""
    lignes, colonnes = torch.triu_indices(matrice.size(0), matrice.size(1), offset=1, device=matrice.device)  # Calcule les indices.
    return matrice[lignes, colonnes]  # Retourne les valeurs sélectionnées.


def mantel_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # Calcule la corrélation de Mantel.
    """Calcule la corrélation de Pearson de deux matrices de distances, sur leurs paires distinctes."""
    x = _triangle_superieur(a)  # Extrait les distances de la première matrice.
    y = _triangle_superieur(b)  # Extrait les distances de la seconde matrice.
    x = x - x.mean()  # Centre la première série.
    y = y - y.mean()  # Centre la seconde série.
    return (x * y).sum() / (x.norm() * y.norm() + 1e-8)  # Une série constante, typique d'un effondrement, donne une corrélation nulle.


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: torch.Tensor, ids_texte: Optional[torch.Tensor] = None) -> torch.Tensor:  # Calcule InfoNCE symétrique.
    """Calcule InfoNCE symétrique ; une légende identique ailleurs dans le lot n'est pas traitée comme un négatif."""
    logits = torch.matmul(a, b.T) / temperature  # Calcule les similarités mises à l'échelle.
    if ids_texte is not None:  # Vérifie la présence des identifiants de texte.
        memes = ids_texte.unsqueeze(0) == ids_texte.unsqueeze(1)  # Détecte les textes identiques.
        faux_negatifs = memes & ~torch.eye(a.size(0), dtype=torch.bool, device=a.device)  # Conserve uniquement les faux négatifs.
        logits = logits.masked_fill(faux_negatifs, float("-inf"))  # Exclut les faux négatifs du dénominateur.
    cibles = torch.arange(a.size(0), device=a.device)  # Les paires correctes occupent la diagonale.
    return 0.5 * (F.cross_entropy(logits, cibles) + F.cross_entropy(logits.T, cibles))  # Moyenne les deux directions.


def geometric_loss(z: Dict[str, torch.Tensor], entrees: Dict[str, torch.Tensor]) -> torch.Tensor:  # Calcule la perte géométrique.
    """Préserve approximativement la géométrie : 1 - corrélation entrée-latent, et 1 - corrélation image-texte.

    Une représentation effondrée a une corrélation nulle et subit donc la perte maximale. Cette perte
    encourage une préservation affine des distances : elle n'établit pas une isométrie exacte.
    Une isométrie exacte vers chaque entrée serait incompatible avec l'alignement entre modalités.
    """
    distances = {nom: pairwise_distances(valeur) for nom, valeur in z.items()}  # Calcule les distances latentes.
    termes = []  # Prépare les termes de la perte.
    for nom, latent in distances.items():  # Parcourt les modalités.
        entree = pairwise_distances(F.normalize(entrees[nom].detach(), dim=-1))  # Calcule les distances d'entrée, sans gradient.
        termes.append(1.0 - mantel_correlation(entree, latent))  # Compare la structure d'entrée et la structure latente.
    if "image" in distances and "text" in distances:  # Vérifie la présence des deux modalités.
        termes.append(1.0 - mantel_correlation(distances["image"], distances["text"]))  # Compare les structures entre modalités.
    return torch.stack(termes).mean()  # Retourne la moyenne des termes.


def _normaliser_distances(d: torch.Tensor) -> torch.Tensor:  # Normalise l'échelle d'une matrice.
    """Divise une matrice de distances par sa moyenne sur les paires distinctes."""
    return d / _triangle_superieur(d).mean().clamp_min(1e-6)  # Retourne la matrice sans unité.


def signature_loss(a: torch.Tensor, b: torch.Tensor, dimensions: Sequence[int]) -> torch.Tensor:  # Calcule la perte de signature.
    """Compare deux matrices de distances sur leurs arêtes critiques exactes (Moor et al., 2020)."""
    na = _normaliser_distances(a)  # Normalise la première matrice.
    nb = _normaliser_distances(b)  # Normalise la seconde matrice.
    termes = []  # Prépare les deux directions.
    for source in (na, nb):  # Parcourt les deux directions.
        aretes = critical_edges(source.detach().cpu().numpy().astype(np.float64), dimensions)  # Calcule l'appariement exact.
        if aretes.shape[0] > 0:  # Vérifie qu'au moins une arête existe.
            i = torch.from_numpy(aretes[:, 0]).to(a.device)  # Convertit le premier sommet.
            j = torch.from_numpy(aretes[:, 1]).to(a.device)  # Convertit le second sommet.
            termes.append((na[i, j] - nb[i, j]).pow(2).mean())  # Compare les valeurs critiques.
    return torch.stack(termes).mean() if termes else torch.zeros(1, device=a.device).squeeze()  # Retourne la moyenne.


def topological_loss(z: Dict[str, torch.Tensor], entrees: Dict[str, torch.Tensor], nb_points: int, dimensions: Sequence[int]) -> torch.Tensor:  # Calcule la perte topologique.
    """Préserve H0 et H1 entre l'entrée et le latent de chaque modalité, et aligne image et texte."""
    premier = next(iter(z.values()))  # Récupère un embedding pour lire les dimensions.
    n = min(int(nb_points), premier.size(0))  # Limite le nombre de points, car le coût croît fortement.
    distances = {nom: pairwise_distances(valeur[:n]) for nom, valeur in z.items()}  # Calcule les distances latentes du sous-lot.
    termes = []  # Prépare les termes de la perte.
    for nom, latent in distances.items():  # Parcourt les modalités.
        entree = pairwise_distances(F.normalize(entrees[nom][:n].detach(), dim=-1))  # Calcule les distances d'entrée, sans gradient.
        termes.append(signature_loss(entree, latent, dimensions))  # Préserve la structure de la modalité.
    if "image" in distances and "text" in distances:  # Vérifie la présence des deux modalités.
        termes.append(signature_loss(distances["image"], distances["text"], dimensions))  # Aligne la topologie entre modalités.
    return torch.stack(termes).mean()  # Retourne la moyenne des termes.


def variance_hinge(x: torch.Tensor, cible: float = 0.1) -> torch.Tensor:  # Calcule la pénalité anti-effondrement.
    """Pénalise une variance insuffisante de chaque dimension sur le lot (Bardes et al., 2022)."""
    return F.relu(cible - torch.sqrt(x.var(dim=0) + 1e-6)).mean()  # Retourne la moyenne du dépassement.


def robustness_loss(propre: Dict[str, Any], perturbe: Dict[str, Any], temperature: torch.Tensor, couches: int, pas: int, ids_texte: Optional[torch.Tensor]) -> torch.Tensor:  # Calcule la perte de robustesse.
    """Combine cohérence des embeddings, recherche sous perturbation, cohérence topologique et cohérence du graphe."""
    termes = []  # Prépare les termes de la perte.
    for nom, z in perturbe["embeddings"].items():  # Parcourt les modalités.
        termes.append((1.0 - (propre["embeddings"][nom].detach() * z).sum(dim=-1)).mean())  # Rapproche la vue perturbée de la vue propre figée.
        vue_p = perturbe["views"].get(nom)  # Récupère la vue structurelle perturbée.
        vue_c = propre["views"].get(nom)  # Récupère la vue structurelle propre.
        if vue_p is not None and vue_p.persistence is not None:  # Vérifie la présence de diagrammes.
            termes.append(landscape_l2_distance(vue_p.persistence, _detacher_diagrammes(vue_c.persistence), couches, pas))  # Rapproche les paysages.
        if vue_p is not None and vue_p.adjacency is not None:  # Vérifie la présence du graphe.
            s = vue_c.log_variance  # Lit la log-variance des arêtes.
            termes.append((0.5 * torch.exp(-s) * (vue_c.adjacency.detach() - vue_p.adjacency).pow(2) + 0.5 * s).mean())  # Applique la vraisemblance hétéroscédastique.
    if "image" in perturbe["embeddings"] and "text" in perturbe["embeddings"]:  # Vérifie la présence des deux modalités.
        termes.append(info_nce(perturbe["embeddings"]["image"], propre["embeddings"]["text"], temperature, ids_texte))  # Recherche depuis l'image perturbée.
        termes.append(info_nce(propre["embeddings"]["image"], perturbe["embeddings"]["text"], temperature, ids_texte))  # Recherche depuis le texte perturbé.
    return torch.stack(termes).mean()  # Retourne la moyenne des termes.


def _detacher_diagrammes(diagrammes: Any) -> Any:  # Détache un lot de diagrammes.
    """Retourne une copie des diagrammes sans gradient, utilisée comme cible de cohérence."""
    from dataclasses import replace  # Importe la copie partielle d'un conteneur.

    return replace(diagrammes, h0_birth=diagrammes.h0_birth.detach(), h0_death=diagrammes.h0_death.detach(), h1_birth=diagrammes.h1_birth.detach(), h1_death=diagrammes.h1_death.detach())  # Retourne les valeurs détachées.


class CompositeLoss:  # Assemble les termes actifs de l'objectif.
    """Calcule l'objectif composite selon la variante et la phase d'entraînement."""

    TERMES_PAR_PHASE = {1: ("fusion", "geom", "orth", "rec", "var"), 2: ("fusion", "geom", "orth", "rec", "var", "top"), 3: ("fusion", "geom", "orth", "rec", "var", "top", "rob")}  # Fixe l'activation progressive des termes.

    def __init__(self, cfg: Dict[str, Any], variante: VariantSpec) -> None:  # Mémorise les paramètres.
        """Mémorise les poids et les paramètres topologiques de l'objectif."""
        self.poids = cfg["loss_weights"]  # Conserve les poids des termes.
        self.variante = variante  # Conserve la variante.
        t = cfg["topology"]  # Récupère la section topologique.
        self.nb_points = int(t["loss_points"])  # Conserve le nombre de points de la perte topologique.
        self.dimensions = tuple(t["loss_homology_dims"])  # Conserve les dimensions utilisées.
        self.couches = int(t["landscape_layers"])  # Conserve le nombre de couches de paysage.
        self.pas = int(t["landscape_steps"])  # Conserve le nombre de points de grille.

    def active_terms(self, phase: int) -> Tuple[str, ...]:  # Détermine les termes actifs.
        """Retourne les termes actifs pour une phase, en ne gardant que ceux de la variante."""
        return tuple(t for t in self.TERMES_PAR_PHASE[phase] if t in self.variante.pertes)  # Filtre les termes.

    def __call__(self, sorties: Dict[str, Any], sorties_perturbees: Optional[Dict[str, Any]], temperature: torch.Tensor, ids_texte: Optional[torch.Tensor], phase: int = 3) -> Tuple[torch.Tensor, Dict[str, float]]:  # Calcule l'objectif.
        """Retourne l'objectif et la valeur de chaque terme, sans gradient, pour le suivi."""
        z = sorties["embeddings"]  # Récupère les embeddings.
        entrees = sorties["pooled_input"]  # Récupère les caractéristiques d'entrée.
        termes: Dict[str, torch.Tensor] = {}  # Prépare les termes calculés.
        if self.variante.perte_naive:  # Traite la variante B0.
            termes["sem"] = (1.0 - (z["image"] * z["text"]).sum(dim=-1)).mean()  # Applique l'alignement cosinus sans négatifs.
        else:  # Traite les autres variantes.
            termes["sem"] = info_nce(z["image"], z["text"], temperature, ids_texte)  # Applique InfoNCE.
        actifs = self.active_terms(phase)  # Détermine les termes actifs.
        if "fusion" in actifs and "fused_embedding" in sorties:  # Vérifie le terme de fusion.
            termes["fusion"] = 0.5 * (info_nce(sorties["fused_embedding"], z["image"], temperature, None) + info_nce(sorties["fused_embedding"], z["text"], temperature, ids_texte))  # Relie l'embedding fusionné à chaque modalité.
        if "geom" in actifs:  # Vérifie le terme géométrique.
            termes["geom"] = geometric_loss(z, entrees)  # Calcule la perte géométrique.
        if "top" in actifs:  # Vérifie le terme topologique.
            termes["top"] = topological_loss(z, entrees, self.nb_points, self.dimensions)  # Calcule la perte topologique.
        if "orth" in actifs and sorties["shared"]:  # Vérifie le terme d'orthogonalité.
            termes["orth"] = torch.stack([orthogonality_penalty(sorties["shared"][n], sorties["private"][n]) for n in sorties["shared"]]).mean()  # Sépare les espaces partagé et privé.
        if "rec" in actifs and sorties["reconstruction"]:  # Vérifie le terme de reconstruction.
            termes["rec"] = torch.stack([(1.0 - F.cosine_similarity(sorties["reconstruction"][n], entrees[n].detach(), dim=-1)).mean() for n in sorties["reconstruction"]]).mean()  # Donne un rôle réel à l'espace privé.
        if "var" in actifs and sorties["shared"]:  # Vérifie le terme de variance.
            termes["var"] = torch.stack([variance_hinge(sorties["shared"][n].mean(dim=1)) + variance_hinge(sorties["private"][n].mean(dim=1)) for n in sorties["shared"]]).mean()  # Empêche l'effondrement des tokens.
        if "rob" in actifs and sorties_perturbees is not None:  # Vérifie le terme de robustesse.
            termes["rob"] = robustness_loss(sorties, sorties_perturbees, temperature, self.couches, self.pas, ids_texte)  # Calcule la perte de robustesse.
        total = sum(self.poids[nom] * valeur for nom, valeur in termes.items())  # Calcule la somme pondérée.
        details = {f"L_{nom}": float(valeur.detach().item()) for nom, valeur in termes.items()}  # Convertit les termes pour le suivi.
        details["total"] = float(total.detach().item())  # Ajoute la valeur totale.
        return total, details  # Retourne l'objectif et le détail.
