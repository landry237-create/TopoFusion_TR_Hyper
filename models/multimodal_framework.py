# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Architecture TopoFusion-TR Hyper et variantes du protocole expérimental.

L'embedding de recherche suit z_m = normalize(g_m(moyenne(P_m(tokens))) + gamma * h(structure_m)).
La voie structurelle contient la banque, le graphe, la filtration, la persistance et l'attention.
Contrairement à la version d'origine, elle contribue réellement à l'embedding, et gamma mesure ce poids.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from dataclasses import dataclass, field, replace  # Définit la description des variantes.
from typing import Any, Dict, List, Optional, Tuple  # Importe les types des signatures.

import torch  # Fournit les tenseurs.
import torch.nn as nn  # Fournit les modules de réseaux de neurones.
import torch.nn.functional as F  # Fournit la normalisation.

from models.attention import TopologicalTransformerBlock  # Importe le bloc d'attention.
from models.encoders import ModalityProjector, pool_tokens  # Importe les projecteurs et le pooling.
from models.latent_bank import AdaptiveFusion, SemanticLatentBank, SharedPrivateFactorization  # Importe la banque et la fusion.
from models.topology import AdaptiveLatentGraph, LipschitzFiltration, PersistenceBatch, TopologicalEncoder, diffuse_on_graph, differentiable_persistence, topological_bias  # Importe le moteur topologique.


@dataclass(frozen=True)  # Rend la description d'une variante immuable.
class VariantSpec:  # Décrit une variante expérimentale.
    """Interrupteurs d'architecture et termes de perte d'une variante."""

    nom: str  # Nom de la variante.
    description: str  # Description lisible de la variante.
    banque: bool = False  # Active la banque latente et le partage partagé/privé.
    tokens_topo: bool = False  # Active les tokens H0 et H1 dans l'embedding.
    attention_topo: bool = False  # Active le biais topologique de l'attention.
    graphe: bool = False  # Active le graphe latent adaptatif.
    perte_naive: bool = False  # Remplace InfoNCE par un alignement sans négatifs.
    pertes: Tuple[str, ...] = field(default_factory=tuple)  # Liste les termes actifs en plus du terme sémantique.
    perturbations: bool = False  # Active les perturbations d'entraînement.


TOUS_TERMES_B4 = ("fusion", "geom", "top", "rob", "orth", "rec", "var")  # Liste les termes de l'objectif complet.
SANS_TOP = tuple(t for t in TOUS_TERMES_B4 if t != "top")  # Retire le terme topologique.
SANS_ROB = tuple(t for t in TOUS_TERMES_B4 if t != "rob")  # Retire le terme de robustesse.
B4 = VariantSpec("B4", "TopoFusion-TR Hyper complet", banque=True, tokens_topo=True, attention_topo=True, graphe=True, pertes=TOUS_TERMES_B4, perturbations=True)  # Définit le modèle complet.

VARIANTS: Dict[str, VariantSpec] = {  # Enregistre toutes les variantes.
    "B0": VariantSpec("B0", "Projection naïve : alignement cosinus sans négatifs", perte_naive=True),  # Définit la référence naïve.
    "B1": VariantSpec("B1", "Alignement contrastif InfoNCE"),  # Définit la référence contrastive.
    "B2": VariantSpec("B2", "InfoNCE + isométrie géométrique approchée", pertes=("geom",)),  # Ajoute la géométrie.
    "B3": VariantSpec("B3", "B2 + représentation topologique", tokens_topo=True, pertes=("geom", "top")),  # Ajoute la représentation topologique.
    "B4": B4,  # Enregistre le modèle complet.
    "B4_sans_L_top": replace(B4, nom="B4_sans_L_top", description="B4 sans perte topologique", pertes=SANS_TOP),  # Retire uniquement la perte topologique.
    "B4_sans_tokens_topo": replace(B4, nom="B4_sans_tokens_topo", description="B4 sans tokens H0 et H1", tokens_topo=False),  # Retire uniquement les tokens topologiques.
    "B4_sans_attention_topo": replace(B4, nom="B4_sans_attention_topo", description="B4 sans biais topologique de l'attention", attention_topo=False),  # Retire uniquement le biais de l'attention.
    "B4_sans_robustesse": replace(B4, nom="B4_sans_robustesse", description="B4 sans perturbation ni perte de robustesse", pertes=SANS_ROB, perturbations=False),  # Retire la robustesse.
    "B4_sans_topologie": replace(B4, nom="B4_sans_topologie", description="Banque et fusion seules, sans aucune topologie", tokens_topo=False, attention_topo=False, graphe=False, pertes=SANS_TOP),  # Retire toute la topologie.
}  # Termine le registre.


def get_variant(nom: str) -> VariantSpec:  # Retrouve une variante par son nom.
    """Retourne une variante ou lève une erreur qui liste les noms valides."""
    if nom not in VARIANTS:  # Vérifie que le nom existe.
        raise KeyError(f"Variante inconnue : {nom}. Variantes disponibles : {sorted(VARIANTS)}")  # Explique l'erreur.
    return VARIANTS[nom]  # Retourne la variante.


@dataclass  # Génère le constructeur du conteneur.
class StructuralView:  # Regroupe les résultats de la voie structurelle.
    """Résultats de la voie structurelle pour un nuage de tokens."""

    pooled: torch.Tensor  # Représentation structurelle agrégée (B, d).
    filtration: Optional[torch.Tensor] = None  # Matrice de filtration (B, N, N).
    persistence: Optional[PersistenceBatch] = None  # Diagrammes différentiables.
    adjacency: Optional[torch.Tensor] = None  # Adjacence du graphe (B, N, N).
    log_variance: Optional[torch.Tensor] = None  # Log-variance des arêtes (B, N, N).
    bias: Optional[torch.Tensor] = None  # Biais topologique T_ij (B, N, N).
    attention: Optional[torch.Tensor] = None  # Poids d'attention (B, H, N, N).
    descriptors: Dict[str, torch.Tensor] = field(default_factory=dict)  # Descripteurs topologiques.
    points: Optional[torch.Tensor] = None  # Points d'entrée de la filtration (B, N, d).


class TopoFusionModel(nn.Module):  # Définit le modèle entraînable.
    """Modèle entraînable : reçoit uniquement les modalités réelles disponibles."""

    def __init__(self, cfg: Dict[str, Any], variante: VariantSpec, dims_entree: Dict[str, int]) -> None:  # Construit le modèle.
        """Construit uniquement les modules utiles à la variante et aux modalités actives."""
        super().__init__()  # Initialise le module parent.
        self.variante = variante  # Conserve la description de la variante.
        self.modalites = sorted(dims_entree)  # Conserve la liste des modalités réelles.
        m = cfg["model"]  # Récupère la section du modèle.
        t = cfg["topology"]  # Récupère la section topologique.
        d = int(m["latent_dim"])  # Lit la dimension latente.
        self.dim_latente = d  # Conserve la dimension latente.
        self.projecteurs = nn.ModuleDict({nom: ModalityProjector(dim, d, m["dropout"]) for nom, dim in dims_entree.items()})  # Crée les projecteurs P_m.
        self.tetes_semantiques = nn.ModuleDict({nom: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)) for nom in dims_entree})  # Crée les têtes sémantiques g_m.
        self.log_temperature = nn.Parameter(torch.tensor(float(m["initial_temperature"])).log())  # Rend la température réellement apprenable.
        utilise_structure = variante.banque or variante.tokens_topo  # Détermine si la voie structurelle existe.
        utilise_persistance = variante.tokens_topo or variante.attention_topo  # Détermine si la persistance est calculée.
        if utilise_structure:  # Vérifie la présence de la voie structurelle.
            valeur = float(m["initial_structural_gate"])  # Lit la valeur initiale de la porte.
            self.porte_brute = nn.Parameter(torch.logit(torch.tensor(valeur)))  # Crée la porte apprise gamma.
            self.tete_structurelle = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d, bias=False))  # Crée la tête structurelle h, sans décalage constant en sortie.
            nn.init.zeros_(self.tete_structurelle[-1].weight)  # Démarre à zéro : la voie structurelle ne s'écarte de la voie sémantique que si le gradient le justifie.
        if variante.banque:  # Vérifie la présence de la banque.
            self.banque = SemanticLatentBank(m["num_latent_tokens"], d, m["num_heads"], m["dropout"])  # Crée la banque latente.
            self.factorisation = SharedPrivateFactorization(m["num_shared_tokens"], d, self.modalites)  # Crée la factorisation.
            self.fusion = AdaptiveFusion(d)  # Crée la fusion adaptative.
            self.bloc = TopologicalTransformerBlock(d, m["num_heads"], m["dropout"])  # Crée le bloc d'attention.
            self.reconstruction = nn.ModuleDict({nom: nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, dim)) for nom, dim in dims_entree.items()})  # Crée les décodeurs de la reconstruction.
        if variante.graphe:  # Vérifie la présence du graphe.
            self.graphe = AdaptiveLatentGraph(d, int(m["graph_hidden_dim"]))  # Crée le graphe latent.
        if utilise_persistance:  # Vérifie la présence de la persistance.
            self.filtration = LipschitzFiltration(d, int(t["filtration_dim"]), int(t["num_scales"]))  # Crée la filtration.
        if variante.tokens_topo:  # Vérifie la présence des tokens topologiques.
            self.encodeur_topo = TopologicalEncoder(d, int(m["num_h0_tokens"]), int(m["num_h1_tokens"]), int(t["landscape_layers"]), int(t["landscape_steps"]), int(t["num_scales"]), float(t["soft_betti_temperature"]))  # Crée l'encodeur topologique.

    @property  # Expose la température sous forme d'attribut.
    def temperature(self) -> torch.Tensor:  # Calcule la température.
        """Retourne la température InfoNCE, bornée pour la stabilité numérique."""
        return self.log_temperature.exp().clamp(0.01, 0.5)  # Borne la température apprise.

    @property  # Expose la porte sous forme d'attribut.
    def porte(self) -> torch.Tensor:  # Calcule la porte.
        """Retourne la porte gamma dans ]0, 1[, ou 0 si la voie structurelle n'existe pas."""
        return torch.sigmoid(self.porte_brute) if hasattr(self, "porte_brute") else torch.zeros(1, device=self.log_temperature.device)  # Calcule la porte.

    def _voie_structurelle(self, tokens: torch.Tensor, masque: Optional[torch.Tensor], desactiver_topo: bool = False, nom_vue: str = "image") -> StructuralView:  # Exécute la voie structurelle.
        """Exécute graphe, filtration, persistance, encodage et attention, selon la variante."""
        v = self.variante  # Raccourcit l'accès à la variante.
        adjacence, log_variance = (None, None)  # Initialise le graphe.
        nuage = tokens  # Initialise le nuage de points.
        if v.graphe:  # Vérifie la présence du graphe.
            adjacence, log_variance = self.graphe(tokens, masque)  # Calcule le graphe latent.
            nuage = diffuse_on_graph(tokens, adjacence)  # Le graphe alimente la filtration.
        points = F.normalize(nuage, dim=-1)  # Projette les points sur la sphère unité.
        filtration, diagrammes, biais = (None, None, None)  # Initialise les résultats topologiques.
        tokens_topo: List[torch.Tensor] = []  # Prépare les tokens topologiques.
        descripteurs: Dict[str, torch.Tensor] = {}  # Prépare les descripteurs.
        if v.tokens_topo or v.attention_topo:  # Vérifie si la persistance est utile.
            filtration = self.filtration(points)  # Calcule la filtration de Rips pondérée.
            if v.tokens_topo:  # Vérifie la présence des tokens topologiques.
                diagrammes = differentiable_persistence(filtration, masque)  # Calcule les diagrammes différentiables.
                h0, h1, descripteurs = self.encodeur_topo(diagrammes, self.filtration.echelles, nom_vue)  # Encode les diagrammes.
                tokens_topo = [h0, h1]  # Conserve les tokens topologiques.
            if v.attention_topo:  # Vérifie la présence du biais.
                biais = topological_bias(filtration, masque)  # Calcule le biais issu de H0.
                if adjacence is not None:  # Vérifie la présence du graphe.
                    biais = 0.5 * (biais + adjacence)  # Combine la structure persistante et le graphe appris.
        if desactiver_topo:  # Vérifie si l'ablation à l'inférence est demandée.
            tokens_topo = [torch.zeros_like(t) for t in tokens_topo]  # Remplace les tokens topologiques par des zéros.
        if not v.banque:  # Traite la variante B3, sans bloc d'attention.
            agrege = torch.cat(tokens_topo, dim=1).mean(dim=1)  # Agrège les tokens H0 et H1.
            return StructuralView(agrege, filtration, diagrammes, None, None, None, None, descripteurs, points)  # Retourne la vue de B3.
        tous = torch.cat([tokens] + tokens_topo, dim=1)  # Assemble les tokens partagés et topologiques.
        biais_complet = None  # Initialise le biais complet.
        if biais is not None:  # Vérifie la présence du biais.
            biais_complet = F.pad(biais, (0, tous.size(1) - tokens.size(1), 0, tous.size(1) - tokens.size(1)))  # Étend le biais aux tokens topologiques avec des zéros.
        sortie, attention = self.bloc(tous, biais_complet)  # Applique le bloc d'attention.
        return StructuralView(sortie.mean(dim=1), filtration, diagrammes, adjacence, log_variance, biais, attention, descripteurs, points)  # Retourne la vue complète.

    def forward(self, entrees: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]], disponibilite: Optional[Dict[str, torch.Tensor]] = None, desactiver_topo: bool = False, fusionner: bool = True) -> Dict[str, Any]:  # Exécute le modèle.
        """Encode chaque modalité réelle présente dans entrees, sous la forme nom -> (tokens, masque)."""
        inconnues = set(entrees) - set(self.modalites)  # Détecte des modalités non construites.
        if inconnues:  # Vérifie l'absence de modalité inconnue.
            raise KeyError(f"Modalités non construites par ce modèle : {sorted(inconnues)}")  # Explique l'erreur.
        sorties: Dict[str, Any] = {"embeddings": {}, "semantic": {}, "pooled_input": {}, "views": {}, "shared": {}, "private": {}, "reconstruction": {}}  # Prépare les sorties.
        v = self.variante  # Raccourcit l'accès à la variante.
        for nom, (tokens, masque) in entrees.items():  # Parcourt uniquement les modalités réelles reçues.
            masque_complet = masque if masque is not None else torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)  # Construit un masque complet si nécessaire.
            projetes = self.projecteurs[nom](tokens)  # Applique le projecteur P_m.
            semantique = self.tetes_semantiques[nom](pool_tokens(projetes, masque_complet))  # Calcule la partie sémantique.
            sorties["semantic"][nom] = semantique  # Conserve la partie sémantique.
            sorties["pooled_input"][nom] = pool_tokens(tokens, masque_complet)  # Conserve la caractéristique d'entrée moyenne.
            if v.banque:  # Traite les variantes avec banque.
                partage, prive = self.factorisation(self.banque(projetes, masque_complet), nom)  # Sépare les espaces partagé et privé.
                sorties["shared"][nom] = partage  # Conserve les tokens partagés.
                sorties["private"][nom] = prive  # Conserve les tokens privés.
                sorties["reconstruction"][nom] = self.reconstruction[nom](torch.cat([partage.mean(dim=1), prive.mean(dim=1)], dim=-1))  # Reconstruit la caractéristique d'entrée.
                vue = self._voie_structurelle(partage, None, desactiver_topo, nom)  # Exécute la voie structurelle sur les tokens partagés.
                sorties["embeddings"][nom] = F.normalize(semantique + self.porte * self.tete_structurelle(vue.pooled), dim=-1)  # Combine les deux voies.
                sorties["views"][nom] = vue  # Conserve la vue structurelle.
            elif v.tokens_topo:  # Traite la variante B3.
                nuage = masque_complet.clone()  # Copie le masque pour définir le nuage de points.
                nuage[:, 0] = nuage.sum(dim=1) <= 1  # Exclut le jeton global du nuage, sauf s'il est le seul token.
                vue = self._voie_structurelle(projetes, nuage, desactiver_topo, nom)  # Exécute la voie structurelle sur les tokens projetés.
                sorties["embeddings"][nom] = F.normalize(semantique + self.porte * self.tete_structurelle(vue.pooled), dim=-1)  # Combine les deux voies.
                sorties["views"][nom] = vue  # Conserve la vue structurelle.
            else:  # Traite les variantes B0, B1 et B2.
                sorties["embeddings"][nom] = F.normalize(semantique, dim=-1)  # Normalise l'embedding sémantique.
        if v.banque and fusionner:  # Vérifie que la fusion existe et qu'elle est demandée.
            taille_lot = next(iter(entrees.values()))[0].size(0)  # Lit la taille du lot.
            appareil = next(iter(entrees.values()))[0].device  # Lit le device du lot.
            presence = disponibilite if disponibilite is not None else {nom: torch.ones(taille_lot, dtype=torch.bool, device=appareil) for nom in entrees}  # Considère toutes les modalités reçues comme présentes.
            presence = {nom: presence[nom] for nom in entrees}  # Limite la présence aux modalités reçues.
            fusionne, alphas, noms = self.fusion(sorties["shared"], presence)  # Fusionne les tokens partagés.
            vue_fusion = self._voie_structurelle(fusionne, None, desactiver_topo, "fusion")  # Exécute la voie structurelle fusionnée.
            poids = torch.stack([presence[nom].to(fusionne.dtype) for nom in noms], dim=1)  # Empile la présence de chaque modalité.
            semantiques = torch.stack([sorties["semantic"][nom] for nom in noms], dim=1)  # Empile les parties sémantiques.
            moyenne = (semantiques * poids.unsqueeze(-1)).sum(dim=1) / poids.sum(dim=1, keepdim=True).clamp_min(1.0)  # Moyenne les seules modalités présentes.
            sorties["fused_embedding"] = F.normalize(moyenne + self.porte * self.tete_structurelle(vue_fusion.pooled), dim=-1)  # Calcule l'embedding fusionné.
            sorties["fused_view"] = vue_fusion  # Conserve la vue fusionnée.
            sorties["alphas"] = alphas  # Conserve les poids de fusion.
            sorties["alpha_names"] = noms  # Conserve l'ordre des modalités.
        return sorties  # Retourne toutes les sorties.

    def parameter_groups(self, decroissance: float) -> List[Dict[str, Any]]:  # Sépare les groupes de paramètres.
        """Exclut de la décroissance des poids les biais, les normalisations, la température, la porte et les échelles."""
        avec: List[nn.Parameter] = []  # Prépare les paramètres avec décroissance.
        sans: List[nn.Parameter] = []  # Prépare les paramètres sans décroissance.
        for nom, parametre in self.named_parameters():  # Parcourt les paramètres du modèle.
            if parametre.ndim <= 1 or "banque.tokens" in nom or "lambda_biais" in nom or "echelles_brutes" in nom:  # Détecte les paramètres à exclure.
                sans.append(parametre)  # Exclut le paramètre de la décroissance.
            else:  # Traite un poids ordinaire.
                avec.append(parametre)  # Applique la décroissance.
        return [{"params": avec, "weight_decay": decroissance}, {"params": sans, "weight_decay": 0.0}]  # Retourne les deux groupes.


def count_parameters(modele: nn.Module) -> Dict[str, int]:  # Compte les paramètres.
    """Retourne le nombre de paramètres entraînables du modèle."""
    return {"entrainables": sum(p.numel() for p in modele.parameters() if p.requires_grad)}  # Compte les paramètres entraînables.


def modality_inputs(lot: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:  # Convertit un lot.
    """Convertit un lot de caractéristiques au format attendu par le modèle."""
    return {"image": (lot["image_tokens"], None), "text": (lot["text_tokens"], lot["text_mask"])}  # Associe les tokens et les masques.
