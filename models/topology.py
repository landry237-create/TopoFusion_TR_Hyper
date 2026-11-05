# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Moteur topologique : graphe latent, filtration, persistance et encodage.

Trois niveaux sont volontairement distingués :
A. Persistance exacte : GUDHI calcule les appariements et les diagrammes.
B. Persistance différentiable : les valeurs de naissance et de mort sont relues dans
   le tenseur de filtration aux simplexes critiques trouvés par GUDHI. C'est la valeur
   exacte, dérivable presque partout, l'appariement restant constant par morceaux
   (Carrière et al., 2021 ; Leygonie et al., 2021).
C. Objectifs auxiliaires : paysages et courbes de Betti douces. Le paysage est un
   calcul exact sur la grille, tandis que la courbe de Betti douce est une approximation
   réglée par une température.

Garantie démontrée (voir docs/FONDEMENTS_MATHEMATIQUES.md) : pour la filtration de Rips
pondérée, et deux nuages de mêmes sommets, d_B <= (L_phi + L_w) * max_i ||u_i - u'_i||.
Cette borne porte sur les points d'entrée de la filtration, jamais sur les pixels ou les
mots, car les encodeurs et les projecteurs ne sont pas contraints.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from dataclasses import dataclass  # Définit des conteneurs de résultats.
from typing import Dict, List, Optional, Sequence, Tuple  # Importe les types des signatures.

import gudhi  # Calcule l'homologie persistante exacte.
import numpy as np  # Manipule les tableaux lus par GUDHI.
import torch  # Fournit les tenseurs différentiables.
import torch.nn as nn  # Fournit les modules de réseaux de neurones.
import torch.nn.functional as F  # Fournit softplus, normalize et linear.


class ExactSpectralLinear(nn.Module):  # Couche linéaire à norme d'opérateur bornée.
    """Couche linéaire dont la norme spectrale est bornée par 1 grâce à une SVD exacte.

    La normalisation spectrale usuelle estime sigma par itération de la puissance : la norme
    réelle peut alors dépasser 1 (1,0033 a été mesuré dans la version d'origine).
    """

    def __init__(self, dim_entree: int, dim_sortie: int) -> None:  # Construit la couche.
        """Crée les poids et le biais de la couche."""
        super().__init__()  # Initialise le module parent.
        self.poids = nn.Parameter(torch.empty(dim_sortie, dim_entree))  # Déclare la matrice de poids.
        self.biais = nn.Parameter(torch.zeros(dim_sortie))  # Déclare le biais, sans effet sur la constante de Lipschitz.
        nn.init.orthogonal_(self.poids)  # Initialise la matrice avec des lignes orthonormées.

    def poids_contraints(self) -> torch.Tensor:  # Calcule la matrice contrainte.
        """Retourne W / max(1, ||W||_2), dont la norme spectrale vaut au plus 1."""
        norme = torch.linalg.matrix_norm(self.poids, ord=2)  # Calcule exactement la plus grande valeur singulière.
        return self.poids / norme.clamp_min(1.0)  # Réduit la matrice seulement si nécessaire.

    def forward(self, entree: torch.Tensor) -> torch.Tensor:  # Applique la couche.
        """Applique la transformation affine contrainte."""
        return F.linear(entree, self.poids_contraints(), self.biais)  # Retourne la sortie affine.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def norme_spectrale(self) -> float:  # Mesure la norme réelle.
        """Retourne la norme spectrale exacte de la matrice contrainte."""
        return float(torch.linalg.matrix_norm(self.poids_contraints(), ord=2).item())  # Convertit en flottant.


class AdaptiveLatentGraph(nn.Module):  # Construit le graphe latent appris.
    """Graphe adaptatif sur les tokens : similarité, relations et incertitude apprise.

    La variance log_sigma² est hétéroscédastique. Elle n'est pas bayésienne : elle prend son
    sens grâce au terme de vraisemblance de la perte de robustesse (Kendall et Gal, 2017).
    """

    def __init__(self, dim_latente: int, dim_cachee: int) -> None:  # Construit les sous-réseaux.
        """Crée les projections de similarité, de relation et d'incertitude."""
        super().__init__()  # Initialise le module parent.
        self.compression = nn.Sequential(nn.Linear(dim_latente, dim_cachee), nn.LayerNorm(dim_cachee))  # Réduit le coût des paires.
        self.requete = nn.Linear(dim_cachee, dim_cachee, bias=False)  # Projette le token source.
        self.cle = nn.Linear(dim_cachee, dim_cachee, bias=False)  # Projette le token cible.
        self.relation = nn.Sequential(nn.Linear(dim_cachee * 4, dim_cachee), nn.GELU(), nn.Linear(dim_cachee, 1))  # Estime la relation non linéaire.
        self.incertitude = nn.Sequential(nn.Linear(dim_cachee * 2, dim_cachee), nn.GELU(), nn.Linear(dim_cachee, 1))  # Estime la log-variance de chaque arête.
        self.echelle = dim_cachee ** -0.5  # Fixe le facteur de normalisation du produit scalaire.

    def forward(self, tokens: torch.Tensor, masque: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:  # Calcule le graphe.
        """Retourne l'adjacence symétrique dans [0, 1] et la log-variance symétrique des arêtes."""
        _, nombre, _ = tokens.shape  # Lit le nombre de tokens.
        h = self.compression(tokens)  # Compresse les tokens.
        similarite = torch.matmul(self.requete(h), self.cle(h).transpose(1, 2)) * self.echelle  # Calcule la similarité bilinéaire.
        h_i = h.unsqueeze(2).expand(-1, -1, nombre, -1)  # Répète les tokens sources.
        h_j = h.unsqueeze(1).expand(-1, nombre, -1, -1)  # Répète les tokens cibles.
        relation = self.relation(torch.cat([h_i, h_j, (h_i - h_j).abs(), h_i * h_j], dim=-1)).squeeze(-1)  # Calcule la relation par paire.
        log_variance = self.incertitude(torch.cat([h_i, h_j], dim=-1)).squeeze(-1).clamp(-4.0, 4.0)  # Calcule et borne la log-variance.
        log_variance = 0.5 * (log_variance + log_variance.transpose(1, 2))  # Rend la log-variance symétrique.
        scores = (similarite + relation) * torch.exp(-0.5 * log_variance)  # Atténue les arêtes les plus incertaines.
        adjacence = torch.sigmoid(scores)  # Convertit les scores en poids d'arête.
        adjacence = 0.5 * (adjacence + adjacence.transpose(1, 2))  # Rend l'adjacence symétrique.
        identite = torch.eye(nombre, device=tokens.device, dtype=tokens.dtype).unsqueeze(0)  # Prépare la diagonale.
        adjacence = adjacence * (1.0 - identite) + identite  # Fixe les boucles réflexives à 1.
        if masque is not None:  # Vérifie la présence d'un masque de padding.
            paires = masque.unsqueeze(1) & masque.unsqueeze(2)  # Calcule les paires de tokens valides.
            adjacence = adjacence * paires.to(tokens.dtype)  # Supprime les arêtes vers le padding.
        return adjacence, log_variance  # Retourne le graphe et son incertitude.


def diffuse_on_graph(tokens: torch.Tensor, adjacence: torch.Tensor, beta: float = 0.5) -> torch.Tensor:  # Diffuse l'information sur le graphe.
    """Applique une étape de passage de messages avec connexion résiduelle ; le graphe alimente ainsi la filtration."""
    identite = torch.eye(tokens.size(1), device=tokens.device, dtype=tokens.dtype).unsqueeze(0)  # Prépare la diagonale.
    voisins = adjacence * (1.0 - identite)  # Retire les boucles réflexives.
    message = torch.matmul(voisins, tokens) / voisins.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # Calcule la moyenne pondérée des voisins.
    return tokens + beta * message  # Ajoute le message au token d'origine.


class LipschitzFiltration(nn.Module):  # Calcule la filtration de Rips pondérée.
    """Filtration de Rips pondérée sur une métrique apprise (Anai et al., 2020).

    F(i)  = w_i = softplus(g(u_i) - décalage) >= 0
    F(ij) = max(w_i, w_j, (D_ij + w_i + w_j) / 2), avec D_ij = ||phi(u_i) - phi(u_j)||
    F(ijk) = max des arêtes, calculé par l'expansion de GUDHI.
    phi et g sont 1-Lipschitz, si bien que F dépend de façon lipschitzienne des points u_i.
    """

    def __init__(self, dim_latente: int, dim_filtration: int, nb_echelles: int, decalage: float = 2.0) -> None:  # Construit la filtration.
        """Crée la métrique apprise, le poids des sommets et les échelles ordonnées."""
        super().__init__()  # Initialise le module parent.
        self.metrique = nn.Sequential(ExactSpectralLinear(dim_latente, dim_filtration), nn.ReLU(), ExactSpectralLinear(dim_filtration, dim_filtration))  # Définit phi, de constante au plus 1.
        self.sommets = nn.Sequential(ExactSpectralLinear(dim_latente, dim_filtration), nn.ReLU(), ExactSpectralLinear(dim_filtration, 1))  # Définit g, de constante au plus 1.
        self.decalage = float(decalage)  # Fixe le décalage qui rend les poids initiaux faibles.
        cibles = torch.linspace(0.2, 1.4, nb_echelles)  # Fixe les échelles initiales.
        ecarts = torch.diff(torch.cat([torch.zeros(1), cibles]))  # Calcule les écarts successifs.
        self.echelles_brutes = nn.Parameter(torch.log(torch.expm1(ecarts)))  # Applique l'inverse exact de softplus.

    @property  # Expose les échelles sous forme d'attribut.
    def echelles(self) -> torch.Tensor:  # Calcule les échelles.
        """Retourne les échelles strictement croissantes epsilon_1 < ... < epsilon_S."""
        return torch.cumsum(F.softplus(self.echelles_brutes), dim=0)  # Cumule des écarts positifs.

    def poids_sommets(self, points: torch.Tensor) -> torch.Tensor:  # Calcule les poids des sommets.
        """Retourne les valeurs de filtration des sommets, positives ou nulles."""
        return F.softplus(self.sommets(points).squeeze(-1) - self.decalage)  # Softplus est 1-Lipschitz et positif.

    def distances(self, points: torch.Tensor) -> torch.Tensor:  # Calcule la métrique apprise.
        """Retourne les distances par paires dans l'espace appris."""
        image = self.metrique(points)  # Applique la transformation phi.
        carres = (image.unsqueeze(1) - image.unsqueeze(2)).pow(2).sum(dim=-1)  # Calcule les carrés des distances.
        return torch.sqrt(carres + 1e-12)  # Évite un gradient infini lorsque la distance est nulle.

    def forward(self, points: torch.Tensor) -> torch.Tensor:  # Calcule la matrice de filtration.
        """Retourne la matrice (B, N, N) : arêtes hors diagonale, sommets sur la diagonale."""
        w = self.poids_sommets(points)  # Calcule les poids des sommets.
        w_i = w.unsqueeze(2).expand(-1, -1, w.size(1))  # Répète les poids en ligne.
        w_j = w.unsqueeze(1).expand(-1, w.size(1), -1)  # Répète les poids en colonne.
        aretes = torch.maximum(torch.maximum(w_i, w_j), 0.5 * (self.distances(points) + w_i + w_j))  # Applique la formule du Rips pondéré.
        identite = torch.eye(w.size(1), device=points.device, dtype=points.dtype).unsqueeze(0)  # Prépare la diagonale.
        return aretes * (1.0 - identite) + torch.diag_embed(w)  # Place les poids des sommets sur la diagonale.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def lipschitz_bound(self) -> Dict[str, float]:  # Calcule les constantes de Lipschitz.
        """Retourne les majorants exacts de L_phi, L_w et de la borne de stabilité L_phi + L_w."""
        l_phi = float(np.prod([couche.norme_spectrale() for couche in self.metrique if isinstance(couche, ExactSpectralLinear)]))  # Multiplie les normes des couches de phi.
        l_w = float(np.prod([couche.norme_spectrale() for couche in self.sommets if isinstance(couche, ExactSpectralLinear)]))  # Multiplie les normes des couches de g.
        return {"L_phi": l_phi, "L_w": l_w, "borne_stabilite": l_phi + l_w}  # Retourne les trois valeurs.


@dataclass  # Génère le constructeur du conteneur.
class PersistencePairs:  # Décrit les simplexes critiques d'un nuage.
    """Indices des simplexes critiques calculés exactement par GUDHI."""

    h0_birth: np.ndarray  # Sommets de naissance des composantes finies, de forme (n0,).
    h0_death: np.ndarray  # Arêtes de mort des composantes finies, de forme (n0, 2).
    h0_essential: np.ndarray  # Sommets de naissance des composantes essentielles, de forme (e0,).
    h1_birth: np.ndarray  # Arêtes de naissance des cycles, de forme (n1, 2).
    h1_death: np.ndarray  # Arêtes critiques de mort des cycles, de forme (n1, 2).


def compute_flag_pairs(filtration: np.ndarray) -> PersistencePairs:  # Calcule la persistance exacte.
    """Calcule H0 et H1 du complexe de drapeaux pondéré par la matrice reçue.

    La diagonale donne les valeurs des sommets, et chaque triangle hérite du maximum de ses arêtes.
    """
    nombre = filtration.shape[0]  # Lit le nombre de points.
    if nombre <= 1:  # Traite le cas limite où GUDHI ignore la dimension maximale du complexe.
        vide = np.empty((0, 2), dtype=np.int64)  # Prépare un tableau d'arêtes vide.
        return PersistencePairs(np.empty(0, dtype=np.int64), vide, np.arange(nombre, dtype=np.int64), vide, vide)  # Retourne la seule composante essentielle possible.
    arbre = gudhi.SimplexTree()  # Crée l'arbre de simplexes.
    arbre.insert_batch(np.arange(nombre, dtype=np.int64).reshape(1, -1), np.diag(filtration).astype(np.float64))  # Insère les sommets.
    lignes, colonnes = np.triu_indices(nombre, k=1)  # Énumère les arêtes.
    if lignes.size > 0:  # Vérifie qu'au moins une arête existe.
        arbre.insert_batch(np.stack([lignes, colonnes]).astype(np.int64), filtration[lignes, colonnes].astype(np.float64))  # Insère toutes les arêtes.
    arbre.expansion(2)  # Ajoute les triangles, indispensables pour tuer les cycles.
    arbre.compute_persistence(homology_coeff_field=2, min_persistence=0.0)  # Calcule la persistance sur Z/2.
    h0_regulier, superieures, h0_essentiel, _ = arbre.flag_persistence_generators()  # Lit directement les sommets et les arêtes qui fixent chaque valeur critique.
    h1_regulier = superieures[0] if len(superieures) > 0 else np.empty((0, 4), dtype=np.int64)  # Sélectionne les cycles de dimension 1.
    h0_regulier = np.asarray(h0_regulier, dtype=np.int64).reshape(-1, 3)  # Normalise la forme de H0.
    h1_regulier = np.asarray(h1_regulier, dtype=np.int64).reshape(-1, 4)  # Normalise la forme de H1.
    garde_0 = filtration[h0_regulier[:, 1], h0_regulier[:, 2]] > filtration[h0_regulier[:, 0], h0_regulier[:, 0]]  # Écarte les paires de persistance nulle de H0.
    garde_1 = filtration[h1_regulier[:, 2], h1_regulier[:, 3]] > filtration[h1_regulier[:, 0], h1_regulier[:, 1]]  # Écarte les paires de persistance nulle de H1.
    return PersistencePairs(  # Regroupe les simplexes critiques.
        h0_regulier[garde_0, 0],  # Sommets de naissance de H0.
        h0_regulier[garde_0, 1:3],  # Arêtes de mort de H0.
        np.asarray(h0_essentiel, dtype=np.int64).reshape(-1),  # Composantes essentielles.
        h1_regulier[garde_1, 0:2],  # Arêtes de naissance de H1.
        h1_regulier[garde_1, 2:4],  # Arêtes de mort de H1.
    )  # Termine la construction.


def exact_diagrams(filtration: np.ndarray) -> Dict[int, np.ndarray]:  # Retourne les diagrammes exacts.
    """Retourne les diagrammes exacts, sans aucune valeur inventée ; une classe essentielle garde une mort infinie."""
    paires = compute_flag_pairs(filtration)  # Calcule les simplexes critiques.
    h0_fini = np.stack([filtration[paires.h0_birth, paires.h0_birth], filtration[paires.h0_death[:, 0], paires.h0_death[:, 1]]], axis=1) if paires.h0_birth.size else np.empty((0, 2))  # Lit H0 fini.
    h0_infini = np.stack([filtration[paires.h0_essential, paires.h0_essential], np.full(paires.h0_essential.size, np.inf)], axis=1) if paires.h0_essential.size else np.empty((0, 2))  # Lit H0 essentiel.
    h1 = np.stack([filtration[paires.h1_birth[:, 0], paires.h1_birth[:, 1]], filtration[paires.h1_death[:, 0], paires.h1_death[:, 1]]], axis=1) if paires.h1_birth.size else np.empty((0, 2))  # Lit H1.
    return {0: np.concatenate([h0_fini, h0_infini], axis=0), 1: h1}  # Retourne H0 et H1.


@dataclass  # Génère le constructeur du conteneur.
class PersistenceBatch:  # Regroupe les diagrammes différentiables d'un lot.
    """Diagrammes d'un lot, sous forme de tenseurs complétés accompagnés de masques."""

    h0_birth: torch.Tensor  # Naissances finies de H0 (B, P0).
    h0_death: torch.Tensor  # Morts finies de H0 (B, P0).
    h0_mask: torch.Tensor  # Masque de H0 (B, P0).
    h0_essential: torch.Tensor  # Naissance de la composante essentielle (B,).
    h1_birth: torch.Tensor  # Naissances de H1 (B, P1).
    h1_death: torch.Tensor  # Morts de H1 (B, P1).
    h1_mask: torch.Tensor  # Masque de H1 (B, P1).
    paires: List[PersistencePairs]  # Simplexes critiques de chaque échantillon.


def _completer(listes: Sequence[np.ndarray], colonnes: int) -> Tuple[np.ndarray, np.ndarray]:  # Complète des tableaux de tailles variables.
    """Complète une liste de tableaux par des zéros et retourne le masque correspondant."""
    maximum = max((len(tableau) for tableau in listes), default=0)  # Calcule la plus grande taille.
    maximum = max(maximum, 1)  # Conserve au moins une colonne pour simplifier les calculs.
    sortie = np.zeros((len(listes), maximum, colonnes), dtype=np.int64)  # Prépare le tableau complété.
    masque = np.zeros((len(listes), maximum), dtype=bool)  # Prépare le masque.
    for rang, tableau in enumerate(listes):  # Parcourt les tableaux.
        if len(tableau) > 0:  # Ignore un tableau vide.
            sortie[rang, : len(tableau)] = tableau.reshape(len(tableau), colonnes)  # Copie les valeurs.
            masque[rang, : len(tableau)] = True  # Marque les positions valides.
    return sortie, masque  # Retourne le tableau et le masque.


def differentiable_persistence(filtration: torch.Tensor, masque: Optional[torch.Tensor] = None) -> PersistenceBatch:  # Calcule la persistance d'un lot.
    """Calcule l'appariement exact sans gradient, puis relit les valeurs dans le tenseur différentiable."""
    tableau = filtration.detach().cpu().numpy().astype(np.float64)  # Détache uniquement pour trouver l'appariement.
    taille_lot, nombre, _ = tableau.shape  # Lit les dimensions du lot.
    valides = masque.detach().cpu().numpy() if masque is not None else np.ones((taille_lot, nombre), dtype=bool)  # Lit les positions valides.
    toutes: List[PersistencePairs] = []  # Prépare les paires de chaque échantillon.
    for rang in range(taille_lot):  # Parcourt les échantillons.
        indices = np.flatnonzero(valides[rang])  # Sélectionne les tokens valides.
        paires = compute_flag_pairs(tableau[rang][np.ix_(indices, indices)])  # Calcule la persistance du sous-nuage.
        toutes.append(PersistencePairs(indices[paires.h0_birth], indices[paires.h0_death], indices[paires.h0_essential], indices[paires.h1_birth], indices[paires.h1_death]))  # Revient aux indices d'origine.
    appareil = filtration.device  # Récupère le device du tenseur.
    h0_b, h0_masque = _completer([p.h0_birth for p in toutes], 1)  # Complète les naissances de H0.
    h0_d, _ = _completer([p.h0_death for p in toutes], 2)  # Complète les morts de H0.
    h1_b, h1_masque = _completer([p.h1_birth for p in toutes], 2)  # Complète les naissances de H1.
    h1_d, _ = _completer([p.h1_death for p in toutes], 2)  # Complète les morts de H1.
    lignes = torch.arange(taille_lot, device=appareil).unsqueeze(1)  # Prépare l'index du lot.
    t_h0_b = torch.from_numpy(h0_b).to(appareil)  # Convertit les indices de naissance de H0.
    t_h0_d = torch.from_numpy(h0_d).to(appareil)  # Convertit les indices de mort de H0.
    t_h1_b = torch.from_numpy(h1_b).to(appareil)  # Convertit les indices de naissance de H1.
    t_h1_d = torch.from_numpy(h1_d).to(appareil)  # Convertit les indices de mort de H1.
    m0 = torch.from_numpy(h0_masque).to(appareil)  # Convertit le masque de H0.
    m1 = torch.from_numpy(h1_masque).to(appareil)  # Convertit le masque de H1.
    essentiels = torch.as_tensor([int(p.h0_essential[0]) if p.h0_essential.size else 0 for p in toutes], device=appareil)  # Sélectionne la composante essentielle.
    return PersistenceBatch(  # Relit les valeurs dans le tenseur différentiable.
        filtration[lignes, t_h0_b[..., 0], t_h0_b[..., 0]] * m0,  # Naissances de H0.
        filtration[lignes, t_h0_d[..., 0], t_h0_d[..., 1]] * m0,  # Morts de H0.
        m0,  # Masque de H0.
        filtration[torch.arange(taille_lot, device=appareil), essentiels, essentiels],  # Naissance essentielle.
        filtration[lignes, t_h1_b[..., 0], t_h1_b[..., 1]] * m1,  # Naissances de H1.
        filtration[lignes, t_h1_d[..., 0], t_h1_d[..., 1]] * m1,  # Morts de H1.
        m1,  # Masque de H1.
        toutes,  # Paires de chaque échantillon.
    )  # Termine la construction.


def critical_edges(distances: np.ndarray, dimensions: Sequence[int] = (0, 1)) -> np.ndarray:  # Sélectionne les arêtes critiques.
    """Retourne les arêtes critiques de Vietoris-Rips, de forme (E, 2), d'une matrice de distances."""
    matrice = np.array(distances, dtype=np.float64, copy=True)  # Copie la matrice reçue.
    np.fill_diagonal(matrice, 0.0)  # Impose une naissance nulle pour tous les sommets.
    paires = compute_flag_pairs(matrice)  # Calcule les simplexes critiques.
    morceaux: List[np.ndarray] = []  # Prépare les groupes d'arêtes.
    if 0 in dimensions and paires.h0_death.size:  # Vérifie que H0 est demandé.
        morceaux.append(paires.h0_death)  # Ajoute les arêtes de fusion de composantes.
    if 1 in dimensions and paires.h1_birth.size:  # Vérifie que H1 est demandé.
        morceaux.extend([paires.h1_birth, paires.h1_death])  # Ajoute les arêtes de naissance et de mort des cycles.
    return np.concatenate(morceaux, axis=0) if morceaux else np.empty((0, 2), dtype=np.int64)  # Retourne toutes les arêtes.


def single_linkage_edges(filtration: np.ndarray, masque: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:  # Calcule l'ultramétrique de H0.
    """Retourne, pour chaque paire, l'arête critique qui fixe la hauteur de fusion dans la filtration.

    Cette valeur minimax est l'ultramétrique de liaison simple, qui code exactement la hiérarchie H0
    et reste stable en norme infinie (Carlsson et Mémoli, 2010). Le calcul suit l'algorithme de
    Floyd-Warshall en version minimax, vectorisée sur tout le lot. Accepte (N, N) ou (B, N, N).
    """
    unique = filtration.ndim == 2  # Détecte une matrice isolée.
    u = (filtration[None] if unique else filtration).astype(np.float64, copy=True)  # Copie les hauteurs directes.
    taille_lot, nombre, _ = u.shape  # Lit les dimensions du lot.
    if masque is not None:  # Vérifie la présence d'un masque.
        valides = masque[None] if unique else masque  # Harmonise la forme du masque.
        u[~(valides[:, None, :] & valides[:, :, None])] = np.inf  # Interdit les chemins passant par le padding.
    diagonale = np.arange(nombre)  # Prépare les indices diagonaux.
    u[:, diagonale, diagonale] = -np.inf  # Un sommet est relié à lui-même sans coût.
    critique = np.broadcast_to(np.arange(nombre * nombre).reshape(nombre, nombre), (taille_lot, nombre, nombre)).copy()  # Chaque paire a d'abord pour arête critique son arête directe.
    for k in range(nombre):  # Autorise progressivement le passage par le sommet k.
        vers_k = u[:, :, k, None]  # Lit la hauteur de i vers k.
        depuis_k = u[:, k, None, :]  # Lit la hauteur de k vers j.
        chemin = np.maximum(vers_k, depuis_k)  # Calcule la hauteur du chemin passant par k.
        meilleur = chemin < u  # Détecte les chemins améliorés.
        heritage = np.where(vers_k >= depuis_k, critique[:, :, k, None], critique[:, k, None, :])  # Conserve l'arête qui fixe la hauteur de ce chemin.
        critique = np.where(meilleur, heritage, critique)  # Met à jour les arêtes critiques.
        u = np.where(meilleur, chemin, u)  # Met à jour les hauteurs.
    critique_i, critique_j = np.divmod(critique, nombre)  # Convertit les indices aplatis en sommets.
    return (critique_i[0], critique_j[0]) if unique else (critique_i, critique_j)  # Retourne les arêtes critiques.


def topological_bias(filtration: torch.Tensor, masque: Optional[torch.Tensor] = None) -> torch.Tensor:  # Calcule le biais issu de H0.
    """Calcule le biais T = 1 - u / max(u) à partir de l'ultramétrique de liaison simple, de façon différentiable."""
    tableau = filtration.detach().cpu().numpy()  # Détache uniquement pour trouver les arêtes critiques.
    valides = masque.detach().cpu().numpy() if masque is not None else None  # Lit les positions valides.
    ci, cj = single_linkage_edges(tableau, valides)  # Calcule les arêtes critiques de tout le lot.
    appareil = filtration.device  # Récupère le device du tenseur.
    lignes = torch.arange(filtration.size(0), device=appareil).view(-1, 1, 1)  # Prépare l'index du lot.
    hauteur = filtration[lignes, torch.from_numpy(ci).to(appareil), torch.from_numpy(cj).to(appareil)]  # Relit les hauteurs de façon différentiable.
    identite = torch.eye(filtration.size(1), device=appareil, dtype=filtration.dtype).unsqueeze(0)  # Prépare la diagonale.
    hauteur = hauteur * (1.0 - identite)  # Annule les hauteurs diagonales.
    if masque is not None:  # Vérifie la présence d'un masque.
        hauteur = hauteur * (masque.unsqueeze(1) & masque.unsqueeze(2)).to(filtration.dtype)  # Annule les positions de padding.
    maximum = hauteur.flatten(1).max(dim=1).values.clamp_min(1e-6).view(-1, 1, 1)  # Calcule la hauteur maximale de chaque échantillon.
    return 1.0 - hauteur / maximum  # Une fusion précoce donne un biais proche de 1.


def adaptive_grid(naissance: torch.Tensor, mort: torch.Tensor, masque: torch.Tensor, pas: int) -> torch.Tensor:  # Construit une grille propre à chaque échantillon.
    """Construit une grille (B, M) couvrant les barres de chaque échantillon.

    Une grille globale fixe peut laisser une barre courte entre deux points, ce qui rend le paysage
    aveugle à cette classe. Cette grille reste indépendante des autres échantillons du lot.
    """
    n = naissance.detach().masked_fill(~masque, float("inf")).min(dim=1).values  # Calcule la première naissance valide.
    m = mort.detach().masked_fill(~masque, float("-inf")).max(dim=1).values  # Calcule la dernière mort valide.
    vide = ~masque.any(dim=1)  # Détecte les diagrammes vides.
    n = n.masked_fill(vide, 0.0)  # Fixe un intervalle arbitraire pour un diagramme vide.
    m = m.masked_fill(vide, 1.0)  # Fixe un intervalle arbitraire pour un diagramme vide.
    fractions = torch.linspace(0.0, 1.0, pas, device=naissance.device, dtype=naissance.dtype).unsqueeze(0)  # Prépare les positions relatives.
    return n.unsqueeze(1) + (m - n).unsqueeze(1) * fractions  # Retourne la grille de chaque échantillon.


def persistence_landscape(naissance: torch.Tensor, mort: torch.Tensor, masque: torch.Tensor, grille: torch.Tensor, couches: int) -> torch.Tensor:  # Calcule le paysage de persistance.
    """Calcule les paysages de Bubenik sur une grille (M) ou (B, M) : lambda_k(t) est la k-ième plus grande tente en t."""
    t = grille.view(1, 1, -1) if grille.dim() == 1 else grille.unsqueeze(1)  # Prépare la grille pour le calcul vectoriel.
    tentes = torch.minimum(t - naissance.unsqueeze(-1), mort.unsqueeze(-1) - t).clamp_min(0.0)  # Évalue les tentes sur la grille.
    tentes = tentes * masque.unsqueeze(-1).to(tentes.dtype)  # Annule les positions de complément.
    if tentes.size(1) < couches:  # Vérifie qu'il existe assez de points.
        tentes = F.pad(tentes, (0, 0, 0, couches - tentes.size(1)))  # Ajoute des tentes nulles avant le tri.
    return torch.topk(tentes, k=couches, dim=1).values  # Retourne les k plus grandes tentes en chaque point de la grille.


def landscape_l2_distance(a: PersistenceBatch, b: PersistenceBatch, couches: int, pas: int) -> torch.Tensor:  # Compare deux lots de diagrammes.
    """Retourne le carré de la distance L2 entre paysages, évaluée sur une grille commune à chaque paire (Bubenik, 2015)."""
    total = 0.0  # Initialise la somme des deux dimensions.
    for attribut in ("h0", "h1"):  # Parcourt H0 et H1.
        an, am, ak = getattr(a, f"{attribut}_birth"), getattr(a, f"{attribut}_death"), getattr(a, f"{attribut}_mask")  # Lit les diagrammes du premier lot.
        bn, bm, bk = getattr(b, f"{attribut}_birth"), getattr(b, f"{attribut}_death"), getattr(b, f"{attribut}_mask")  # Lit les diagrammes du second lot.
        grille = adaptive_grid(torch.cat([an, bn], dim=1), torch.cat([am, bm], dim=1), torch.cat([ak, bk], dim=1), pas)  # Construit la grille commune de chaque paire.
        ecart = persistence_landscape(an, am, ak, grille, couches) - persistence_landscape(bn, bm, bk, grille, couches)  # Calcule la différence des paysages.
        pas_grille = (grille[:, 1] - grille[:, 0]).clamp_min(1e-9).view(-1, 1, 1)  # Lit l'écart entre deux points de grille.
        total = total + (ecart.pow(2) * pas_grille).sum(dim=(1, 2)).mean()  # Approche l'intégrale par une somme de Riemann.
    return total  # Retourne la somme sur H0 et H1.


def soft_betti_curve(naissance: torch.Tensor, mort: Optional[torch.Tensor], masque: torch.Tensor, echelles: torch.Tensor, temperature: float) -> torch.Tensor:  # Approxime la courbe de Betti.
    """Approxime le nombre de classes vivantes à chaque échelle ; converge vers la valeur exacte quand la température tend vers 0."""
    e = echelles.view(1, 1, -1)  # Prépare les échelles pour le calcul vectoriel.
    apres_naissance = torch.sigmoid((e - naissance.unsqueeze(-1)) / temperature)  # Mesure si la classe est déjà née.
    avant_mort = torch.sigmoid((mort.unsqueeze(-1) - e) / temperature) if mort is not None else torch.ones_like(apres_naissance)  # Mesure si la classe est encore vivante.
    return (apres_naissance * avant_mort * masque.unsqueeze(-1).to(e.dtype)).sum(dim=1)  # Somme sur toutes les classes.


class TopologicalEncoder(nn.Module):  # Encode les diagrammes en tokens.
    """Convertit les diagrammes H0 et H1 en tokens, sans jamais les détacher du graphe de calcul."""

    PLANCHER_VARIANCE = 2.5e-3  # Borne l'amplification du gradient par la standardisation à un facteur 20 : une variance minuscule ferait exploser le gradient.

    def __init__(self, dim_latente: int, nb_h0: int, nb_h1: int, couches: int, pas: int, nb_echelles: int, temperature: float, vues: Sequence[str] = ("image", "text", "fusion")) -> None:  # Construit l'encodeur.
        """Crée les projecteurs vers les tokens ; la grille des paysages dépend de chaque échantillon."""
        super().__init__()  # Initialise le module parent.
        self.dim_latente = dim_latente  # Conserve la dimension latente.
        self.nb_h0 = nb_h0  # Conserve le nombre de tokens H0.
        self.nb_h1 = nb_h1  # Conserve le nombre de tokens H1.
        self.couches = couches  # Conserve le nombre de couches de paysage.
        self.temperature = temperature  # Conserve la température de Betti.
        self.pas = pas  # Conserve le nombre de points de grille.
        dim_h0 = couches * pas + nb_echelles + 5  # Calcule la taille du descripteur de H0.
        dim_h1 = couches * pas + nb_echelles + 5  # Calcule la taille du descripteur de H1.
        self.projection_h0 = nn.Sequential(nn.Linear(dim_h0, dim_latente), nn.LayerNorm(dim_latente), nn.GELU(), nn.Linear(dim_latente, nb_h0 * dim_latente))  # Crée le projecteur de H0.
        self.projection_h1 = nn.Sequential(nn.Linear(dim_h1, dim_latente), nn.LayerNorm(dim_latente), nn.GELU(), nn.Linear(dim_latente, nb_h1 * dim_latente))  # Crée le projecteur de H1.
        self.normalisation = nn.LayerNorm(dim_latente)  # Normalise les tokens produits.
        self.standard_h0 = nn.ModuleDict({vue: nn.BatchNorm1d(dim_h0, affine=False, eps=self.PLANCHER_VARIANCE) for vue in vues})  # Standardise les descripteurs de H0, séparément pour chaque vue.
        self.standard_h1 = nn.ModuleDict({vue: nn.BatchNorm1d(dim_h1, affine=False, eps=self.PLANCHER_VARIANCE) for vue in vues})  # Standardise les descripteurs de H1, séparément pour chaque vue.

    def _statistiques(self, naissance: torch.Tensor, mort: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:  # Résume un diagramme.
        """Calcule la persistance totale et maximale, le nombre de classes et l'intervalle couvert."""
        duree = (mort - naissance) * masque.to(naissance.dtype)  # Calcule les durées de vie valides.
        total = duree.sum(dim=1)  # Calcule la persistance totale.
        maximale = duree.max(dim=1).values  # Calcule la persistance maximale.
        compte = torch.log1p(masque.to(naissance.dtype).sum(dim=1))  # Calcule le logarithme du nombre de classes.
        debut = naissance.masked_fill(~masque, float("inf")).min(dim=1).values  # Calcule la première naissance.
        fin = mort.masked_fill(~masque, float("-inf")).max(dim=1).values  # Calcule la dernière mort.
        debut = torch.where(torch.isfinite(debut), debut, torch.zeros_like(debut))  # Remplace la valeur d'un diagramme vide par 0.
        fin = torch.where(torch.isfinite(fin), fin, torch.zeros_like(fin))  # Remplace la valeur d'un diagramme vide par 0.
        return torch.stack([total, maximale, compte, debut, fin], dim=1)  # Conserve l'échelle absolue perdue par la grille adaptative.

    def descriptors(self, diagrammes: PersistenceBatch, echelles: torch.Tensor) -> Dict[str, torch.Tensor]:  # Calcule les descripteurs.
        """Calcule les paysages, les courbes de Betti douces et les statistiques des diagrammes."""
        g0 = adaptive_grid(diagrammes.h0_birth, diagrammes.h0_death, diagrammes.h0_mask, self.pas)  # Construit la grille de H0.
        g1 = adaptive_grid(diagrammes.h1_birth, diagrammes.h1_death, diagrammes.h1_mask, self.pas)  # Construit la grille de H1.
        p0 = persistence_landscape(diagrammes.h0_birth, diagrammes.h0_death, diagrammes.h0_mask, g0, self.couches)  # Calcule le paysage de H0.
        p1 = persistence_landscape(diagrammes.h1_birth, diagrammes.h1_death, diagrammes.h1_mask, g1, self.couches)  # Calcule le paysage de H1.
        essentiel = diagrammes.h0_essential.unsqueeze(1)  # Prépare la composante essentielle.
        b0_fini = soft_betti_curve(diagrammes.h0_birth, diagrammes.h0_death, diagrammes.h0_mask, echelles, self.temperature)  # Compte les composantes finies vivantes.
        b0_essentiel = soft_betti_curve(essentiel, None, torch.ones_like(essentiel, dtype=torch.bool), echelles, self.temperature)  # Compte la composante essentielle.
        b1 = soft_betti_curve(diagrammes.h1_birth, diagrammes.h1_death, diagrammes.h1_mask, echelles, self.temperature)  # Compte les cycles vivants.
        return {  # Retourne les descripteurs.
            "paysage_h0": p0,  # Paysage de H0.
            "paysage_h1": p1,  # Paysage de H1.
            "betti_h0": b0_fini + b0_essentiel,  # Courbe de Betti douce de H0.
            "betti_h1": b1,  # Courbe de Betti douce de H1.
            "stats_h0": self._statistiques(diagrammes.h0_birth, diagrammes.h0_death, diagrammes.h0_mask),  # Statistiques de H0.
            "stats_h1": self._statistiques(diagrammes.h1_birth, diagrammes.h1_death, diagrammes.h1_mask),  # Statistiques de H1.
        }  # Termine le dictionnaire.

    def forward(self, diagrammes: PersistenceBatch, echelles: torch.Tensor, vue: str = "image") -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:  # Encode les diagrammes.
        """Retourne les tokens H0 (B, K0, d), les tokens H1 (B, K1, d) et les descripteurs.

        Les descripteurs varient peu d'un échantillon à l'autre : les standardiser rend cette variation
        exploitable. À l'évaluation, les statistiques apprises rendent le résultat indépendant du lot.
        """
        descripteurs = self.descriptors(diagrammes, echelles)  # Calcule les descripteurs.
        taille_lot = diagrammes.h0_birth.size(0)  # Lit la taille du lot.
        x0 = torch.cat([descripteurs["paysage_h0"].flatten(1), descripteurs["betti_h0"], descripteurs["stats_h0"]], dim=1)  # Assemble le descripteur de H0.
        x1 = torch.cat([descripteurs["paysage_h1"].flatten(1), descripteurs["betti_h1"], descripteurs["stats_h1"]], dim=1)  # Assemble le descripteur de H1.
        t0 = self.normalisation(self.projection_h0(self.standard_h0[vue](x0)).view(taille_lot, self.nb_h0, self.dim_latente))  # Produit les tokens H0.
        t1 = self.normalisation(self.projection_h1(self.standard_h1[vue](x1)).view(taille_lot, self.nb_h1, self.dim_latente))  # Produit les tokens H1.
        return t0, t1, descripteurs  # Retourne les tokens et les descripteurs.
