# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Mesures topologiques exactes, utilisées pour l'évaluation et non pour l'optimisation."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from typing import Dict, List, Optional  # Importe les types des signatures.

import gudhi  # Fournit la distance de bottleneck.
import gudhi.hera  # Fournit la distance de Wasserstein exacte, sans dépendance à POT.
import numpy as np  # Effectue les calculs matriciels.
import ripser  # Calcule rapidement la persistance de grands nuages.
import torch  # Fournit les tenseurs.

from data.feature_cache import FeatureSplit  # Importe l'accès aux caractéristiques.
from models.topology import exact_diagrams  # Importe les diagrammes exacts.
from utils.metrics import kruskal_stress, mantel_r, neighborhood_preservation, pairwise_euclidean, trustworthiness_score  # Importe les métriques de structure.


def rips_diagrams(distances: np.ndarray, dimension_max: int = 1) -> List[np.ndarray]:  # Calcule les diagrammes d'un nuage.
    """Calcule les diagrammes de Vietoris-Rips exacts d'une matrice de distances avec Ripser."""
    return ripser.ripser(distances, distance_matrix=True, maxdim=dimension_max)["dgms"]  # Retourne H0 et H1.


def finite_points(diagramme: np.ndarray) -> np.ndarray:  # Retire les classes essentielles.
    """Retourne les points de mort finie d'un diagramme."""
    return diagramme[np.isfinite(diagramme[:, 1])] if diagramme.size else np.empty((0, 2))  # Filtre les morts infinies.


def diagram_distances(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:  # Compare deux diagrammes.
    """Calcule les distances de bottleneck et de Wasserstein-1 entre les parties finies de deux diagrammes."""
    pa = finite_points(a)  # Garde les points finis du premier diagramme.
    pb = finite_points(b)  # Garde les points finis du second diagramme.
    return {"bottleneck": float(gudhi.bottleneck_distance(pa, pb)), "wasserstein_1": float(gudhi.hera.wasserstein_distance(pa, pb, order=1.0, internal_p=np.inf))}  # Retourne les deux distances.


def diagram_statistics(diagramme: np.ndarray) -> Dict[str, float]:  # Résume un diagramme.
    """Retourne le nombre de classes, la persistance totale et la persistance maximale."""
    points = finite_points(diagramme)  # Garde les points finis.
    duree = points[:, 1] - points[:, 0] if points.size else np.zeros(0)  # Calcule les durées de vie.
    return {"nombre": int(points.shape[0]), "persistance_totale": float(duree.sum()), "persistance_maximale": float(duree.max()) if duree.size else 0.0}  # Retourne les statistiques.


def _normaliser(d: np.ndarray) -> np.ndarray:  # Normalise une matrice de distances.
    """Divise une matrice de distances par sa moyenne hors diagonale, pour comparer des espaces d'échelles différentes."""
    return d / max(float(d[np.triu_indices(d.shape[0], k=1)].mean()), 1e-12)  # Retourne la matrice sans unité.


def structure_report(z_image: np.ndarray, z_texte: np.ndarray, x_image: np.ndarray, x_texte: np.ndarray) -> Dict[str, float]:  # Mesure la préservation de structure.
    """Mesure la préservation géométrique et topologique, de l'entrée vers le latent et entre modalités."""
    rapport: Dict[str, float] = {}  # Prépare le rapport.
    d_zi = pairwise_euclidean(z_image)  # Calcule les distances latentes des images.
    d_zt = pairwise_euclidean(z_texte)  # Calcule les distances latentes des textes.
    d_xi = pairwise_euclidean(x_image)  # Calcule les distances d'entrée des images.
    d_xt = pairwise_euclidean(x_texte)  # Calcule les distances d'entrée des textes.
    for nom, entree, latent, brut, embedding in (("image", d_xi, d_zi, x_image, z_image), ("texte", d_xt, d_zt, x_texte, z_texte)):  # Parcourt les modalités.
        rapport[f"mantel_entree_latent_{nom}"] = mantel_r(entree, latent)  # Mesure la corrélation des distances.
        rapport[f"stress_entree_latent_{nom}"] = kruskal_stress(entree, latent)  # Mesure l'écart à une isométrie à l'échelle près.
        rapport[f"voisinage10_entree_latent_{nom}"] = neighborhood_preservation(entree, latent, 10)  # Mesure la préservation des voisinages.
        rapport[f"fiabilite10_entree_latent_{nom}"] = trustworthiness_score(brut, embedding, 10)  # Mesure la fiabilité des voisinages.
    rapport["mantel_image_texte"] = mantel_r(d_zi, d_zt)  # Mesure l'accord des géométries latentes.
    diagrammes = {cle: rips_diagrams(_normaliser(d)) for cle, d in (("zi", d_zi), ("zt", d_zt), ("xi", d_xi), ("xt", d_xt))}  # Calcule tous les diagrammes exacts.
    for dimension in (0, 1):  # Parcourt H0 et H1.
        for nom, gauche, droite in (("image_texte_latent", "zi", "zt"), ("entree_latent_image", "xi", "zi"), ("entree_latent_texte", "xt", "zt")):  # Parcourt les comparaisons.
            distances = diagram_distances(diagrammes[gauche][dimension], diagrammes[droite][dimension])  # Compare les diagrammes.
            rapport[f"h{dimension}_bottleneck_{nom}"] = distances["bottleneck"]  # Enregistre la distance de bottleneck.
            rapport[f"h{dimension}_wasserstein1_{nom}"] = distances["wasserstein_1"]  # Enregistre la distance de Wasserstein.
    return rapport  # Retourne le rapport.


@torch.no_grad()  # Désactive le calcul du gradient.
def token_cloud_filtrations(modele: torch.nn.Module, split: FeatureSplit, modalite: str, nombre: int, appareil: torch.device) -> Optional[Dict[str, np.ndarray]]:  # Lit les filtrations.
    """Retourne, pour les premiers échantillons, les points et les filtrations des nuages de tokens."""
    if not (modele.variante.tokens_topo or modele.variante.attention_topo) or not modele.variante.banque:  # Vérifie que la variante produit ces nuages.
        return None  # Signale l'absence de nuage.
    if modalite == "image":  # Traite les images.
        entree = {"image": (torch.from_numpy(np.asarray(split.images[:nombre], dtype=np.float32)).to(appareil), None)}  # Prépare les tokens d'images.
    else:  # Traite les textes.
        premieres = [int(liste[0]) for liste in split.legendes_par_image[:nombre]]  # Sélectionne la première légende de chaque image.
        entree = {"text": (torch.from_numpy(np.asarray(split.textes[premieres], dtype=np.float32)).to(appareil), torch.from_numpy(np.asarray(split.masques_texte[premieres], dtype=bool)).to(appareil))}  # Prépare les tokens de textes.
    modele.eval()  # Désactive le dropout.
    vue = modele(entree, fusionner=False)["views"][modalite]  # Exécute la voie structurelle.
    return {"points": vue.points.cpu().numpy(), "filtrations": vue.filtration.cpu().numpy().astype(np.float64)}  # Retourne les points et les filtrations.


def token_stability(propres: Dict[str, np.ndarray], perturbees: Dict[str, np.ndarray], borne: float) -> Dict[str, float]:  # Vérifie la stabilité.
    """Mesure la distance de bottleneck entre diagrammes de tokens, et vérifie la borne de stabilité démontrée."""
    distances = {0: [], 1: []}  # Prépare les distances de H0 et H1.
    deplacements: List[float] = []  # Prépare les déplacements des points.
    violations = 0  # Initialise le compteur de violations.
    rapports: List[float] = []  # Prépare les rapports entre distance et déplacement.
    for u, v, f, g in zip(propres["points"], perturbees["points"], propres["filtrations"], perturbees["filtrations"]):  # Parcourt les échantillons.
        delta = float(np.linalg.norm(u - v, axis=-1).max())  # Calcule le déplacement maximal d'un point.
        deplacements.append(delta)  # Enregistre le déplacement.
        a = exact_diagrams(f)  # Calcule les diagrammes propres.
        b = exact_diagrams(g)  # Calcule les diagrammes perturbés.
        for dimension in (0, 1):  # Parcourt H0 et H1.
            distance = float(gudhi.bottleneck_distance(finite_points(a[dimension]), finite_points(b[dimension])))  # Calcule la distance de bottleneck.
            distances[dimension].append(distance)  # Enregistre la distance.
            if distance > borne * delta + 1e-4:  # Vérifie la borne démontrée.
                violations += 1  # Compte la violation.
            if delta > 1e-9:  # Évite une division par zéro.
                rapports.append(distance / delta)  # Enregistre le rapport.
    return {  # Retourne le résumé.
        "bottleneck_h0_moyen": float(np.mean(distances[0])),  # Distance moyenne de H0.
        "bottleneck_h1_moyen": float(np.mean(distances[1])),  # Distance moyenne de H1.
        "deplacement_moyen": float(np.mean(deplacements)),  # Déplacement moyen.
        "rapport_maximal_observe": float(np.max(rapports)) if rapports else 0.0,  # Rapport maximal observé.
        "borne_theorique": float(borne),  # Borne théorique.
        "violations_de_la_borne": int(violations),  # Nombre de violations.
    }  # Termine le résumé.
