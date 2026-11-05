# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Métriques de recherche inter-modale, de structure et de comparaison statistique."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from typing import Dict, Sequence, Tuple  # Importe les types des signatures.

import numpy as np  # Effectue les calculs matriciels.
from sklearn.manifold import trustworthiness as _trustworthiness  # Fournit la mesure de fiabilité des voisinages.


def _classer(scores: np.ndarray, pertinence: np.ndarray, graine: int) -> np.ndarray:  # Classe les éléments de chaque requête.
    """Retourne la pertinence triée par score décroissant ; un bruit infime départage les égalités au hasard.

    Sans ce bruit, un modèle effondré dont tous les scores sont égaux obtiendrait un classement
    arbitraire favorable à certains indices, ce qui fausserait les résultats.
    """
    bruit = np.random.default_rng(graine).random(scores.shape) * 1e-10  # Tire le bruit de départage, très inférieur à la précision des scores.
    ordre = np.argsort(-(scores.astype(np.float64) + bruit), axis=1)  # Trie les scores en ordre décroissant.
    return np.take_along_axis(pertinence, ordre, axis=1)  # Réordonne la pertinence.


def _par_requete(trie: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # Calcule les mesures par requête.
    """Retourne le rang du premier élément pertinent (base 0) et la précision moyenne de chaque requête."""
    premier = trie.argmax(axis=1)  # Trouve la position du premier élément pertinent.
    positions = np.arange(1, trie.shape[1] + 1, dtype=np.float64)  # Prépare les rangs en base 1.
    precision = np.cumsum(trie, axis=1, dtype=np.float64) / positions  # Calcule la précision à chaque rang.
    ap = (precision * trie).sum(axis=1) / trie.sum(axis=1).clip(min=1)  # Calcule la précision moyenne.
    return premier, ap  # Retourne les deux mesures.


def retrieval_per_query(similarites: np.ndarray, pertinence: np.ndarray, graine: int = 0) -> Dict[str, np.ndarray]:  # Calcule les mesures brutes.
    """Calcule, pour les deux directions, le rang et la précision moyenne de chaque requête."""
    rangs_it, ap_it = _par_requete(_classer(similarites, pertinence, graine))  # Traite la direction image vers texte.
    rangs_ti, ap_ti = _par_requete(_classer(similarites.T, pertinence.T, graine + 1))  # Traite la direction texte vers image.
    return {"rangs_i2t": rangs_it, "ap_i2t": ap_it, "rangs_t2i": rangs_ti, "ap_t2i": ap_ti}  # Retourne les mesures.


def chance_level(pertinence: np.ndarray, k: int) -> Dict[str, float]:  # Calcule le niveau du hasard.
    """Retourne le Recall@k attendu d'un classement aléatoire, pour situer les résultats."""
    nb_images, nb_legendes = pertinence.shape  # Lit les dimensions.
    pertinents_it = float(pertinence.sum(axis=1).mean())  # Calcule le nombre moyen de légendes pertinentes.
    pertinents_ti = float(pertinence.sum(axis=0).mean())  # Calcule le nombre moyen d'images pertinentes.
    i2t = 1.0 - np.prod([(nb_legendes - pertinents_it - j) / (nb_legendes - j) for j in range(k)])  # Calcule la probabilité d'au moins un succès.
    t2i = 1.0 - np.prod([(nb_images - pertinents_ti - j) / (nb_images - j) for j in range(k)])  # Calcule la probabilité d'au moins un succès.
    return {"i2t": float(i2t), "t2i": float(t2i)}  # Retourne les deux niveaux.


def summarize_retrieval(mesures: Dict[str, np.ndarray], ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:  # Résume les mesures brutes.
    """Résume les mesures par requête : Recall@k, mAP et rang médian, dans les deux directions."""
    resume: Dict[str, float] = {}  # Prépare le résumé.
    for sens in ("i2t", "t2i"):  # Parcourt les deux directions.
        rangs = mesures[f"rangs_{sens}"]  # Lit les rangs de cette direction.
        for k in ks:  # Parcourt les rangs de coupure.
            resume[f"{sens}_recall_at_{k}"] = float((rangs < k).mean())  # Calcule le Recall@k.
        resume[f"{sens}_map"] = float(mesures[f"ap_{sens}"].mean())  # Calcule le mAP.
        resume[f"{sens}_median_rank"] = float(np.median(rangs) + 1.0)  # Calcule le rang médian en base 1.
    for k in ks:  # Parcourt les rangs de coupure.
        resume[f"mean_recall_at_{k}"] = 0.5 * (resume[f"i2t_recall_at_{k}"] + resume[f"t2i_recall_at_{k}"])  # Moyenne les deux directions.
    resume["mean_map"] = 0.5 * (resume["i2t_map"] + resume["t2i_map"])  # Moyenne les mAP.
    return resume  # Retourne le résumé.


def retrieval_metrics(images: np.ndarray, textes: np.ndarray, pertinence: np.ndarray, ks: Sequence[int] = (1, 5, 10), graine: int = 0) -> Dict[str, float]:  # Calcule les métriques de recherche.
    """Calcule toutes les métriques de recherche à partir d'embeddings normalisés."""
    return summarize_retrieval(retrieval_per_query(np.matmul(images, textes.T), pertinence, graine), ks)  # Retourne le résumé.


def bootstrap_ci(valeurs: np.ndarray, echantillons: int = 1000, graine: int = 0, niveau: float = 0.95) -> Tuple[float, float]:  # Calcule un intervalle de confiance.
    """Retourne l'intervalle de confiance par bootstrap de la moyenne des requêtes."""
    generateur = np.random.default_rng(graine)  # Crée le générateur.
    tirages = generateur.integers(0, len(valeurs), size=(echantillons, len(valeurs)))  # Tire les rééchantillonnages.
    moyennes = valeurs[tirages].mean(axis=1)  # Calcule la moyenne de chaque tirage.
    alpha = (1.0 - niveau) / 2.0  # Calcule le quantile de chaque côté.
    return float(np.quantile(moyennes, alpha)), float(np.quantile(moyennes, 1.0 - alpha))  # Retourne les bornes.


def paired_bootstrap_difference(a: np.ndarray, b: np.ndarray, echantillons: int = 2000, graine: int = 0) -> Dict[str, float]:  # Compare deux modèles.
    """Compare deux modèles sur les mêmes requêtes : les tirages sont appariés pour réduire la variance."""
    ecart = a.astype(np.float64) - b.astype(np.float64)  # Calcule la différence requête par requête.
    bas, haut = bootstrap_ci(ecart, echantillons, graine)  # Calcule l'intervalle de la différence.
    return {"difference": float(ecart.mean()), "ic95_bas": bas, "ic95_haut": haut, "significatif": bool(bas > 0.0 or haut < 0.0)}  # Retourne la comparaison.


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:  # Calcule les distances par paires.
    """Retourne la matrice des distances euclidiennes d'un nuage de points."""
    produit = np.matmul(x, x.T)  # Calcule les produits scalaires.
    diagonale = np.diag(produit)  # Lit les normes au carré.
    return np.sqrt(np.clip(diagonale[:, None] + diagonale[None, :] - 2.0 * produit, 0.0, None))  # Retourne les distances.


def _triangle(d: np.ndarray) -> np.ndarray:  # Extrait les paires distinctes.
    """Retourne le triangle supérieur strict d'une matrice carrée."""
    return d[np.triu_indices(d.shape[0], k=1)]  # Retourne les valeurs des paires distinctes.


def mantel_r(a: np.ndarray, b: np.ndarray) -> float:  # Calcule la corrélation de Mantel.
    """Calcule la corrélation de Pearson entre deux matrices de distances, hors diagonale."""
    x = _triangle(a)  # Extrait les distances de la première matrice.
    y = _triangle(b)  # Extrait les distances de la seconde matrice.
    if x.std() < 1e-12 or y.std() < 1e-12:  # Détecte une matrice constante, donc une corrélation non définie.
        return 0.0  # Retourne une absence de corrélation.
    return float(np.corrcoef(x, y)[0, 1])  # Retourne la corrélation.


def kruskal_stress(reference: np.ndarray, latente: np.ndarray) -> float:  # Calcule le stress de Kruskal.
    """Calcule le stress-1 de Kruskal avec échelle optimale : il vaut 0 pour une isométrie à un facteur près."""
    x = _triangle(reference)  # Extrait les distances de référence.
    y = _triangle(latente)  # Extrait les distances latentes.
    if float((y * y).sum()) < 1e-12:  # Détecte un nuage entièrement effondré.
        return 1.0  # Retourne la valeur maximale.
    echelle = float((x * y).sum() / max((x * x).sum(), 1e-12))  # Calcule le facteur optimal.
    return float(np.sqrt(((y - echelle * x) ** 2).sum() / (y * y).sum()))  # Retourne le stress.


def neighborhood_preservation(reference: np.ndarray, latente: np.ndarray, k: int = 10) -> float:  # Mesure la préservation des voisinages.
    """Retourne la fraction moyenne des k plus proches voisins conservés."""
    a = reference.copy()  # Copie la première matrice.
    b = latente.copy()  # Copie la seconde matrice.
    np.fill_diagonal(a, np.inf)  # Exclut chaque point de ses propres voisins.
    np.fill_diagonal(b, np.inf)  # Exclut chaque point de ses propres voisins.
    voisins_a = np.argpartition(a, k, axis=1)[:, :k]  # Calcule les voisins de référence.
    voisins_b = np.argpartition(b, k, axis=1)[:, :k]  # Calcule les voisins latents.
    return float(np.mean([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(voisins_a, voisins_b)]))  # Retourne la fraction moyenne.


def trustworthiness_score(reference: np.ndarray, latente: np.ndarray, k: int = 10) -> float:  # Calcule la fiabilité des voisinages.
    """Calcule la fiabilité : les voisins du latent étaient-ils déjà voisins dans la référence ?"""
    return float(_trustworthiness(reference, latente, n_neighbors=k))  # Retourne la mesure de scikit-learn.
