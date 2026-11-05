# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Figures scientifiques : chaque figure lit des résultats mesurés et n'invente aucune valeur.

Les nuages projetés (ACP, t-SNE) sont des illustrations qualitatives. La preuve quantitative
se trouve dans les tableaux de docs/RESULTATS.md.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import textwrap  # Découpe les légendes trop longues.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Dict, List, Optional, Sequence, Tuple  # Importe les types des signatures.

import matplotlib  # Configure le moteur de rendu.

matplotlib.use("Agg")  # Sélectionne un rendu sans écran, compatible avec les serveurs.

import matplotlib.pyplot as plt  # Fournit l'interface de tracé.
import numpy as np  # Effectue les calculs matriciels.
from sklearn.decomposition import PCA  # Fournit l'analyse en composantes principales.
from sklearn.manifold import MDS, TSNE  # Fournit les projections non linéaires.

SURFACE = "#fcfcfb"  # Définit la couleur de fond.
ENCRE = "#0b0b0b"  # Définit la couleur du texte principal.
ENCRE_2 = "#52514e"  # Définit la couleur du texte secondaire.
MUTE = "#898781"  # Définit la couleur des axes et des annotations.
GRILLE = "#e1e0d9"  # Définit la couleur de la grille.
AXE = "#c3c2b7"  # Définit la couleur de la ligne de base.
COULEURS = {"B0": "#6250d6", "B1": "#2a78d6", "B2": "#eb6834", "B3": "#1baf7a", "B4": "#eda100", "CLIP": "#e87ba4"}  # Fixe une couleur par modèle ; la couleur suit le modèle, jamais son rang.
BLEU = "#2a78d6"  # Fixe la couleur de la modalité image et de la valeur principale.
ORANGE = "#eb6834"  # Fixe la couleur de la modalité texte et de la valeur secondaire.
ETIQUETTES = {"image": "Image", "text": "Texte"}  # Associe les modalités à leurs noms affichés.


def appliquer_style() -> None:  # Applique le style commun.
    """Applique un style sobre et homogène à toutes les figures."""
    plt.rcParams.update({  # Met à jour les paramètres de Matplotlib.
        "figure.facecolor": SURFACE,  # Définit le fond de la figure.
        "axes.facecolor": SURFACE,  # Définit le fond des axes.
        "savefig.facecolor": SURFACE,  # Définit le fond à l'enregistrement.
        "axes.edgecolor": AXE,  # Définit la couleur des axes.
        "axes.labelcolor": ENCRE_2,  # Définit la couleur des titres d'axes.
        "axes.titlecolor": ENCRE,  # Définit la couleur des titres.
        "axes.titlesize": 11,  # Définit la taille des titres.
        "axes.titleweight": "bold",  # Met les titres en gras.
        "axes.labelsize": 9.5,  # Définit la taille des titres d'axes.
        "axes.grid": True,  # Active la grille.
        "axes.axisbelow": True,  # Place la grille sous les données.
        "axes.spines.top": False,  # Retire le cadre supérieur.
        "axes.spines.right": False,  # Retire le cadre droit.
        "grid.color": GRILLE,  # Définit la couleur de la grille.
        "grid.linewidth": 0.8,  # Affine la grille.
        "xtick.color": MUTE,  # Définit la couleur des graduations horizontales.
        "ytick.color": MUTE,  # Définit la couleur des graduations verticales.
        "xtick.labelsize": 8.5,  # Définit la taille des graduations horizontales.
        "ytick.labelsize": 8.5,  # Définit la taille des graduations verticales.
        "legend.frameon": False,  # Retire le cadre de la légende.
        "legend.fontsize": 8.5,  # Définit la taille de la légende.
        "text.color": ENCRE,  # Définit la couleur du texte.
        "font.family": "DejaVu Sans",  # Fixe une police disponible partout.
        "lines.linewidth": 2.0,  # Définit l'épaisseur des courbes.
    })  # Termine la mise à jour.


def enregistrer(figure: plt.Figure, chemin: Path) -> None:  # Enregistre une figure.
    """Enregistre une figure en PNG et libère sa mémoire."""
    chemin.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier si nécessaire.
    figure.savefig(chemin, dpi=170, bbox_inches="tight")  # Écrit l'image.
    plt.close(figure)  # Libère la mémoire.
    print(f"Figure enregistrée : {chemin}", flush=True)  # Annonce l'écriture.


def echantillons_bruts(images: Sequence, legendes: Sequence[str], chemin: Path) -> None:  # Trace les échantillons.
    """Affiche de vraies paires image-légende, pour vérifier la correspondance à l'œil."""
    colonnes = 4  # Fixe le nombre de colonnes.
    lignes = int(np.ceil(len(images) / colonnes))  # Calcule le nombre de lignes.
    figure, axes = plt.subplots(lignes, colonnes, figsize=(11, 3.3 * lignes))  # Crée la grille.
    for axe, image, legende in zip(np.ravel(axes), images, legendes):  # Parcourt les paires.
        axe.imshow(image)  # Affiche l'image réelle.
        axe.set_title("\n".join(textwrap.wrap(legende, 34)), fontsize=8.5, fontweight="normal", color=ENCRE_2)  # Affiche la légende réelle.
        axe.axis("off")  # Masque les axes.
    figure.suptitle("Paires réelles image-légende (COCO train2014, légendes du jeu cible)", fontsize=12, fontweight="bold")  # Ajoute le titre général.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def _courbe(axe: plt.Axes, series: List[List[float]], etiquette: str, couleur: str, abscisses: Optional[np.ndarray] = None, style: str = "-") -> None:  # Trace une courbe moyenne.
    """Trace la moyenne d'une série sur les graines, et une bande de ± un écart-type."""
    longueur = min(len(s) for s in series)  # Aligne les graines arrêtées à des époques différentes.
    tableau = np.asarray([s[:longueur] for s in series], dtype=np.float64)  # Assemble les séries.
    x = np.arange(longueur) if abscisses is None else abscisses[:longueur]  # Calcule les abscisses.
    moyenne = tableau.mean(axis=0)  # Calcule la moyenne.
    axe.plot(x, moyenne, color=couleur, label=etiquette, linestyle=style)  # Trace la moyenne.
    if tableau.shape[0] > 1:  # Vérifie la présence de plusieurs graines.
        ecart = tableau.std(axis=0, ddof=1)  # Calcule l'écart-type.
        axe.fill_between(x, moyenne - ecart, moyenne + ecart, color=couleur, alpha=0.12, linewidth=0)  # Trace la bande.


def _marquer_phases(axe: plt.Axes, historique: Dict) -> None:  # Marque les phases.
    """Marque les débuts de phase, car l'objectif change et les niveaux de perte ne sont pas comparables."""
    phases = [e["phase"] for e in historique["train"]]  # Lit la phase de chaque époque.
    for rang in range(1, len(phases)):  # Parcourt les époques.
        if phases[rang] != phases[rang - 1]:  # Détecte un changement de phase.
            axe.axvline(rang, color=MUTE, linewidth=1.0, linestyle=":")  # Trace le repère.
            axe.annotate(f"phase {phases[rang]}", (rang, 1.0), xycoords=("data", "axes fraction"), fontsize=8, color=MUTE, xytext=(3, -10), textcoords="offset points")  # Nomme la phase.


def courbes_perte(historiques: Dict[str, List[Dict]], chemin: Path) -> None:  # Trace les pertes.
    """Trace les pertes d'entraînement et de validation : objectif de B4, et terme sémantique commun."""
    figure, (gauche, droite) = plt.subplots(1, 2, figsize=(12, 4.2))  # Crée les deux panneaux.
    if "B4" in historiques:  # Vérifie la présence de B4.
        h = historiques["B4"]  # Lit les historiques de B4.
        _courbe(gauche, [[e["total"] for e in r["train"]] for r in h], "Entraînement", BLEU)  # Trace l'objectif d'entraînement.
        _courbe(gauche, [[e["perte_total"] for e in r["val"][1:]] for r in h], "Validation", ORANGE, style="--")  # Trace l'objectif de validation.
        _marquer_phases(gauche, h[0])  # Marque les phases.
    gauche.set_title("B4 : objectif composite")  # Ajoute le titre.
    gauche.set_xlabel("Époque")  # Nomme l'axe horizontal.
    gauche.set_ylabel("Perte (les niveaux changent avec la phase)")  # Nomme l'axe vertical.
    gauche.legend(loc="upper right")  # Ajoute la légende.
    for nom in ("B1", "B2", "B3", "B4"):  # Parcourt les variantes.
        if nom in historiques:  # Vérifie la présence de la variante.
            _courbe(droite, [[e["L_sem"] for e in r["train"]] for r in historiques[nom]], f"{nom} entraînement", COULEURS[nom])  # Trace le terme sémantique d'entraînement.
            _courbe(droite, [[e["perte_L_sem"] for e in r["val"][1:]] for r in historiques[nom]], f"{nom} validation", COULEURS[nom], style="--")  # Trace le terme sémantique de validation.
    droite.set_title("Terme commun L_sem (InfoNCE)")  # Ajoute le titre.
    droite.set_xlabel("Époque")  # Nomme l'axe horizontal.
    droite.set_ylabel("L_sem")  # Nomme l'axe vertical.
    droite.legend(ncol=2, loc="upper right")  # Ajoute la légende.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def courbes_recall(historiques: Dict[str, List[Dict]], chemin: Path, clip: Optional[Dict[str, float]] = None) -> None:  # Trace les rappels.
    """Trace Recall@1 et Recall@5 sur la validation, y compris avant tout entraînement."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)  # Crée les deux panneaux.
    for axe, cle, titre in zip(axes, ("mean_recall_at_1", "mean_recall_at_5"), ("Recall@1 moyen (validation)", "Recall@5 moyen (validation)")):  # Parcourt les métriques.
        for nom in ("B0", "B1", "B2", "B3", "B4"):  # Parcourt les variantes.
            if nom in historiques:  # Vérifie la présence de la variante.
                series = [[e[cle] * 100.0 for e in r["val"]] for r in historiques[nom]]  # Lit les valeurs en pourcentage.
                _courbe(axe, series, nom, COULEURS[nom], abscisses=np.arange(-1, 200))  # Trace la courbe.
                longueur = min(len(s) for s in series)  # Calcule la longueur commune.
                axe.annotate(nom, (longueur - 2, float(np.mean([s[longueur - 1] for s in series]))), fontsize=8.5, color=ENCRE_2, xytext=(5, 0), textcoords="offset points", va="center")  # Ajoute l'étiquette directe.
        axe.set_title(titre)  # Ajoute le titre.
        axe.set_xlabel("Époque (−1 : avant entraînement)")  # Nomme l'axe horizontal.
        axe.set_ylabel("%")  # Nomme l'axe vertical.
        axe.set_ylim(bottom=0)  # Fixe le bas de l'axe à zéro.
        axe.legend(loc="lower right")  # Ajoute la légende.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def _projeter(z: np.ndarray, methode: str, graine: int = 0) -> np.ndarray:  # Projette un nuage en deux dimensions.
    """Projette un nuage en deux dimensions par ACP ou par t-SNE."""
    if methode == "acp":  # Vérifie la méthode demandée.
        return PCA(n_components=2, random_state=graine).fit_transform(z)  # Retourne la projection linéaire.
    reduit = PCA(n_components=min(30, z.shape[1]), random_state=graine).fit_transform(z)  # Réduit d'abord la dimension.
    return TSNE(n_components=2, init="pca", perplexity=30, random_state=graine, learning_rate="auto").fit_transform(reduit)  # Retourne la projection non linéaire.


def _nuage_modalites(axe: plt.Axes, z_i: np.ndarray, z_t: np.ndarray, titre: str, methode: str) -> None:  # Trace un nuage image-texte.
    """Trace un nuage commun image-texte, en reliant quelques vraies paires ; la projection est ajustée sur les deux modalités."""
    p = _projeter(np.concatenate([z_i, z_t], axis=0), methode)  # Projette les deux modalités ensemble.
    p_i, p_t = p[: len(z_i)], p[len(z_i):]  # Sépare les deux modalités.
    for rang in range(0, min(len(z_i), 40)):  # Parcourt quelques paires.
        axe.plot([p_i[rang, 0], p_t[rang, 0]], [p_i[rang, 1], p_t[rang, 1]], color=AXE, linewidth=0.7, zorder=1)  # Relie la paire.
    axe.scatter(p_i[:, 0], p_i[:, 1], s=18, color=BLEU, marker="o", label="Image", edgecolors=SURFACE, linewidths=0.5, zorder=2)  # Trace les images.
    axe.scatter(p_t[:, 0], p_t[:, 1], s=20, color=ORANGE, marker="^", label="Texte", edgecolors=SURFACE, linewidths=0.5, zorder=2)  # Trace les textes.
    distance = float(np.mean(np.sum(z_i * z_t, axis=1)))  # Mesure la similarité moyenne des vraies paires.
    axe.set_title(f"{titre}\ncosinus moyen des vraies paires : {distance:.3f}")  # Ajoute le titre et la mesure.
    axe.set_xticks([])  # Masque les graduations horizontales, sans signification ici.
    axe.set_yticks([])  # Masque les graduations verticales, sans signification ici.
    axe.legend(loc="best")  # Ajoute la légende.


def latent_avant_apres(z_avant: Tuple[np.ndarray, np.ndarray], z_apres: Tuple[np.ndarray, np.ndarray], chemin: Path, methode: str = "tsne") -> None:  # Trace l'espace latent.
    """Compare l'espace latent partagé avant et après entraînement, sur les mêmes images de test."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))  # Crée les deux panneaux.
    _nuage_modalites(axes[0], z_avant[0], z_avant[1], "B4 avant entraînement", methode)  # Trace l'état initial.
    _nuage_modalites(axes[1], z_apres[0], z_apres[1], "B4 après entraînement", methode)  # Trace l'état final.
    figure.text(0.5, -0.02, "Illustration qualitative par t-SNE : les distances projetées ne constituent pas une preuve.", ha="center", fontsize=8.5, color=MUTE)  # Rappelle le statut de la figure.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def clip_vs_topofusion(z_clip: Tuple[np.ndarray, np.ndarray], z_topo: Tuple[np.ndarray, np.ndarray], chemin: Path) -> None:  # Compare CLIP et TopoFusion.
    """Compare les espaces de CLIP et de TopoFusion, sur les mêmes images de test."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))  # Crée les deux panneaux.
    _nuage_modalites(axes[0], z_clip[0], z_clip[1], "CLIP pré-entraîné", "tsne")  # Trace CLIP.
    _nuage_modalites(axes[1], z_topo[0], z_topo[1], "TopoFusion-TR Hyper (B4)", "tsne")  # Trace TopoFusion.
    figure.text(0.5, -0.02, "Illustration qualitative par t-SNE : les cosinus de deux modèles différents ne sont pas directement comparables.", ha="center", fontsize=8.5, color=MUTE)  # Rappelle le statut de la figure.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def _diagramme(axe: plt.Axes, h0: np.ndarray, h1: np.ndarray, titre: str) -> None:  # Trace un diagramme.
    """Trace un diagramme de persistance exact, avec les classes essentielles en haut de l'axe."""
    finis = [d[np.isfinite(d[:, 1])] for d in (h0, h1) if d.size]  # Garde les points finis.
    maximum = max([float(d.max()) for d in finis if d.size] + [1e-3]) * 1.08  # Calcule la borne de l'axe.
    axe.plot([0, maximum], [0, maximum], color=AXE, linewidth=1.0)  # Trace la diagonale.
    fini_0 = h0[np.isfinite(h0[:, 1])] if h0.size else np.empty((0, 2))  # Sélectionne H0 fini.
    infini_0 = h0[np.isinf(h0[:, 1])] if h0.size else np.empty((0, 2))  # Sélectionne H0 essentiel.
    axe.scatter(fini_0[:, 0], fini_0[:, 1], s=20, color=BLEU, marker="o", label=f"H0 ({len(fini_0)} finies)", edgecolors=SURFACE, linewidths=0.5)  # Trace H0 fini.
    if len(infini_0):  # Vérifie la présence de classes essentielles.
        axe.scatter(infini_0[:, 0], np.full(len(infini_0), maximum), s=34, color=BLEU, marker="v", label=f"H0 essentielle ({len(infini_0)}, mort infinie)", edgecolors=SURFACE, linewidths=0.5)  # Trace les classes essentielles.
    if h1.size:  # Vérifie la présence de cycles.
        axe.scatter(h1[:, 0], h1[:, 1], s=24, color=ORANGE, marker="^", label=f"H1 ({len(h1)} cycles)", edgecolors=SURFACE, linewidths=0.5)  # Trace H1.
    axe.set_xlabel("Naissance")  # Nomme l'axe horizontal.
    axe.set_ylabel("Mort")  # Nomme l'axe vertical.
    axe.set_title(titre)  # Ajoute le titre.
    axe.legend(loc="lower right")  # Ajoute la légende.


def diagrammes_persistance(diagrammes: Dict[str, Tuple[np.ndarray, np.ndarray]], chemin: Path) -> None:  # Trace plusieurs diagrammes.
    """Trace plusieurs diagrammes de persistance exacts côte à côte."""
    figure, axes = plt.subplots(1, len(diagrammes), figsize=(5.2 * len(diagrammes), 4.8))  # Crée les panneaux.
    for axe, (titre, (h0, h1)) in zip(np.ravel(axes), diagrammes.items()):  # Parcourt les diagrammes.
        _diagramme(axe, h0, h1, titre)  # Trace le diagramme.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def paysages(grille: np.ndarray, p_image: Dict[int, np.ndarray], p_texte: Dict[int, np.ndarray], chemin: Path) -> None:  # Trace les paysages.
    """Trace les paysages de persistance exacts de H0 et H1, pour l'image et pour sa légende."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))  # Crée les deux panneaux.
    for axe, dimension in zip(axes, (0, 1)):  # Parcourt H0 et H1.
        for couche in range(p_image[dimension].shape[0]):  # Parcourt les couches.
            style = ["-", "--", ":"][couche % 3]  # Distingue les couches par le style de trait.
            axe.plot(grille, p_image[dimension][couche], color=BLEU, linestyle=style, label=f"Image, λ{couche + 1}")  # Trace la couche de l'image.
            axe.plot(grille, p_texte[dimension][couche], color=ORANGE, linestyle=style, label=f"Texte, λ{couche + 1}")  # Trace la couche du texte.
        axe.set_title(f"Paysage de persistance de H{dimension}")  # Ajoute le titre.
        axe.set_xlabel("Valeur de filtration t")  # Nomme l'axe horizontal.
        axe.set_ylabel("λ(t)")  # Nomme l'axe vertical.
        axe.set_ylim(bottom=0)  # Fixe le bas de l'axe à zéro.
        axe.legend(ncol=2, loc="upper right")  # Ajoute la légende.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def graphe_latent(adjacence: np.ndarray, filtration: np.ndarray, chemin: Path, nb_aretes: int = 18) -> None:  # Trace le graphe latent.
    """Trace le graphe latent appris : la position vient de la filtration, l'épaisseur du poids de l'arête."""
    distances = filtration.copy()  # Copie la matrice de filtration.
    np.fill_diagonal(distances, 0.0)  # Annule la diagonale pour obtenir une dissimilarité.
    positions = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto", n_init=4).fit_transform(distances)  # Calcule les positions des nœuds.
    n = adjacence.shape[0]  # Lit le nombre de nœuds.
    lignes, colonnes = np.triu_indices(n, k=1)  # Énumère les arêtes.
    ordre = np.argsort(-adjacence[lignes, colonnes])[:nb_aretes]  # Garde les arêtes les plus fortes, pour la lisibilité.
    figure, axe = plt.subplots(figsize=(6.5, 5.6))  # Crée la figure.
    for rang in ordre:  # Parcourt les arêtes retenues.
        i, j = lignes[rang], colonnes[rang]  # Lit les extrémités.
        axe.plot(positions[[i, j], 0], positions[[i, j], 1], color=BLEU, alpha=0.55, linewidth=0.5 + 3.0 * float(adjacence[i, j]), zorder=1)  # Trace l'arête.
    axe.scatter(positions[:, 0], positions[:, 1], s=210, color=SURFACE, edgecolors=ENCRE_2, linewidths=1.2, zorder=2)  # Trace les nœuds.
    for rang in range(n):  # Parcourt les nœuds.
        axe.annotate(str(rang), positions[rang], ha="center", va="center", fontsize=8.5, color=ENCRE, zorder=3)  # Numérote le nœud.
    axe.set_title(f"Graphe latent sur les {n} tokens partagés d'une image de test\n({nb_aretes} arêtes les plus fortes ; position par MDS sur la filtration)")  # Ajoute le titre.
    axe.set_xticks([])  # Masque les graduations horizontales, sans signification ici.
    axe.set_yticks([])  # Masque les graduations verticales, sans signification ici.
    axe.grid(False)  # Retire la grille.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def attention_topologique(biais: np.ndarray, attention: np.ndarray, chemin: Path, nb_partages: int) -> None:  # Trace l'attention.
    """Affiche le biais T_ij et les poids d'attention, avec une échelle séquentielle à une seule teinte."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))  # Crée les deux panneaux.
    teinte = matplotlib.colors.LinearSegmentedColormap.from_list("sequentiel_bleu", ["#f3f7fd", "#9ec5f4", "#2a78d6", "#0d366b"])  # Crée une échelle séquentielle à une teinte.
    image_1 = axes[0].imshow(biais, cmap=teinte, vmin=0.0, vmax=1.0, interpolation="nearest")  # Affiche le biais.
    axes[0].set_title("Biais topologique T_ij (H0 et graphe)")  # Ajoute le titre.
    figure.colorbar(image_1, ax=axes[0], fraction=0.046, label="T_ij")  # Ajoute l'échelle.
    image_2 = axes[1].imshow(attention, cmap=teinte, interpolation="nearest")  # Affiche l'attention.
    axes[1].set_title("Poids d'attention A_ij (moyenne des têtes)")  # Ajoute le titre.
    figure.colorbar(image_2, ax=axes[1], fraction=0.046, label="A_ij")  # Ajoute l'échelle.
    axes[1].axhline(nb_partages - 0.5, color=ORANGE, linewidth=1.2)  # Sépare les tokens partagés et topologiques.
    axes[1].axvline(nb_partages - 0.5, color=ORANGE, linewidth=1.2)  # Sépare les tokens partagés et topologiques.
    for axe in axes:  # Parcourt les panneaux.
        axe.set_xlabel("Token j")  # Nomme l'axe horizontal.
        axe.set_ylabel("Token i")  # Nomme l'axe vertical.
        axe.grid(False)  # Retire la grille.
    figure.text(0.76, -0.01, f"Au-delà du trait orange : tokens H0 et H1 (indices ≥ {nb_partages})", ha="center", fontsize=8.5, color=MUTE)  # Explique le repère.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def robustesse(resultats: Dict[str, Dict[str, Dict[str, float]]], scenarios: List[str], chemin: Path) -> None:  # Trace la robustesse.
    """Compare la performance absolue et la dégradation relative sous corruption, en petits multiples."""
    modeles = list(resultats)  # Liste les modèles.
    figure, axes = plt.subplots(1, 2, figsize=(13, 0.48 * len(scenarios) + 2.2), sharey=True)  # Crée les deux panneaux.
    positions = np.arange(len(scenarios))  # Calcule les positions des scénarios.
    hauteur = 0.8 / len(modeles)  # Calcule l'épaisseur des barres.
    for rang, nom in enumerate(modeles):  # Parcourt les modèles.
        decalage = (rang - (len(modeles) - 1) / 2.0) * hauteur  # Calcule le décalage de la barre.
        couleur = COULEURS.get(nom, MUTE)  # Lit la couleur du modèle.
        axes[0].barh(positions + decalage, [resultats[nom][s]["r5"] for s in scenarios], height=hauteur * 0.85, color=couleur, label=nom, xerr=[resultats[nom][s]["r5_std"] for s in scenarios], error_kw={"elinewidth": 0.8, "ecolor": ENCRE_2})  # Trace la performance absolue.
        axes[1].barh(positions + decalage, [resultats[nom][s]["degradation"] for s in scenarios], height=hauteur * 0.85, color=couleur, xerr=[resultats[nom][s]["degradation_std"] for s in scenarios], error_kw={"elinewidth": 0.8, "ecolor": ENCRE_2})  # Trace la dégradation relative.
    axes[0].set_yticks(positions)  # Place les scénarios.
    axes[0].set_yticklabels(scenarios, color=ENCRE_2)  # Nomme les scénarios.
    axes[0].invert_yaxis()  # Place le premier scénario en haut.
    axes[0].set_xlabel("Recall@5 moyen sous corruption (%)")  # Nomme l'axe horizontal.
    axes[0].set_title("Performance absolue")  # Ajoute le titre.
    axes[1].set_xlabel("Perte relative du Recall@5 (%) : plus court est mieux")  # Nomme l'axe horizontal.
    axes[1].set_title("Dégradation relative")  # Ajoute le titre.
    axes[0].legend(loc="lower right")  # Ajoute la légende.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def ablation(valeurs: Dict[str, Dict[str, Tuple[float, float]]], chemin: Path, reference: Optional[str] = None) -> None:  # Trace l'étude d'ablation.
    """Trace l'étude d'ablation : une seule teinte par mesure, et une barre d'erreur sur les graines."""
    noms = list(valeurs)  # Liste les modèles.
    mesures = [("mean_recall_at_1", "Recall@1 moyen (%)"), ("mean_recall_at_5", "Recall@5 moyen (%)"), ("mean_map", "mAP moyen (%)")]  # Liste les mesures.
    figure, axes = plt.subplots(1, 3, figsize=(14, 0.5 * len(noms) + 1.9), sharey=True)  # Crée les petits multiples.
    positions = np.arange(len(noms))  # Calcule les positions des modèles.
    for axe, (cle, titre) in zip(axes, mesures):  # Parcourt les mesures.
        moyennes = [valeurs[n][cle][0] for n in noms]  # Lit les moyennes.
        ecarts = [valeurs[n][cle][1] for n in noms]  # Lit les écarts-types.
        axe.barh(positions, moyennes, xerr=ecarts, height=0.62, color=BLEU, error_kw={"elinewidth": 1.0, "ecolor": ENCRE_2})  # Trace les barres.
        for rang, (moyenne, ecart) in enumerate(zip(moyennes, ecarts)):  # Parcourt les valeurs.
            axe.annotate(f"{moyenne:.1f}", (moyenne + ecart, rang), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=ENCRE_2)  # Ajoute l'étiquette directe.
        if reference and reference in valeurs:  # Vérifie la présence de la référence.
            axe.axvline(valeurs[reference][cle][0], color=ORANGE, linewidth=1.2, linestyle="--")  # Trace le niveau de référence.
        axe.set_title(titre)  # Ajoute le titre.
        axe.set_xlim(left=0)  # Fixe le début de l'axe à zéro.
        axe.margins(x=0.15)  # Laisse la place aux étiquettes.
    axes[0].set_yticks(positions)  # Place les modèles.
    axes[0].set_yticklabels(noms, color=ENCRE_2)  # Nomme les modèles.
    axes[0].invert_yaxis()  # Place le premier modèle en haut.
    if reference:  # Vérifie la présence de la référence.
        figure.text(0.5, -0.02, f"Trait orange : niveau de {reference}. Barres d'erreur : écart-type sur les graines.", ha="center", fontsize=8.5, color=MUTE)  # Explique le repère.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def stabilite(deplacements: np.ndarray, distances: Dict[str, np.ndarray], borne: float, chemin: Path) -> None:  # Trace la vérification de la borne.
    """Trace la distance de bottleneck en fonction du déplacement, avec la borne théorique démontrée."""
    figure, axe = plt.subplots(figsize=(6.8, 4.8))  # Crée la figure.
    axe.scatter(deplacements, distances["H0"], s=14, color=BLEU, marker="o", alpha=0.65, label="H0", edgecolors="none")  # Trace H0.
    axe.scatter(deplacements, distances["H1"], s=16, color=ORANGE, marker="^", alpha=0.65, label="H1", edgecolors="none")  # Trace H1.
    x = np.linspace(0.0, float(deplacements.max()) * 1.05, 10)  # Calcule les abscisses de la borne.
    axe.plot(x, borne * x, color=ENCRE_2, linestyle="--", label=f"Borne démontrée : d_B = {borne:.2f} · δ")  # Trace la borne.
    axe.set_xlabel("Déplacement maximal des points δ = max ||u_i − u'_i||")  # Nomme l'axe horizontal.
    axe.set_ylabel("Distance de bottleneck")  # Nomme l'axe vertical.
    axe.set_title("Vérification numérique de la borne de stabilité\n(tokens de test, corruptions réelles)")  # Ajoute le titre.
    axe.set_xlim(left=0)  # Fixe le début de l'axe horizontal.
    axe.set_ylim(bottom=0)  # Fixe le début de l'axe vertical.
    axe.legend(loc="upper left")  # Ajoute la légende.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.


def betti_douce_vs_exacte(echelles: np.ndarray, exacte: Dict[int, np.ndarray], douce: Dict[int, np.ndarray], temperature: float, chemin: Path) -> None:  # Compare les courbes de Betti.
    """Compare la courbe de Betti douce, optimisée, à la courbe exacte, utilisée pour le diagnostic."""
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4))  # Crée les deux panneaux.
    for axe, dimension in zip(axes, (0, 1)):  # Parcourt H0 et H1.
        axe.step(echelles, exacte[dimension], where="post", color=BLEU, label="Exacte (GUDHI)")  # Trace la courbe exacte.
        axe.plot(echelles, douce[dimension], color=ORANGE, linestyle="--", label=f"Douce (τ = {temperature})")  # Trace la courbe douce.
        erreur = float(np.mean(np.abs(exacte[dimension] - douce[dimension])))  # Calcule l'erreur moyenne.
        axe.set_title(f"β{dimension}(ε), erreur absolue moyenne : {erreur:.3f}")  # Ajoute le titre et l'erreur.
        axe.set_xlabel("Échelle ε")  # Nomme l'axe horizontal.
        axe.set_ylabel("Nombre de classes vivantes")  # Nomme l'axe vertical.
        axe.set_ylim(bottom=0)  # Fixe le début de l'axe vertical.
        axe.legend(loc="upper right")  # Ajoute la légende.
    figure.tight_layout()  # Ajuste la mise en page.
    enregistrer(figure, chemin)  # Enregistre la figure.
