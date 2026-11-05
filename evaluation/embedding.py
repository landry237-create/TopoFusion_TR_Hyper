# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Calcul des embeddings d'un sous-ensemble et évaluation de la recherche inter-modale."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from typing import Dict, Sequence, Tuple  # Importe les types des signatures.

import numpy as np  # Fournit les tableaux de résultats.
import torch  # Fournit les tenseurs.

from data.feature_cache import FeatureSplit  # Importe l'accès aux caractéristiques.
from utils.metrics import retrieval_metrics  # Importe les métriques de recherche.


@torch.no_grad()  # Désactive le calcul du gradient.
def encode_split(modele: torch.nn.Module, split: FeatureSplit, taille_lot: int, appareil: torch.device, desactiver_topo: bool = False) -> Tuple[np.ndarray, np.ndarray]:  # Encode un sous-ensemble.
    """Encode séparément les images et les légendes : aucune modalité ne voit l'autre."""
    modele.eval()  # Désactive le dropout.
    images = []  # Prépare les embeddings d'images.
    for debut in range(0, split.nb_images, taille_lot):  # Parcourt les images par lots.
        lot = torch.from_numpy(np.asarray(split.images[debut: debut + taille_lot], dtype=np.float32)).to(appareil)  # Charge le lot d'images.
        images.append(modele({"image": (lot, None)}, desactiver_topo=desactiver_topo, fusionner=False)["embeddings"]["image"].cpu().numpy())  # Encode le lot d'images.
    textes = []  # Prépare les embeddings de textes.
    for debut in range(0, split.nb_legendes, taille_lot):  # Parcourt les légendes par lots.
        lot = torch.from_numpy(np.asarray(split.textes[debut: debut + taille_lot], dtype=np.float32)).to(appareil)  # Charge le lot de textes.
        masque = torch.from_numpy(np.array(split.masques_texte[debut: debut + taille_lot], dtype=bool)).to(appareil)  # Copie le masque, car un tableau projeté en mémoire n'est pas modifiable.
        textes.append(modele({"text": (lot, masque)}, desactiver_topo=desactiver_topo, fusionner=False)["embeddings"]["text"].cpu().numpy())  # Encode le lot de textes.
    return np.concatenate(images, axis=0), np.concatenate(textes, axis=0)  # Retourne les embeddings.


def evaluate_retrieval(modele: torch.nn.Module, split: FeatureSplit, taille_lot: int, appareil: torch.device, ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:  # Évalue la recherche.
    """Calcule les métriques de recherche d'un modèle sur un sous-ensemble."""
    images, textes = encode_split(modele, split, taille_lot, appareil)  # Encode le sous-ensemble.
    return retrieval_metrics(images, textes, split.relevance_matrix(), ks)  # Retourne les métriques.
