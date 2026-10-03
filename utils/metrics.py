# -*- coding: utf-8 -*-
"""
Métriques d'évaluation pour l'architecture Topo-Multimodal.
Comprend:
- Recall@K (R@1, R@5, R@10) pour l'alignement cross-modal (Image-Texte)
- Mean Average Precision (mAP)
- Distance géodésique / Isométrie relative
"""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np


def recall_at_k(image_embeds: torch.Tensor, text_embeds: torch.Tensor, k: int = 1) -> float:
    """
    Calcule le Recall@K pour l'alignement et la recherche cross-modale image-texte.
    Args:
        image_embeds: (N, D)
        text_embeds: (N, D)
        k: Rang K pour le rappel
    Returns:
        Valeur du Recall@K dans [0, 1]
    """
    if image_embeds.size(0) == 0:
        return 0.0

    # Normalisation L2
    i_norm = F.normalize(image_embeds, p=2, dim=-1)
    t_norm = F.normalize(text_embeds, p=2, dim=-1)

    # Matrice de similarité cosinus (N, N)
    sims = torch.matmul(i_norm, t_norm.T)

    n_samples = sims.size(0)
    effective_k = min(k, n_samples)

    # Top-K indices
    _, topk_indices = sims.topk(effective_k, dim=-1)

    # Cibles attendues (appariement 1-à-1 sur la diagonale)
    targets = torch.arange(n_samples, device=sims.device).unsqueeze(1)

    # Vrai si la cible se trouve dans les top-K prédictions
    correct = (topk_indices == targets).any(dim=-1).float()
    return float(correct.mean().item())


def mean_average_precision(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> float:
    """
    Calcule le Mean Average Precision (mAP) pour l'alignement cross-modal.
    """
    if image_embeds.size(0) == 0:
        return 0.0

    i_norm = F.normalize(image_embeds, p=2, dim=-1)
    t_norm = F.normalize(text_embeds, p=2, dim=-1)
    sims = torch.matmul(i_norm, t_norm.T)

    n_samples = sims.size(0)
    # Tri décroissant des scores
    _, sorted_indices = sims.sort(dim=-1, descending=True)

    targets = torch.arange(n_samples, device=sims.device).unsqueeze(1)
    # Positions (1-indexed) de la vraie correspondance
    matches = (sorted_indices == targets).nonzero(as_tuple=False)
    if matches.size(0) == 0:
        return 0.0

    ranks = matches[:, 1].float() + 1.0
    reciprocal_ranks = 1.0 / ranks
    return float(reciprocal_ranks.mean().item())


def topological_isometry_score(shared_img: torch.Tensor, shared_txt: torch.Tensor) -> float:
    """
    Mesure la corrélation de Spearman / fidélité métrique entre les distances
    du graphe latent image et texte (préservation du manifold).
    """
    d_img = torch.cdist(shared_img, shared_img).view(-1)
    d_txt = torch.cdist(shared_txt, shared_txt).view(-1)

    # Corrélation de Pearson sur les distances
    vx = d_img - d_img.mean()
    vy = d_txt - d_txt.mean()
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return float(corr.item())