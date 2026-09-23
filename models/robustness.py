# -*- coding: utf-8 -*-
"""
Module de Robustesse (Bloc 9 du schéma architectural).
Applique des perturbations réalistes pour entraîner le modèle à l'invariance:
- Bruit gaussien additif sur les caractéristiques
- Masquage stochastique de tokens (Token Cropping / Dropout)
- Décrochage multimodal (Modality Dropout - simulation de capteur défaillant)
- Compression / perturbation d'amplitude
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustnessPerturbationModule(nn.Module):
    """
    Générateur de perturbations multimodales (Bloc 9).
    Injecte des dégradations contrôlées pendant l'entraînement pour maximiser la robustesse
    et permettre le calcul de L_robust.
    """
    def __init__(self, noise_std: float = 0.05, mask_prob: float = 0.15, modality_drop_prob: float = 0.1):
        super().__init__()
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.modality_drop_prob = modality_drop_prob

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Perturbe un tenseur de caractéristiques / tokens.
        Args:
            x: (batch_size, seq_len, dim)
            training: Si False, retourne x sans altération.
        Returns:
            x_perturbed: (batch_size, seq_len, dim)
        """
        if not training:
            return x

        # 1. Bruit gaussien
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # 2. Token Masking / Cropping
        if self.mask_prob > 0:
            b, n, _ = x.shape
            mask = (torch.rand(b, n, 1, device=x.device) > self.mask_prob).float()
            x = x * mask

        return x

    def apply_modality_dropout(self, shared_modalities: List[torch.Tensor], training: bool = True) -> List[torch.Tensor]:
        """
        Applique un dropout de modalité pour simuler la perte d'un flux de données (ex: audio muet, vidéo corrompue).
        Garantit qu'au moins une modalité reste active pour chaque échantillon.
        """
        if not training or self.modality_drop_prob <= 0 or len(shared_modalities) <= 1:
            return shared_modalities

        num_modalities = len(shared_modalities)
        batch_size = shared_modalities[0].size(0)
        device = shared_modalities[0].device

        # Masque aléatoire par modalité
        drop_mask = torch.rand(batch_size, num_modalities, device=device) >= self.modality_drop_prob

        # Forcer au moins une modalité active par batch item
        all_dropped = (drop_mask.sum(dim=1) == 0)
        if all_dropped.any():
            random_keep = torch.randint(0, num_modalities, (batch_size,), device=device)
            drop_mask[torch.arange(batch_size, device=device), random_keep] = True

        perturbed_list = []
        for m_idx, m_tensor in enumerate(shared_modalities):
            m_mask = drop_mask[:, m_idx].view(batch_size, 1, 1).float()
            perturbed_list.append(m_tensor * m_mask)

        return perturbed_list
