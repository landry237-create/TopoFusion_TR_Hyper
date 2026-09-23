# -*- coding: utf-8 -*-
"""
Semantic Latent Bank, Factorisation Shared / Private et Fusion Adaptative (Bloc 2 du schéma architectural).
- Semantic Latent Bank: ensemble commun de K tokens latents apprenables S in R^{K x d}.
- Split Shared / Private:
    * Tokens partagés: S_c in R^{K_c x d}
    * Tokens privés: S_p in R^{K_p x d}
    avec K = K_c + K_p
- Fusion Adaptative: combinaison convexe guidée par un réseau de confiance alpha_m.
"""

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLatentBank(nn.Module):
    """
    Banque de tokens sémantiques latents (Bloc 2).
    Interagit avec chaque modalité via Cross-Attention pour extraire une représentation
    compacte et normalisée de K tokens latents.
    """
    def __init__(self, num_tokens: int = 32, latent_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        # Initialisation orthogonale des K tokens latents
        self.latent_tokens = nn.Parameter(torch.empty(1, num_tokens, latent_dim))
        nn.init.orthogonal_(self.latent_tokens)

        # Cross-Attention: Q = latents, K, V = modality tokens
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.mha = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)

        # MLP avec connexion résiduelle
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, modality_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            modality_tokens: Tokens locaux d'une modalité (batch_size, seq_len, latent_dim)
            mask: Masque d'attention optionnel (batch_size, seq_len)
        Returns:
            Latents mis à jour pour la modalité: (batch_size, K, latent_dim)
        """
        batch_size = modality_tokens.size(0)
        latents = self.latent_tokens.expand(batch_size, -1, -1)

        q = self.q_proj(latents)
        k = self.k_proj(modality_tokens)
        v = self.v_proj(modality_tokens)

        # key_padding_mask attend un masque booléen True pour les tokens à ignorer
        key_padding_mask = (mask == 0) if mask is not None else None

        attn_out, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask)
        latents = self.norm1(latents + attn_out)
        latents = self.norm2(latents + self.mlp(latents))

        return latents


class SharedPrivateFactorization(nn.Module):
    """
    Factorisation en sous-espaces Partagé (Shared S_c) et Privé (Private S_p).
    Permet de découpler l'information cross-modale commune de l'information spécifique à chaque capteur.
    """
    def __init__(self, num_tokens: int = 32, num_shared: int = 16, latent_dim: int = 256):
        super().__init__()
        assert num_shared <= num_tokens, "num_shared doit être <= num_tokens"
        self.num_tokens = num_tokens
        self.num_shared = num_shared
        self.num_private = num_tokens - num_shared
        self.latent_dim = latent_dim

        # Projecteurs dédiés pour spécialiser chaque sous-espace
        self.shared_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        self.private_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latents: Tenseur (batch_size, num_tokens, latent_dim)
        Returns:
            shared: Tenseur S_c (batch_size, num_shared, latent_dim)
            private: Tenseur S_p (batch_size, num_private, latent_dim)
        """
        shared_raw = latents[:, :self.num_shared, :]
        private_raw = latents[:, self.num_shared:, :]

        shared = self.shared_proj(shared_raw)
        private = self.private_proj(private_raw)
        return shared, private

    def orthogonality_loss(self, shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
        """
        Pénalité d'orthogonalité / décorrélation entre les représentations partagées et privées.
        Minimise la corrélation croisée pour éviter la fuite d'information.
        """
        # Normalisation le long de la dimension latente
        s_norm = F.normalize(shared.mean(dim=1), p=2, dim=-1) # (B, D)
        p_norm = F.normalize(private.mean(dim=1), p=2, dim=-1) # (B, D)
        cosine_sim = torch.sum(s_norm * p_norm, dim=-1) # (B,)
        return torch.mean(cosine_sim ** 2)


class AdaptiveFusion(nn.Module):
    """
    Fusion Adaptative des tokens partagés S_c de toutes les modalités (Bloc 2).
    Calcule des coefficients de confiance / pertinence alpha_m pour chaque slot et chaque modalité.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.confidence_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1)
        )

    def forward(self, shared_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            shared_list: Liste des tenseurs S_c de chaque modalité, chacun de shape (batch_size, num_shared, latent_dim)
        Returns:
            fused_shared: Tenseur fusionné S_c^fused (batch_size, num_shared, latent_dim)
            alphas: Poids de confiance (batch_size, num_modalities, num_shared, 1)
        """
        # Empilement: (B, M, K_c, D)
        stacked = torch.stack(shared_list, dim=1)

        # Scores de confiance non-normalisés: (B, M, K_c, 1)
        scores = self.confidence_net(stacked)

        # Softmax sur la dimension des modalités M (dim=1)
        alphas = F.softmax(scores, dim=1)

        # Combinaison convexe pondérée
        fused_shared = (stacked * alphas).sum(dim=1) # (B, K_c, D)

        return fused_shared, alphas