# -*- coding: utf-8 -*-
"""
Topological Attention (Schéma architectural - Bloc Attention Topologique).
Implémente le mécanisme d'attention guidé par la structure topologique:
    A_{ij} = softmax( (q_i k_j^T) / sqrt(d) + lambda * T_{ij} )
Fournit un bloc Transformer complet avec connexions résiduelles, LayerNorm et MLP.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalAttention(nn.Module):
    """
    Mécanisme d'attention Multi-Tête avec biais topologique additif (T_{ij}).
    Le biais topologique peut représenter l'adjacence du graphe latent, la distance géodésique
    ou la matrice de persistance relative.
    """
    def __init__(self, latent_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert latent_dim % num_heads == 0, f"latent_dim ({latent_dim}) doit être divisible par num_heads ({num_heads})"
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        # Projections linéaires Q, K, V
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        # Paramètre lambda apprenable par tête pour équilibrer l'attention sémantique et topologique
        self.lambda_param = nn.Parameter(torch.full((num_heads, 1, 1), 0.2))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, topological_bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tenseur d'entrée (batch_size, seq_len, latent_dim)
            topological_bias: Matrice de biais topologique T_{ij} (batch_size, seq_len, seq_len)
            mask: Masque optionnel (batch_size, seq_len)
        Returns:
            out: Tenseur projeté après attention (batch_size, seq_len, latent_dim)
            attn_weights: Poids d'attention (batch_size, num_heads, seq_len, seq_len)
        """
        b, n, d = x.shape

        # Projections Q, K, V
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, D_h)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, D_h)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, D_h)

        # Produit scalaire Q * K^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (B, H, N, N)

        # Ajout du biais topologique lambda * T_{ij}
        if topological_bias is not None:
            if topological_bias.dim() == 3:
                bias = topological_bias.unsqueeze(1) # (B, 1, N, N)
            else:
                bias = topological_bias # (B, H, N, N)
            scores = scores + self.lambda_param * bias

        # Masquage d'attention si spécifié
        if mask is not None:
            mask_bias = (1.0 - mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            scores = scores + mask_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Application aux valeurs V
        out = torch.matmul(attn_weights, v) # (B, H, N, D_h)
        out = out.transpose(1, 2).contiguous().view(b, n, d) # (B, N, D)
        out = self.out_proj(out)

        return out, attn_weights


class TopologicalTransformerBlock(nn.Module):
    """
    Bloc Transformer complet intégrant l'attention topologique (Pre-LayerNorm).
    Structure:
        x -> LayerNorm -> TopologicalAttention -> Residual -> LayerNorm -> MLP -> Residual
    """
    def __init__(self, latent_dim: int = 256, num_heads: int = 4, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = TopologicalAttention(latent_dim=latent_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(latent_dim)

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * mlp_ratio, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, topological_bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN + Self-Attention topologique + résidu
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x, topological_bias=topological_bias, mask=mask)
        x = x + attn_out

        # Pre-LN + FFN + résidu
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights