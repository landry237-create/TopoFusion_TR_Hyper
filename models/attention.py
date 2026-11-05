# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Attention biaisée par la topologie : A_ij = softmax(q_i k_j^T / sqrt(d_h) + lambda T_ij)."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from typing import Optional, Tuple  # Importe les types des signatures.

import torch  # Fournit les tenseurs.
import torch.nn as nn  # Fournit les modules de réseaux de neurones.


class TopologicalAttention(nn.Module):  # Calcule l'attention multi-tête biaisée.
    """Attention multi-tête avec un biais additif et un coefficient appris par tête."""

    def __init__(self, dim_latente: int, nb_tetes: int, dropout: float) -> None:  # Construit l'attention.
        """Crée les projections et le coefficient lambda de chaque tête."""
        super().__init__()  # Initialise le module parent.
        if dim_latente % nb_tetes != 0:  # Vérifie la compatibilité des dimensions.
            raise ValueError(f"dim_latente ({dim_latente}) doit être divisible par nb_tetes ({nb_tetes}).")  # Explique l'erreur.
        self.nb_tetes = nb_tetes  # Conserve le nombre de têtes.
        self.dim_tete = dim_latente // nb_tetes  # Calcule la dimension de chaque tête.
        self.qkv = nn.Linear(dim_latente, dim_latente * 3)  # Crée les projections Q, K et V.
        self.sortie = nn.Linear(dim_latente, dim_latente)  # Crée la projection de sortie.
        self.lambda_biais = nn.Parameter(torch.full((nb_tetes, 1, 1), 1.0))  # Crée le coefficient du biais pour chaque tête.
        self.dropout = nn.Dropout(dropout)  # Crée le dropout des poids d'attention.

    def forward(self, x: torch.Tensor, biais: Optional[torch.Tensor] = None, masque: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:  # Applique l'attention.
        """Retourne les tokens transformés et les poids d'attention (B, H, N, N), avant dropout."""
        b, n, d = x.shape  # Lit les dimensions de l'entrée.
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # Calcule Q, K et V.
        q = q.view(b, n, self.nb_tetes, self.dim_tete).transpose(1, 2)  # Sépare Q par tête.
        k = k.view(b, n, self.nb_tetes, self.dim_tete).transpose(1, 2)  # Sépare K par tête.
        v = v.view(b, n, self.nb_tetes, self.dim_tete).transpose(1, 2)  # Sépare V par tête.
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_tete ** 0.5)  # Calcule le produit scalaire normalisé.
        if biais is not None:  # Vérifie la présence d'un biais topologique.
            scores = scores + self.lambda_biais * biais.unsqueeze(1)  # Ajoute lambda * T_ij.
        if masque is not None:  # Vérifie la présence d'un masque.
            scores = scores.masked_fill(~masque.view(b, 1, 1, n), float("-inf"))  # Exclut les positions de padding.
        poids = torch.softmax(scores, dim=-1)  # Calcule les poids d'attention.
        resultat = torch.matmul(self.dropout(poids), v).transpose(1, 2).reshape(b, n, d)  # Applique les poids aux valeurs.
        return self.sortie(resultat), poids  # Retourne les tokens et les poids.


class TopologicalTransformerBlock(nn.Module):  # Assemble un bloc Transformer.
    """Bloc Transformer pré-normalisé : attention topologique puis MLP, avec connexions résiduelles."""

    def __init__(self, dim_latente: int, nb_tetes: int, dropout: float) -> None:  # Construit le bloc.
        """Crée les normalisations, l'attention et le MLP."""
        super().__init__()  # Initialise le module parent.
        self.norme_attention = nn.LayerNorm(dim_latente)  # Normalise avant l'attention.
        self.attention = TopologicalAttention(dim_latente, nb_tetes, dropout)  # Crée l'attention biaisée.
        self.norme_mlp = nn.LayerNorm(dim_latente)  # Normalise avant le MLP.
        self.mlp = nn.Sequential(nn.Linear(dim_latente, dim_latente * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_latente * 2, dim_latente), nn.Dropout(dropout))  # Crée le MLP.

    def forward(self, x: torch.Tensor, biais: Optional[torch.Tensor] = None, masque: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:  # Applique le bloc.
        """Retourne les tokens transformés et les poids d'attention."""
        delta, poids = self.attention(self.norme_attention(x), biais, masque)  # Applique l'attention.
        x = x + delta  # Ajoute la connexion résiduelle.
        return x + self.mlp(self.norme_mlp(x)), poids  # Applique le MLP résiduel.
