# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Banque latente sémantique, factorisation partagé/privé et fusion adaptative avec masques."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from typing import Dict, List, Optional, Sequence, Tuple  # Importe les types des signatures.

import torch  # Fournit les tenseurs.
import torch.nn as nn  # Fournit les modules de réseaux de neurones.
import torch.nn.functional as F  # Fournit la normalisation et le softmax.


class SemanticLatentBank(nn.Module):  # Interroge une modalité avec des requêtes apprises.
    """Banque de K tokens appris, servant de requêtes universelles par attention croisée."""

    def __init__(self, nb_tokens: int, dim_latente: int, nb_tetes: int, dropout: float) -> None:  # Construit la banque.
        """Crée des tokens réellement orthonormés et le bloc d'attention croisée."""
        super().__init__()  # Initialise le module parent.
        self.tokens = nn.Parameter(torch.empty(nb_tokens, dim_latente))  # Déclare la matrice K x d.
        nn.init.orthogonal_(self.tokens)  # Rend les K tokens mutuellement orthogonaux.
        self.attention = nn.MultiheadAttention(dim_latente, nb_tetes, dropout=dropout, batch_first=True)  # Crée l'attention croisée.
        self.norme_requete = nn.LayerNorm(dim_latente)  # Normalise les requêtes.
        self.norme_sortie = nn.LayerNorm(dim_latente)  # Normalise le résultat de l'attention.
        self.norme_mlp = nn.LayerNorm(dim_latente)  # Normalise le résultat du MLP.
        self.mlp = nn.Sequential(nn.Linear(dim_latente, dim_latente * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_latente * 2, dim_latente), nn.Dropout(dropout))  # Crée le MLP résiduel.

    def forward(self, tokens_modalite: torch.Tensor, masque: Optional[torch.Tensor] = None) -> torch.Tensor:  # Applique la banque.
        """Retourne (B, K, d) ; les positions de padding sont exclues de l'attention."""
        requetes = self.norme_requete(self.tokens).unsqueeze(0).expand(tokens_modalite.size(0), -1, -1)  # Répète les requêtes pour le lot.
        padding = ~masque if masque is not None else None  # Convertit le masque au format attendu par PyTorch.
        reponse, _ = self.attention(requetes, tokens_modalite, tokens_modalite, key_padding_mask=padding, need_weights=False)  # Interroge la modalité.
        latents = self.norme_sortie(requetes + reponse)  # Ajoute la connexion résiduelle.
        return self.norme_mlp(latents + self.mlp(latents))  # Applique le MLP résiduel.


class SharedPrivateFactorization(nn.Module):  # Sépare les espaces partagé et privé.
    """Factorisation : le projecteur partagé est commun, le projecteur privé est propre à chaque modalité."""

    def __init__(self, nb_partages: int, dim_latente: int, modalites: Sequence[str]) -> None:  # Construit les projecteurs.
        """Crée un projecteur partagé et un projecteur privé par modalité."""
        super().__init__()  # Initialise le module parent.
        self.nb_partages = nb_partages  # Conserve le nombre de tokens partagés.
        self.partage = nn.Sequential(nn.Linear(dim_latente, dim_latente), nn.GELU(), nn.LayerNorm(dim_latente))  # Crée le projecteur partagé.
        self.prive = nn.ModuleDict({nom: nn.Sequential(nn.Linear(dim_latente, dim_latente), nn.GELU(), nn.LayerNorm(dim_latente)) for nom in modalites})  # Crée les projecteurs privés.

    def forward(self, latents: torch.Tensor, modalite: str) -> Tuple[torch.Tensor, torch.Tensor]:  # Sépare les tokens.
        """Retourne les tokens partagés (B, K_c, d) et privés (B, K_p, d)."""
        return self.partage(latents[:, : self.nb_partages]), self.prive[modalite](latents[:, self.nb_partages:])  # Applique chaque projecteur à sa partie.


def orthogonality_penalty(partage: torch.Tensor, prive: torch.Tensor) -> torch.Tensor:  # Calcule la pénalité d'orthogonalité.
    """Retourne la moyenne des carrés de S_c S_p^T sur des tokens normalisés (Bousmalis et al., 2016)."""
    a = F.normalize(partage, dim=-1)  # Normalise les tokens partagés.
    b = F.normalize(prive, dim=-1)  # Normalise les tokens privés.
    return torch.matmul(a, b.transpose(1, 2)).pow(2).mean()  # Retourne la moyenne des similarités au carré.


class AdaptiveFusion(nn.Module):  # Fusionne les tokens partagés disponibles.
    """Combinaison convexe apprise ; une modalité absente est exclue du softmax, jamais remplacée par des zéros."""

    def __init__(self, dim_latente: int) -> None:  # Construit le réseau de confiance.
        """Crée le réseau qui estime la confiance de chaque token."""
        super().__init__()  # Initialise le module parent.
        self.confiance = nn.Sequential(nn.Linear(dim_latente, dim_latente // 2), nn.LayerNorm(dim_latente // 2), nn.GELU(), nn.Linear(dim_latente // 2, 1))  # Crée le réseau de confiance.

    def forward(self, partages: Dict[str, torch.Tensor], disponibilite: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:  # Fusionne les modalités.
        """Retourne les tokens fusionnés, les poids alpha (B, M, K_c, 1) et l'ordre des modalités."""
        noms = sorted(partages)  # Fixe l'ordre des modalités.
        empiles = torch.stack([partages[nom] for nom in noms], dim=1)  # Empile en (B, M, K_c, d).
        presence = torch.stack([disponibilite[nom] for nom in noms], dim=1)  # Empile la présence en (B, M).
        if not presence.any(dim=1).all():  # Vérifie qu'aucun échantillon n'est entièrement vide.
            raise ValueError("Chaque échantillon doit disposer d'au moins une modalité réelle.")  # Explique l'erreur.
        scores = self.confiance(empiles)  # Calcule les scores de confiance.
        scores = scores.masked_fill(~presence.view(presence.size(0), presence.size(1), 1, 1), float("-inf"))  # Exclut les modalités absentes.
        alphas = torch.softmax(scores, dim=1)  # Normalise uniquement sur les modalités présentes.
        return (empiles * alphas).sum(dim=1), alphas, noms  # Retourne la combinaison convexe.
