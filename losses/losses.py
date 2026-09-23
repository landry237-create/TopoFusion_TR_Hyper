# -*- coding: utf-8 -*-
"""
Fonctions de perte pour l'architecture multimodale topologique (Bloc 7 du schéma architectural).
Comprend les 5 branches de supervision définies dans 'archi.png':
1. L_sem: Perte sémantique contrastive (InfoNCE symétrique)
2. L_geom: Perte géométrique (préservation des voisinages et distances métriques)
3. L_top: Perte topologique (Wasserstein / distance sur paysages de persistance)
4. L_temp: Perte temporelle (continuité, ordre et transitions temporelles)
5. L_robust: Perte de robustesse (invariance sous perturbations sémantiques et topologiques)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLoss(nn.Module):
    """
    Perte sémantique contrastive (InfoNCE symétrique multi-directionnelle).
    Aligne les embeddings globaux des modalités appariées (ex: Image <-> Texte).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: (batch_size, latent_dim)
            text_embeds: (batch_size, latent_dim)
        Returns:
            loss: Scalaire de perte contrastive
        """
        b = image_embeds.size(0)
        if b <= 1:
            return torch.tensor(0.0, device=image_embeds.device, requires_grad=True)

        # Normalisation L2 unitaire
        i_norm = F.normalize(image_embeds, p=2, dim=-1)
        t_norm = F.normalize(text_embeds, p=2, dim=-1)

        # Similarité cosinus mise à l'échelle
        temp = torch.clamp(self.temperature, min=0.01, max=1.0)
        logits = torch.matmul(i_norm, t_norm.T) / temp

        labels = torch.arange(b, device=image_embeds.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_i2t + loss_t2i)


class GeometricLoss(nn.Module):
    """
    Perte géométrique (Bloc 7 - Voisinages et distances).
    Préserve la métrique de l'espace à deux niveaux:
    1. Niveau local (tokens): aligne les matrices de distance inter-tokens partagés S_c
    2. Niveau global (échantillons): préserve les distances inter-échantillons du batch
       pour garantir un plongement isométrique entre les modalités.
    """
    def __init__(self, token_weight: float = 0.5, batch_weight: float = 0.5):
        super().__init__()
        self.token_weight = token_weight
        self.batch_weight = batch_weight

    def forward(self, shared_image: torch.Tensor, shared_text: torch.Tensor,
                embed_image: Optional[torch.Tensor] = None,
                embed_text: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            shared_image: (batch_size, num_shared, latent_dim)
            shared_text: (batch_size, num_shared, latent_dim)
            embed_image: Optionnel (batch_size, latent_dim)
            embed_text: Optionnel (batch_size, latent_dim)
        Returns:
            loss: Scalaire de perte géométrique
        """
        b, k, _ = shared_image.shape

        # 1. Préservation géométrique locale (inter-tokens)
        dist_tokens_img = torch.cdist(shared_image, shared_image, p=2) # (B, K, K)
        dist_tokens_txt = torch.cdist(shared_text, shared_text, p=2) # (B, K, K)

        # Normalisation par la distance moyenne pour robustesse aux échelles
        norm_img = dist_tokens_img / (dist_tokens_img.mean(dim=(-2, -1), keepdim=True) + 1e-6)
        norm_txt = dist_tokens_txt / (dist_tokens_txt.mean(dim=(-2, -1), keepdim=True) + 1e-6)
        token_loss = F.mse_loss(norm_img, norm_txt)

        # 2. Préservation géométrique globale du manifold (inter-échantillons dans le batch)
        if embed_image is not None and embed_text is not None and b > 1:
            batch_dist_img = torch.cdist(embed_image, embed_image, p=2) # (B, B)
            batch_dist_txt = torch.cdist(embed_text, embed_text, p=2) # (B, B)
            b_norm_img = batch_dist_img / (batch_dist_img.mean() + 1e-6)
            b_norm_txt = batch_dist_txt / (batch_dist_txt.mean() + 1e-6)
            batch_loss = F.mse_loss(b_norm_img, b_norm_txt)
        else:
            batch_loss = torch.tensor(0.0, device=shared_image.device)

        return self.token_weight * token_loss + self.batch_weight * batch_loss


class TopologicalLoss(nn.Module):
    """
    Perte topologique (Bloc 7 - Wasserstein / Diagrammes / Paysages de persistance).
    Mesure la distance topologique entre:
    1. Les signatures topologiques de l'image et du texte (homologie persistante H0/H1)
    2. La régularité de la filtration adaptative et des matrices d'adjacence
    Utilise la distance L2 entre Persistence Landscapes (qui est une borne supérieure
    de la distance de Wasserstein W_p selon la théorie TDA différentiable).
    """
    def __init__(self, landscape_weight: float = 1.0, filtration_weight: float = 0.5):
        super().__init__()
        self.landscape_weight = landscape_weight
        self.filtration_weight = filtration_weight

    def forward(self, topo_tokens_img: torch.Tensor, topo_tokens_txt: torch.Tensor,
                filt_img: Optional[torch.Tensor] = None,
                filt_txt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            topo_tokens_img: Tokens topologiques (H0 + H1) de la modalité image (B, K_topo, D)
            topo_tokens_txt: Tokens topologiques (H0 + H1) de la modalité texte (B, K_topo, D)
            filt_img: Valeurs de filtration f_theta image (B, K_c)
            filt_txt: Valeurs de filtration f_theta texte (B, K_c)
        Returns:
            loss: Scalaire de perte topologique
        """
        # Distance entre les descripteurs topologiques structurés
        topo_loss = F.mse_loss(topo_tokens_img, topo_tokens_txt)

        # Distance d'alignement des filtrations
        if filt_img is not None and filt_txt is not None:
            filt_loss = F.mse_loss(filt_img, filt_txt)
        else:
            filt_loss = torch.tensor(0.0, device=topo_tokens_img.device)

        return self.landscape_weight * topo_loss + self.filtration_weight * filt_loss


class TemporalLoss(nn.Module):
    """
    Perte temporelle pour l'audio et la vidéo (Bloc 8).
    Formulation mathématique selon 'archi.png':
    'Préserve la continuité, l'ordre et les transitions temporelles'
    Composants:
    1. Continuité (lissage d'accélération): pénalise les ruptures brusques de dérivée seconde
    2. Ordre temporel (ranking contrastif): ||z_t - z_{t+1}|| < ||z_t - z_{t+2}||
    3. Préservation des transitions (anti-effondrement): empêche la séquence de devenir constante
    """
    def __init__(self, margin: float = 0.1, continuity_weight: float = 1.0,
                 order_weight: float = 0.5, dynamic_weight: float = 0.2):
        super().__init__()
        self.margin = margin
        self.continuity_weight = continuity_weight
        self.order_weight = order_weight
        self.dynamic_weight = dynamic_weight

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_embeds: Séquence temporelle de tokens (batch_size, time_steps, dim)
        Returns:
            loss: Scalaire de perte temporelle
        """
        b, t, d = seq_embeds.shape
        if t < 3:
            return torch.tensor(0.0, device=seq_embeds.device)

        # 1. Continuité: pénalité de dérivée seconde (accélération discrète: z_{t+2} - 2*z_{t+1} + z_t)
        second_diff = seq_embeds[:, 2:, :] - 2 * seq_embeds[:, 1:-1, :] + seq_embeds[:, :-2, :]
        continuity_loss = torch.mean(second_diff ** 2)

        # 2. Ordre temporel: z_t doit être plus proche de z_{t+1} que de z_{t+2}
        dist_1 = torch.norm(seq_embeds[:, :-2, :] - seq_embeds[:, 1:-1, :], p=2, dim=-1) # (B, T-2)
        dist_2 = torch.norm(seq_embeds[:, :-2, :] - seq_embeds[:, 2:, :], p=2, dim=-1)   # (B, T-2)
        order_loss = torch.mean(F.relu(dist_1 - dist_2 + self.margin))

        # 3. Transitions temporelles dynamiques (empêche l'effondrement constant ||z_{t+1} - z_t|| = 0)
        step_diffs = torch.norm(seq_embeds[:, 1:, :] - seq_embeds[:, :-1, :], p=2, dim=-1) # (B, T-1)
        mean_step = torch.mean(step_diffs)
        # Pénalise si le pas moyen tombe en dessous d'un seuil minimum
        collapse_penalty = F.relu(0.05 - mean_step)

        total_loss = (self.continuity_weight * continuity_loss +
                      self.order_weight * order_loss +
                      self.dynamic_weight * collapse_penalty)
        return total_loss


class RobustnessLoss(nn.Module):
    """
    Perte de robustesse (Bloc 9).
    Exige l'invariance aux perturbations (bruit, masque, drop modal):
    1. Invariance des embeddings sémantiques: L_rep(z, z_pert)
    2. Préservation de la cohérence de structure de graphe: L_adj(A, A_pert)
    """
    def __init__(self, rep_weight: float = 1.0, graph_weight: float = 0.5):
        super().__init__()
        self.rep_weight = rep_weight
        self.graph_weight = graph_weight

    def forward(self, original_embed: torch.Tensor, perturbed_embed: torch.Tensor,
                original_adj: Optional[torch.Tensor] = None,
                perturbed_adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            original_embed: Embedding original non corrompu (B, D)
            perturbed_embed: Embedding après perturbation (B, D)
            original_adj: Adjacence originale (B, N, N)
            perturbed_adj: Adjacence après perturbation (B, N, N)
        Returns:
            loss: Scalaire de perte de robustesse
        """
        # Distance cosinus + MSE sur les représentations
        orig_norm = F.normalize(original_embed, p=2, dim=-1)
        pert_norm = F.normalize(perturbed_embed, p=2, dim=-1)
        cos_loss = 1.0 - (orig_norm * pert_norm).sum(dim=-1).mean()
        mse_loss = F.mse_loss(original_embed, perturbed_embed)
        rep_loss = cos_loss + mse_loss

        # Cohérence structurelle du graphe
        if original_adj is not None and perturbed_adj is not None:
            graph_loss = F.mse_loss(original_adj, perturbed_adj)
        else:
            graph_loss = torch.tensor(0.0, device=original_embed.device)

        return self.rep_weight * rep_loss + self.graph_weight * graph_loss


class CompositeMultimodalLoss(nn.Module):
    """
    Agrégateur des 5 pertes de supervision avec pondérations multi-phases.
    """
    def __init__(self, lambda_sem: float = 1.0, lambda_geom: float = 0.5,
                 lambda_top: float = 0.1, lambda_rob: float = 0.1, lambda_temp: float = 0.1):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.lambda_geom = lambda_geom
        self.lambda_top = lambda_top
        self.lambda_rob = lambda_rob
        self.lambda_temp = lambda_temp

        self.sem_loss = SemanticLoss()
        self.geom_loss = GeometricLoss()
        self.top_loss = TopologicalLoss()
        self.temp_loss = TemporalLoss()
        self.rob_loss = RobustnessLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], phase: int = 3) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Dictionnaire retourné par le modèle multimodal
            phase: Phase d'entraînement (1: sémantique/géom, 2: +topologique, 3: +robustesse/temporel)
        Returns:
            total_loss: Tenseur scalaire pour la rétropropagation
            loss_dict: Dictionnaire des valeurs individuelles détachées
        """
        device = outputs['image_embed'].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict: Dict[str, float] = {}

        # Phase I: Sémantique + Géométrique
        if phase >= 1:
            l_sem = self.sem_loss(outputs['image_embed'], outputs['text_embed'])
            l_geom = self.geom_loss(
                outputs['shared_image'], outputs['shared_text'],
                outputs['image_embed'], outputs['text_embed']
            )
            total_loss = total_loss + self.lambda_sem * l_sem + self.lambda_geom * l_geom
            loss_dict['L_sem'] = float(l_sem.item())
            loss_dict['L_geom'] = float(l_geom.item())

        # Phase II: Topologique
        if phase >= 2:
            l_top = self.top_loss(
                outputs['topo_image'], outputs['topo_text'],
                outputs['filt_image'], outputs['filt_text']
            )
            total_loss = total_loss + self.lambda_top * l_top
            loss_dict['L_top'] = float(l_top.item())

        # Phase III: Robustesse + Temporel
        if phase >= 3:
            l_rob = self.rob_loss(
                outputs['image_embed'], outputs['perturbed_embed'],
                outputs.get('adjacency'), outputs.get('perturbed_adjacency')
            )
            l_temp = self.temp_loss(outputs['video_sequence'])
            total_loss = total_loss + self.lambda_rob * l_rob + self.lambda_temp * l_temp
            loss_dict['L_robust'] = float(l_rob.item())
            loss_dict['L_temp'] = float(l_temp.item())

        loss_dict['total_loss'] = float(total_loss.item())
        return total_loss, loss_dict