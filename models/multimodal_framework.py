# -*- coding: utf-8 -*-
"""
Architecture complète Topo-Multimodal (Synthèse de 'archi.png').
Intègre l'ensemble des modules:
1. Encodeurs Spécialisés (Image, Texte, Audio, Vidéo) + Projecteurs P_m
2. Semantic Latent Bank (Cross-Attention) + Factorisation Shared/Private + Fusion Adaptative
3. Graphe Latent Adaptatif (similarité + relations + incertitude)
4. Filtration Adaptative Multi-échelle (f_theta >= 0, Lipschitz)
5. Homologie Persistante (H0 et H1)
6. Encodage Topologique (Persistence Landscapes -> Tokens H0 et H1)
7. Attention Topologique & Espace Structurel Commun
8. Branche Temporelle
9. Module de Robustesse
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import ImageEncoder, TextEncoder, AudioEncoder, VideoEncoder, ModalityProjector
from models.latent_bank import SemanticLatentBank, SharedPrivateFactorization, AdaptiveFusion
from models.topology import AdaptiveLatentGraph, LearnedFiltration, PersistentHomology, TopologicalEncoder
from models.attention import TopologicalAttention, TopologicalTransformerBlock
from models.robustness import RobustnessPerturbationModule


class TopoMultimodalFramework(nn.Module):
    """
    Modèle multimodal complet guidé par la topologie algébrique.
    Conforme point par point au schéma d'architecture théorique 'archi.png'.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_cfg = config.get('models', {})

        self.latent_dim = int(model_cfg.get('latent_dim', 256))
        self.num_latent_tokens = int(model_cfg.get('num_latent_tokens', 32))
        self.num_shared_tokens = int(model_cfg.get('num_shared_tokens', 16))
        self.num_private_tokens = int(model_cfg.get('num_private_tokens', 16))
        self.num_h0_tokens = int(model_cfg.get('num_h0_tokens', 4))
        self.num_h1_tokens = int(model_cfg.get('num_h1_tokens', 4))

        # -------------------------------------------------------------
        # 1. ENCODEURS SPÉCIALISÉS & PROJECTEURS DE MODALITÉ P_m
        # -------------------------------------------------------------
        self.image_encoder = ImageEncoder(model_name=model_cfg.get('encoder_image', "google/vit-base-patch16-224"))
        self.text_encoder = TextEncoder(model_name=model_cfg.get('encoder_text', "bert-base-uncased"))
        self.audio_encoder = AudioEncoder(model_name=model_cfg.get('encoder_audio', "MIT/ast-finetuned-audioset-10-10-0.4593"))
        self.video_encoder = VideoEncoder(model_name=model_cfg.get('encoder_video', "MCG-NJU/videomae-base"))

        self.image_proj = ModalityProjector(self.image_encoder.output_dim, self.latent_dim)
        self.text_proj = ModalityProjector(self.text_encoder.output_dim, self.latent_dim)
        self.audio_proj = ModalityProjector(self.audio_encoder.output_dim, self.latent_dim)
        self.video_proj = ModalityProjector(self.video_encoder.output_dim, self.latent_dim)

        # -------------------------------------------------------------
        # 2. SEMANTIC LATENT BANK + SPLIT SHARED / PRIVATE + FUSION
        # -------------------------------------------------------------
        self.latent_bank = SemanticLatentBank(
            num_tokens=self.num_latent_tokens,
            latent_dim=self.latent_dim,
            num_heads=4
        )
        self.shared_private = SharedPrivateFactorization(
            num_tokens=self.num_latent_tokens,
            num_shared=self.num_shared_tokens,
            latent_dim=self.latent_dim
        )
        self.adaptive_fusion = AdaptiveFusion(latent_dim=self.latent_dim)

        # -------------------------------------------------------------
        # 3 & 4. GRAPHE LATENT ADAPTATIF & FILTRATION
        # -------------------------------------------------------------
        self.latent_graph = AdaptiveLatentGraph(latent_dim=self.latent_dim, hidden_dim=128)
        self.filtration = LearnedFiltration(latent_dim=self.latent_dim, num_scales=4)

        # -------------------------------------------------------------
        # 5 & 6. HOMOLOGIE PERSISTANTE & ENCODAGE TOPOLOGIQUE
        # -------------------------------------------------------------
        self.persistent_homology = PersistentHomology(max_dimension=1)
        self.topological_encoder = TopologicalEncoder(
            latent_dim=self.latent_dim,
            num_h0_tokens=self.num_h0_tokens,
            num_h1_tokens=self.num_h1_tokens
        )

        # -------------------------------------------------------------
        # 7. ATTENTION TOPOLOGIQUE & ESPACE STRUCTUREL COMMUN
        # -------------------------------------------------------------
        self.topo_attention_block = TopologicalTransformerBlock(
            latent_dim=self.latent_dim,
            num_heads=4,
            dropout=0.1
        )

        # Fusion des tokens structurés partagés avec les tokens topologiques H0 et H1
        self.structural_fusion = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU()
        )

        # Têtes de projection finale pour l'espace multimodal
        self.global_embed_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )

        # -------------------------------------------------------------
        # 8 & 9. BRANCHE TEMPORELLE & MODULE DE ROBUSTESSE
        # -------------------------------------------------------------
        self.temporal_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )
        self.robustness_module = RobustnessPerturbationModule(
            noise_std=0.05,
            mask_prob=0.15,
            modality_drop_prob=0.1
        )

    def extract_modality_tokens(self, image: torch.Tensor, input_ids: torch.Tensor,
                               attention_mask: torch.Tensor, audio: torch.Tensor,
                               video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Passe chaque modalité dans son encodeur spécialisé et son projecteur P_m.
        """
        # Image
        img_raw = self.image_encoder(image)
        img_tokens = self.image_proj(img_raw)

        # Texte
        txt_raw = self.text_encoder(input_ids, attention_mask)
        txt_tokens = self.text_proj(txt_raw)

        # Audio
        aud_raw = self.audio_encoder(audio)
        aud_tokens = self.audio_proj(aud_raw)

        # Vidéo
        vid_raw = self.video_encoder(video)
        vid_tokens = self.video_proj(vid_raw)

        return {
            'image': img_tokens,
            'text': txt_tokens,
            'audio': aud_tokens,
            'video': vid_tokens
        }

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, audio: torch.Tensor,
                video: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass complet avec calcul des représentations et diagrammes topologiques.
        """
        # 1. Encodage et projection de modalité P_m (Bloc 1)
        modality_tokens = self.extract_modality_tokens(image, input_ids, attention_mask, audio, video)

        # 2. Semantic Latent Bank (Cross-Attention) (Bloc 2)
        img_latents = self.latent_bank(modality_tokens['image'])
        txt_latents = self.latent_bank(modality_tokens['text'])
        aud_latents = self.latent_bank(modality_tokens['audio'])
        vid_latents = self.latent_bank(modality_tokens['video'])

        # Split Shared / Private (Bloc 2)
        s_img, p_img = self.shared_private(img_latents)
        s_txt, p_txt = self.shared_private(txt_latents)
        s_aud, p_aud = self.shared_private(aud_latents)
        s_vid, p_vid = self.shared_private(vid_latents)

        # Fusion Adaptative (Bloc 2)
        fused_shared, alpha_weights = self.adaptive_fusion([s_img, s_txt, s_aud, s_vid])

        # 3. Graphe Latent Adaptatif (similarité + relations + incertitude) (Bloc 3)
        adjacency, uncertainty = self.latent_graph(fused_shared)

        # Topologies spécifiques par modalité pour alignement topologique direct
        adj_img, _ = self.latent_graph(s_img)
        adj_txt, _ = self.latent_graph(s_txt)

        # 4. Filtration Adaptative Multi-échelle (f_theta >= 0, Lipschitz) (Bloc 4)
        filt_values_fused = self.filtration(fused_shared)
        filt_values_img = self.filtration(s_img)
        filt_values_txt = self.filtration(s_txt)

        # 5. Homologie Persistante H0/H1 via Gudhi (Bloc 5)
        # Calcul sur la topologie fusionnée et les modalités principales
        diagrams_fused = self.persistent_homology.compute_diagrams(adjacency, filt_values_fused)
        diagrams_img = self.persistent_homology.compute_diagrams(adj_img, filt_values_img)
        diagrams_txt = self.persistent_homology.compute_diagrams(adj_txt, filt_values_txt)

        # 6. Encodage Topologique (Bloc 6) -> Tokens H0 et H1
        h0_tokens_fused, h1_tokens_fused = self.topological_encoder(
            diagrams_fused, fused_shared, filt_values_fused
        )
        h0_tokens_img, h1_tokens_img = self.topological_encoder(
            diagrams_img, s_img, filt_values_img
        )
        h0_tokens_txt, h1_tokens_txt = self.topological_encoder(
            diagrams_txt, s_txt, filt_values_txt
        )

        topo_tokens_img = torch.cat([h0_tokens_img, h1_tokens_img], dim=1)
        topo_tokens_txt = torch.cat([h0_tokens_txt, h1_tokens_txt], dim=1)

        # 7. Attention Topologique & Espace Structurel Commun (Bloc 7)
        # L'attention sur les tokens partagés est biaisée par l'adjacence du graphe latent
        structured_shared, attn_weights = self.topo_attention_block(
            fused_shared,
            topological_bias=adjacency
        )

        # Combinaison dans l'espace structurel commun: tokens partagés + tokens H0/H1
        structural_tokens = torch.cat([structured_shared, h0_tokens_fused, h1_tokens_fused], dim=1)
        structural_tokens = self.structural_fusion(structural_tokens)

        # Embeddings globaux pour l'espace multimodal final
        image_embed = self.global_embed_head(s_img.mean(dim=1))
        text_embed = self.global_embed_head(s_txt.mean(dim=1))
        multimodal_embed = self.global_embed_head(structural_tokens.mean(dim=1))

        # 8. Branche Temporelle (Audio / Vidéo) (Bloc 8)
        # Séquence temporelle pour évaluer la continuité, l'ordre et les transitions
        video_sequence = self.temporal_head(s_vid)

        # 9. Module de Robustesse (Perturbations) (Bloc 9)
        # Applique le bruit et le masquage stochastique
        perturbed_s_img = self.robustness_module(s_img, training=self.training)
        perturbed_image_embed = self.global_embed_head(perturbed_s_img.mean(dim=1))
        perturbed_adj, _ = self.latent_graph(perturbed_s_img)

        return {
            'multimodal_embed': multimodal_embed,
            'image_embed': image_embed,
            'text_embed': text_embed,
            'shared_image': s_img,
            'shared_text': s_txt,
            'private_image': p_img,
            'private_text': p_txt,
            'fused_shared': fused_shared,
            'adjacency': adjacency,
            'uncertainty': uncertainty,
            'filt_fused': filt_values_fused,
            'filt_image': filt_values_img,
            'filt_text': filt_values_txt,
            'diagrams_fused': diagrams_fused,
            'diagrams_img': diagrams_img,
            'diagrams_txt': diagrams_txt,
            'h0_tokens': h0_tokens_fused,
            'h1_tokens': h1_tokens_fused,
            'topo_image': topo_tokens_img,
            'topo_text': topo_tokens_txt,
            'structural_tokens': structural_tokens,
            'topological_attention_weights': attn_weights,
            'video_sequence': video_sequence,
            'perturbed_embed': perturbed_image_embed,
            'perturbed_adjacency': perturbed_adj,
            'alpha_weights': alpha_weights
        }
