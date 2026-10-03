# -*- coding: utf-8 -*-
"""
Suite de tests unitaires et d'intégration pour l'architecture Topo-Multimodal.
Vérifie la conformité mathématique avec le schéma 'archi.png':
- Encoders et projecteurs P_m
- Semantic Latent Bank, factorisation et fusion adaptative
- Graphe latent adaptatif (similarité + relations + incertitude)
- Filtration (f_theta >= 0, Lipschitz)
- Homologie persistante (H0, H1) et encodage topologique différentiable
- Attention topologique
- Les 5 pertes de supervision (L_sem, L_geom, L_top, L_temp, L_robust)
- Rétropropagation de bout en bout
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from models.encoders import ModalityProjector, AudioEncoder, VideoEncoder
from models.latent_bank import SemanticLatentBank, SharedPrivateFactorization, AdaptiveFusion
from models.topology import AdaptiveLatentGraph, LearnedFiltration, PersistentHomology, TopologicalEncoder
from models.attention import TopologicalAttention, TopologicalTransformerBlock
from models.robustness import RobustnessPerturbationModule
from losses.losses import (
    SemanticLoss,
    GeometricLoss,
    TopologicalLoss,
    TemporalLoss,
    RobustnessLoss,
    CompositeMultimodalLoss
)
from utils.metrics import recall_at_k, mean_average_precision


def test_modality_projector():
    proj = ModalityProjector(input_dim=768, latent_dim=256)
    x = torch.randn(2, 10, 768)
    out = proj(x)
    assert out.shape == (2, 10, 256)


def test_semantic_latent_bank():
    bank = SemanticLatentBank(num_tokens=32, latent_dim=256, num_heads=4)
    tokens = torch.randn(2, 20, 256)
    latents = bank(tokens)
    assert latents.shape == (2, 32, 256)


def test_shared_private_factorization():
    sp = SharedPrivateFactorization(num_tokens=32, num_shared=16, latent_dim=256)
    latents = torch.randn(2, 32, 256)
    shared, private = sp(latents)
    assert shared.shape == (2, 16, 256)
    assert private.shape == (2, 16, 256)
    ortho_loss = sp.orthogonality_loss(shared, private)
    assert ortho_loss.ndim == 0 and ortho_loss >= 0.0


def test_adaptive_fusion():
    fusion = AdaptiveFusion(latent_dim=256)
    s1 = torch.randn(2, 16, 256)
    s2 = torch.randn(2, 16, 256)
    fused, alphas = fusion([s1, s2])
    assert fused.shape == (2, 16, 256)
    assert alphas.shape == (2, 2, 16, 1)
    # Vérifier que les poids somment à 1 le long de la dimension des modalités
    assert torch.allclose(alphas.sum(dim=1), torch.ones(2, 16, 1), atol=1e-5)


def test_adaptive_latent_graph():
    graph = AdaptiveLatentGraph(latent_dim=256, hidden_dim=128)
    s = torch.randn(2, 16, 256)
    adj, unc = graph(s)
    assert adj.shape == (2, 16, 16)
    assert unc.shape == (2, 16, 16)
    # Symétrie et bornes [0, 1]
    assert torch.allclose(adj, adj.transpose(1, 2), atol=1e-5)
    assert torch.all(adj >= 0.0) and torch.all(adj <= 1.0)


def test_learned_filtration():
    filtration = LearnedFiltration(latent_dim=256, num_scales=4)
    s = torch.randn(2, 16, 256)
    vals = filtration(s)
    assert vals.shape == (2, 16)
    # Condition stricte du schéma: f_theta >= 0
    assert torch.all(vals >= 0.0)
    # Échelles croissantes
    scales = filtration.scales
    assert len(scales) == 4
    assert torch.all(torch.diff(scales) > 0.0)


def test_persistent_homology_and_encoding():
    adj = torch.rand(2, 8, 8)
    adj = 0.5 * (adj + adj.transpose(1, 2))
    filt_vals = torch.rand(2, 8)

    ph = PersistentHomology(max_dimension=1)
    diagrams = ph.compute_diagrams(adj, filt_vals)
    assert len(diagrams) == 2
    assert 'H0' in diagrams[0] and 'H1' in diagrams[0]

    # Encodage topologique en tokens
    enc = TopologicalEncoder(latent_dim=256, num_h0_tokens=4, num_h1_tokens=4)
    shared = torch.randn(2, 8, 256, requires_grad=True)
    h0_tok, h1_tok = enc(diagrams, shared, filt_vals)
    assert h0_tok.shape == (2, 4, 256)
    assert h1_tok.shape == (2, 4, 256)

    # Rétropropagation
    (h0_tok.sum() + h1_tok.sum()).backward()
    assert shared.grad is not None


def test_topological_attention():
    block = TopologicalTransformerBlock(latent_dim=256, num_heads=4)
    x = torch.randn(2, 16, 256)
    bias = torch.rand(2, 16, 16)
    out, weights = block(x, topological_bias=bias)
    assert out.shape == (2, 16, 256)
    assert weights.shape == (2, 4, 16, 16)


def test_supervision_losses():
    b, k, d = 4, 16, 256
    img_emb = torch.randn(b, d, requires_grad=True)
    txt_emb = torch.randn(b, d, requires_grad=True)

    # 1. Semantic
    sem_fn = SemanticLoss()
    l_sem = sem_fn(img_emb, txt_emb)
    assert l_sem.ndim == 0 and l_sem > 0

    # 2. Geometric
    geom_fn = GeometricLoss()
    s_img = torch.randn(b, k, d, requires_grad=True)
    s_txt = torch.randn(b, k, d, requires_grad=True)
    l_geom = geom_fn(s_img, s_txt, img_emb, txt_emb)
    assert l_geom.ndim == 0 and l_geom >= 0

    # 3. Topological
    top_fn = TopologicalLoss()
    topo_img = torch.randn(b, 8, d, requires_grad=True)
    topo_txt = torch.randn(b, 8, d, requires_grad=True)
    l_top = top_fn(topo_img, topo_txt)
    assert l_top.ndim == 0 and l_top >= 0

    # 4. Temporal
    temp_fn = TemporalLoss()
    seq = torch.randn(b, 12, d, requires_grad=True)
    l_temp = temp_fn(seq)
    assert l_temp.ndim == 0 and l_temp >= 0

    # 5. Robustness
    rob_fn = RobustnessLoss()
    l_rob = rob_fn(img_emb, img_emb + 0.05 * torch.randn_like(img_emb))
    assert l_rob.ndim == 0 and l_rob >= 0

    # Composite Loss
    outputs = {
        'image_embed': img_emb,
        'text_embed': txt_emb,
        'shared_image': s_img,
        'shared_text': s_txt,
        'topo_image': topo_img,
        'topo_text': topo_txt,
        'filt_image': torch.randn(b, k),
        'filt_text': torch.randn(b, k),
        'video_sequence': seq,
        'perturbed_embed': img_emb + 0.01 * torch.randn_like(img_emb),
        'adjacency': torch.rand(b, k, k),
        'perturbed_adjacency': torch.rand(b, k, k)
    }
    composite = CompositeMultimodalLoss()
    tot_loss, d_loss = composite(outputs, phase=3)
    assert tot_loss.ndim == 0
    assert 'L_sem' in d_loss and 'L_top' in d_loss and 'L_temp' in d_loss

    tot_loss.backward()
    assert img_emb.grad is not None
    assert s_img.grad is not None
    assert seq.grad is not None


def test_metrics():
    # Parfaite identité (sims maximales sur la diagonale)
    x = torch.eye(8)
    r1 = recall_at_k(x, x, k=1)
    assert r1 == 1.0
    map_score = mean_average_precision(x, x)
    assert map_score == 1.0
