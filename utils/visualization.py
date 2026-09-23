# -*- coding: utf-8 -*-
"""
Outils de visualisation pour l'architecture Topo-Multimodal.
Comprend:
- Visualisation des échantillons du dataset
- Visualisation de l'espace latent partagé vs privé (PCA / t-SNE)
- Diagrammes de persistance H0 et H1 (Birth-Death)
- Paysages de persistance (Persistence Landscapes)
- Matrice d'attention topologique
- Courbes d'entraînement et de validation multi-phases
"""

import os
from typing import Dict, List, Optional, Union, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch


def _finalize_plot(fig, save_path: Optional[str] = None, show: bool = True):
    """Sauvegarde la figure et libère la mémoire."""
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure sauvegardée: {save_path}")
    if show and matplotlib.get_backend() != "Agg" and os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)


def visualize_dataset_samples(dataloader, num_samples: int = 4, save_path: Optional[str] = None):
    """Visualise les premiers échantillons du DataLoader."""
    batch = next(iter(dataloader))
    actual_samples = min(num_samples, batch['image'].size(0))
    fig, axes = plt.subplots(1, actual_samples, figsize=(4 * actual_samples, 4))
    if actual_samples == 1:
        axes = [axes]

    for i in range(actual_samples):
        img = batch['image'][i].detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i].imshow(img)
        q_text = batch['question'][i] if isinstance(batch['question'], list) else "Échantillon"
        axes[i].set_title(f"Q: {q_text[:25]}...", fontsize=9)
        axes[i].axis('off')

    _finalize_plot(fig, save_path=save_path)


def visualize_latent_space(shared_embeds: torch.Tensor, private_embeds: torch.Tensor,
                           labels: Optional[np.ndarray] = None, save_path: Optional[str] = None):
    """
    Visualise les espaces latents Partagé (Shared S_c) et Privé (Private S_p) via ACP.
    """
    s_flat = shared_embeds.view(shared_embeds.size(0), -1).detach().cpu().numpy()
    p_flat = private_embeds.view(private_embeds.size(0), -1).detach().cpu().numpy()

    n_samples = s_flat.shape[0]
    if n_samples < 2:
        print("Nombre d'échantillons insuffisant pour l'ACP (< 2).")
        return

    n_comp = min(2, n_samples)
    pca_s = PCA(n_components=n_comp)
    pca_p = PCA(n_components=n_comp)
    s_2d = pca_s.fit_transform(s_flat)
    p_2d = pca_p.fit_transform(p_flat)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cmap_opt = 'viridis' if labels is not None else None
    sc1 = ax1.scatter(s_2d[:, 0], s_2d[:, 1] if n_comp > 1 else np.zeros_like(s_2d[:, 0]),
                      c=labels if labels is not None else '#2563eb', cmap=cmap_opt, edgecolors='k', s=60)
    ax1.set_title("Espace Latent Partagé (Shared $S_c$)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.grid(True, linestyle="--", alpha=0.5)

    sc2 = ax2.scatter(p_2d[:, 0], p_2d[:, 1] if n_comp > 1 else np.zeros_like(p_2d[:, 0]),
                      c=labels if labels is not None else '#dc2626', cmap=cmap_opt, edgecolors='k', s=60)
    ax2.set_title("Espace Latent Privé (Private $S_p$)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.grid(True, linestyle="--", alpha=0.5)

    _finalize_plot(fig, save_path=save_path)


def visualize_persistence_diagram(diagram_dict: Union[Dict[str, np.ndarray], List[Tuple[int, Tuple[float, float]]]],
                                  title: str = "Diagramme de Persistance ($H_0$ et $H_1$)",
                                  save_path: Optional[str] = None):
    """
    Trace le diagramme de persistance (Birth vs Death) pour H0 et H1.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    h0_pts = []
    h1_pts = []

    if isinstance(diagram_dict, dict):
        h0_pts = diagram_dict.get('H0', [])
        h1_pts = diagram_dict.get('H1', [])
    elif isinstance(diagram_dict, list):
        for item in diagram_dict:
            dim, (b, d) = item
            if dim == 0:
                h0_pts.append([b, d])
            elif dim == 1:
                h1_pts.append([b, d])

    h0_pts = np.array(h0_pts) if len(h0_pts) > 0 else np.empty((0, 2))
    h1_pts = np.array(h1_pts) if len(h1_pts) > 0 else np.empty((0, 2))

    max_val = 2.0
    if h0_pts.size > 0:
        h0_valid = h0_pts[~np.isinf(h0_pts[:, 1])]
        if h0_valid.size > 0:
            max_val = max(max_val, float(np.max(h0_valid)) * 1.1)
    if h1_pts.size > 0:
        h1_valid = h1_pts[~np.isinf(h1_pts[:, 1])]
        if h1_valid.size > 0:
            max_val = max(max_val, float(np.max(h1_valid)) * 1.1)

    # Diagonale birth = death
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, label="Diagonale ($b = d$)")

    if h0_pts.size > 0:
        # Remplacer les infinis pour l'affichage
        h0_deaths = np.where(np.isinf(h0_pts[:, 1]), max_val * 0.95, h0_pts[:, 1])
        ax.scatter(h0_pts[:, 0], h0_deaths, c='#2563eb', label=f"$H_0$ ({len(h0_pts)} composantes)",
                   alpha=0.8, edgecolors='none', s=45)

    if h1_pts.size > 0:
        h1_deaths = np.where(np.isinf(h1_pts[:, 1]), max_val * 0.95, h1_pts[:, 1])
        ax.scatter(h1_pts[:, 0], h1_deaths, c='#dc2626', marker='^', label=f"$H_1$ ({len(h1_pts)} cycles)",
                   alpha=0.8, edgecolors='none', s=55)

    ax.set_xlim(-0.05, max_val)
    ax.set_ylim(-0.05, max_val)
    ax.set_xlabel("Naissance ($Birth$)", fontsize=11)
    ax.set_ylabel("Mort ($Death$)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)

    _finalize_plot(fig, save_path=save_path)


def visualize_topological_attention(attention_matrix: torch.Tensor,
                                    title: str = "Attention Topologique ($A_{ij}$)",
                                    save_path: Optional[str] = None):
    """
    Affiche la carte thermique de la matrice d'attention topologique.
    """
    if attention_matrix.dim() == 4:
        # (B, H, N, N) -> moyenne sur les têtes et premier item du batch
        mat = attention_matrix[0].mean(dim=0).detach().cpu().numpy()
    elif attention_matrix.dim() == 3:
        mat = attention_matrix[0].detach().cpu().numpy()
    else:
        mat = attention_matrix.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap='magma', interpolation='nearest')
    plt.colorbar(im, ax=ax, label="Poids d'attention")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Tokens clés $j$")
    ax.set_ylabel("Tokens requêtes $i$")

    _finalize_plot(fig, save_path=save_path)


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                         metrics: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Trace l'évolution des pertes et des métriques à travers les époques.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Courbes de perte
    ax1.plot(train_losses, label='Train Loss', color='#2563eb', linewidth=2)
    if val_losses:
        ax1.plot(val_losses, label='Val Loss', color='#dc2626', linestyle='--', linewidth=2)
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Perte Composite')
    ax1.set_title("Évolution des Pertes", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 2. Métriques
    colors = ['#10b981', '#f59e0b', '#6366f1', '#ec4899']
    for idx, (m_name, values) in enumerate(metrics.items()):
        if values:
            ax2.plot(values, label=m_name, color=colors[idx % len(colors)], linewidth=2)
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Score')
    ax2.set_title("Métriques d'Évaluation Cross-Modales", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    _finalize_plot(fig, save_path=save_path)