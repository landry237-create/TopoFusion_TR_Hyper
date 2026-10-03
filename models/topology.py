# -*- coding: utf-8 -*-
"""
Module de Topologie Algébrique (Blocs 3, 4, 5 et 6 du schéma architectural).
Comprend:
- Bloc 3: Graphe Latent Adaptatif (similarité + relations + incertitude)
- Bloc 4: Filtration Adaptative Multi-échelle (f_theta >= 0, Lipschitz via normalisation spectrale)
- Bloc 5: Homologie Persistante (calcul de H0 et H1 via Gudhi)
- Bloc 6: Encodage Topologique (Persistence Landscapes / Images différentiables -> Tokens H0 et H1)
"""

from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gudhi


class AdaptiveLatentGraph(nn.Module):
    """
    Bloc 3: Construction d'un graphe adaptatif sur les tokens partagés S_c.
    Poids d'arêtes appris combinant:
    1. Similarité sémantique (produit scalaire normalisé / cosine)
    2. Relations relationnelles profondes (MLP relationnel)
    3. Incertitude d'arête (modélisation bayésienne/aléatoire de la variance)
    """
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # 1. Projecteurs bilinéaires de similarité
        self.q_sim = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.k_sim = nn.Linear(latent_dim, hidden_dim, bias=False)

        # 2. Réseau de relations par paires (concatenation, différence absolue, produit de Hadamard)
        self.relation_net = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # 3. Réseau d'incertitude (prédit la variance log-sigma^2 pour chaque arête)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Température apprenable
        self.scale = nn.Parameter(torch.tensor(1.0 / (hidden_dim ** 0.5)))

    def forward(self, shared_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            shared_tokens: (batch_size, num_nodes, latent_dim)
        Returns:
            adjacency: Matrice d'adjacence symétrique dans [0, 1] (batch_size, num_nodes, num_nodes)
            uncertainty: Matrice d'incertitude sigma (batch_size, num_nodes, num_nodes)
        """
        b, n, d = shared_tokens.shape

        # 1. Similarité
        q = self.q_sim(shared_tokens) # (B, N, H)
        k = self.k_sim(shared_tokens) # (B, N, H)
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (B, N, N)

        # 2. Relations
        node_i = shared_tokens.unsqueeze(2).expand(-1, -1, n, -1)
        node_j = shared_tokens.unsqueeze(1).expand(-1, n, -1, -1)
        diff = torch.abs(node_i - node_j)
        prod = node_i * node_j
        rel_feat = torch.cat([node_i, node_j, diff, prod], dim=-1)
        rel = self.relation_net(rel_feat).squeeze(-1) # (B, N, N)

        # 3. Incertitude
        pair_feat = torch.cat([node_i, node_j], dim=-1)
        log_var = self.uncertainty_net(pair_feat).squeeze(-1) # (B, N, N)
        # Bruit / incertitude bornée
        log_var = torch.clamp(log_var, min=-4.0, max=4.0)
        uncertainty = torch.exp(0.5 * log_var)

        # Pondération modulée par l'incertitude: (Sim + Rel) / sigma
        edge_logits = (sim + rel) / (uncertainty + 1e-6)
        adjacency = torch.sigmoid(edge_logits)

        # Symétrisation stricte et préservation de la diagonale (boucle réflexive = 1.0)
        adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
        eye = torch.eye(n, device=shared_tokens.device).unsqueeze(0).expand(b, -1, -1)
        adjacency = adjacency * (1.0 - eye) + eye

        uncertainty = 0.5 * (uncertainty + uncertainty.transpose(1, 2))

        return adjacency, uncertainty


class LearnedFiltration(nn.Module):
    """
    Bloc 4: Filtration Adaptative Multi-échelle.
    Propriétés mathématiques requises par le schéma:
    - f_theta >= 0: Garantie via activation Softplus
    - Lipschitz-continue: Garantie en appliquant la normalisation spectrale (spectral_norm)
      sur chaque couche linéaire pour borner la constante de Lipschitz (L <= 1).
    - Multi-échelle: produit les valeurs nodales de filtration f_theta(v) ainsi que
      les seuils multi-échelles (epsilon_1, epsilon_2, ..., epsilon_M).
    """
    def __init__(self, latent_dim: int = 256, num_scales: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_scales = num_scales

        # Réseau 1-Lipschitz avec spectral_norm
        linear1 = nn.utils.parametrizations.spectral_norm(nn.Linear(latent_dim, latent_dim // 2))
        linear2 = nn.utils.parametrizations.spectral_norm(nn.Linear(latent_dim // 2, 1))

        self.net = nn.Sequential(
            linear1,
            nn.ReLU(),
            linear2
        )

        # Échelles de filtration apprenables et ordonnées (epsilon_1 < epsilon_2 < ...)
        # Initialisées de manière équidistante dans [0.1, 1.0]
        init_scales = torch.linspace(0.1, 1.0, num_scales)
        self.scale_deltas = nn.Parameter(torch.log(torch.diff(torch.cat([torch.tensor([0.05]), init_scales]))))

    @property
    def scales(self) -> torch.Tensor:
        """Retourne les seuils de filtration strictement croissants epsilon_1 < ... < epsilon_M >= 0."""
        return torch.cumsum(F.softplus(self.scale_deltas), dim=0)

    def forward(self, shared_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_tokens: (batch_size, num_nodes, latent_dim)
        Returns:
            filtration_values: (batch_size, num_nodes) strictement >= 0.
        """
        raw_values = self.net(shared_tokens).squeeze(-1)
        # f_theta >= 0
        filtration_values = F.softplus(raw_values)
        return filtration_values


class PersistentHomology(nn.Module):
    """
    Bloc 5: Calcul de l'homologie persistante H0 (composantes connexes) et H1 (cycles 1D).
    Utilise Gudhi SimplexTree avec lower-star filtration sur le graphe latent adaptatif.
    """
    def __init__(self, max_dimension: int = 1):
        super().__init__()
        self.max_dimension = max_dimension

    @torch.no_grad()
    def compute_diagrams(self, adjacency: torch.Tensor, filtration_values: torch.Tensor) -> List[Dict[str, np.ndarray]]:
        """
        Calcul exact des diagrammes de persistance H0 et H1 via Gudhi.
        Args:
            adjacency: (batch_size, num_nodes, num_nodes)
            filtration_values: (batch_size, num_nodes)
        Returns:
            diagrams: Liste de dictionnaires par échantillon du batch:
                {'H0': np.ndarray(N0, 2), 'H1': np.ndarray(N1, 2)}
        """
        batch_size, num_nodes, _ = adjacency.shape
        batch_diagrams = []

        adj_np = adjacency.detach().cpu().numpy()
        filt_np = filtration_values.detach().cpu().numpy()

        for b in range(batch_size):
            st = gudhi.SimplexTree()

            # 1. Insertion des sommets avec leur valeur de filtration f(v)
            for i in range(num_nodes):
                st.insert([i], filtration=float(filt_np[b, i]))

            # 2. Insertion des arêtes: filtration de lower-star f(e_ij) = max(f(v_i), f(v_j)) * (2.0 - A_ij)
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    weight = adj_np[b, i, j]
                    if weight > 0.05: # Seuil d'arête active
                        edge_filt = max(filt_np[b, i], filt_np[b, j]) + (1.0 - weight)
                        st.insert([i, j], filtration=float(edge_filt))

            # 3. Extension aux 2-simplexes (triangles) pour fermer les 1-cycles (H1)
            st.expansion(2)

            # 4. Calcul de persistance
            st.compute_persistence()
            pers = st.persistence()

            h0_pairs = []
            h1_pairs = []
            for dim, (birth, death) in pers:
                # Remplacer l'infini par une valeur finie pour l'encodage
                d_val = (birth + 2.0) if np.isinf(death) else death
                if dim == 0:
                    h0_pairs.append([birth, d_val])
                elif dim == 1:
                    h1_pairs.append([birth, d_val])

            h0_arr = np.array(h0_pairs, dtype=np.float32) if h0_pairs else np.zeros((1, 2), dtype=np.float32)
            h1_arr = np.array(h1_pairs, dtype=np.float32) if h1_pairs else np.zeros((1, 2), dtype=np.float32)

            batch_diagrams.append({'H0': h0_arr, 'H1': h1_arr})

        return batch_diagrams


class TopologicalEncoder(nn.Module):
    """
    Bloc 6: Encodage Topologique Différentiable.
    Transforme les diagrammes de persistance H0 et H1 en descripteurs continus
    (Persistence Landscapes / Persistence Images) puis projette ces descripteurs
    vers des ensembles de tokens structurés:
    - Tokens H0: (batch_size, num_h0_tokens, latent_dim)
    - Tokens H1: (batch_size, num_h1_tokens, latent_dim)

    Intègre également un pont différentiable de bout en bout sur les valeurs de filtration.
    """
    def __init__(self, latent_dim: int = 256, num_h0_tokens: int = 4, num_h1_tokens: int = 4,
                 num_landscape_steps: int = 32, num_landscapes: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_h0_tokens = num_h0_tokens
        self.num_h1_tokens = num_h1_tokens
        self.num_steps = num_landscape_steps
        self.num_landscapes = num_landscapes

        # Grille d'évaluation discrète des paysages t in [0, 2]
        self.register_buffer("grid", torch.linspace(0.0, 2.0, num_landscape_steps))

        landscape_dim = num_landscapes * num_landscape_steps

        # Projecteur H0 vers tokens latents
        self.h0_projector = nn.Sequential(
            nn.Linear(landscape_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_h0_tokens * latent_dim)
        )

        # Projecteur H1 vers tokens latents
        self.h1_projector = nn.Sequential(
            nn.Linear(landscape_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_h1_tokens * latent_dim)
        )

        # Couche différentiable directe (surrogate) pour propager les gradients
        # vers la filtration et le graphe
        self.diff_h0_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        self.diff_h1_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

    def compute_landscapes_from_diagrams(self, diagrams: List[Dict[str, np.ndarray]],
                                         device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule les Persistence Landscapes pour H0 et H1.
        Landscape tent function: Lambda_(b, d)(t) = max(0, min(t - b, d - t))
        """
        batch_size = len(diagrams)
        grid = self.grid.to(device)
        m = self.num_steps
        k_max = self.num_landscapes

        h0_landscapes = torch.zeros(batch_size, k_max * m, device=device)
        h1_landscapes = torch.zeros(batch_size, k_max * m, device=device)

        for b, diag in enumerate(diagrams):
            # Traitement H0
            h0_pts = torch.tensor(diag['H0'], device=device, dtype=torch.float32) # (N, 2)
            if h0_pts.numel() > 0:
                births = h0_pts[:, 0:1] # (N, 1)
                deaths = h0_pts[:, 1:2] # (N, 1)
                t = grid.unsqueeze(0)   # (1, M)
                # Tent function: max(0, min(t - b, d - t))
                tents = torch.clamp(torch.min(t - births, deaths - t), min=0.0) # (N, M)
                # Trier par valeur décroissante pour chaque t pour obtenir les k-ièmes maxima
                if tents.shape[0] >= k_max:
                    topk_tents, _ = torch.topk(tents, k=k_max, dim=0) # (K, M)
                else:
                    pad = torch.zeros(k_max - tents.shape[0], m, device=device)
                    topk_tents = torch.cat([tents, pad], dim=0)
                h0_landscapes[b] = topk_tents.view(-1)

            # Traitement H1
            h1_pts = torch.tensor(diag['H1'], device=device, dtype=torch.float32)
            if h1_pts.numel() > 0 and diag['H1'].shape[0] > 0 and not np.all(diag['H1'] == 0):
                births = h1_pts[:, 0:1]
                deaths = h1_pts[:, 1:2]
                t = grid.unsqueeze(0)
                tents = torch.clamp(torch.min(t - births, deaths - t), min=0.0)
                if tents.shape[0] >= k_max:
                    topk_tents, _ = torch.topk(tents, k=k_max, dim=0)
                else:
                    pad = torch.zeros(k_max - tents.shape[0], m, device=device)
                    topk_tents = torch.cat([tents, pad], dim=0)
                h1_landscapes[b] = topk_tents.view(-1)

        return h0_landscapes, h1_landscapes

    def forward(self, diagrams: List[Dict[str, np.ndarray]], shared_tokens: torch.Tensor,
                filtration_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            diagrams: Diagrammes calculés par PersistentHomology
            shared_tokens: (batch_size, num_shared, latent_dim)
            filtration_values: (batch_size, num_shared)
        Returns:
            h0_tokens: (batch_size, num_h0_tokens, latent_dim)
            h1_tokens: (batch_size, num_h1_tokens, latent_dim)
        """
        b = shared_tokens.size(0)
        device = shared_tokens.device

        # 1. Descripteurs topologiques de paysages de persistance
        h0_land, h1_land = self.compute_landscapes_from_diagrams(diagrams, device)

        # 2. Projection en tokens
        h0_tokens_raw = self.h0_projector(h0_land).view(b, self.num_h0_tokens, self.latent_dim)
        h1_tokens_raw = self.h1_projector(h1_land).view(b, self.num_h1_tokens, self.latent_dim)

        # 3. Couplage différentiable: pondération par les valeurs de filtration f_theta
        # Permet de rétropropager les gradients directement vers f_theta et les tokens partagés
        filt_weight = F.softmax(filtration_values, dim=-1).unsqueeze(-1) # (B, N, 1)
        topo_context = (shared_tokens * filt_weight).sum(dim=1, keepdim=True) # (B, 1, D)

        h0_tokens = h0_tokens_raw + self.diff_h0_head(topo_context.expand(-1, self.num_h0_tokens, -1))
        h1_tokens = h1_tokens_raw + self.diff_h1_head(topo_context.expand(-1, self.num_h1_tokens, -1))

        return h0_tokens, h1_tokens