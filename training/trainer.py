# -*- coding: utf-8 -*-
"""
Boucle d'entraînement et d'évaluation en 3 phases pour l'architecture Topo-Multimodal.
Phases d'entraînement:
- Phase I: Amorçage Sémantique et Géométrique (L_sem + L_geom)
- Phase II: Stabilisation Structurelle et Topologique (+ L_top)
- Phase III: Robustesse et Cohérence Temporelle (+ L_rob + L_temp)
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from tqdm import tqdm

from losses.losses import CompositeMultimodalLoss
from utils.metrics import recall_at_k, mean_average_precision, topological_isometry_score


class Trainer:
    """
    Gestionnaire d'entraînement et de validation multi-phases.
    """
    @staticmethod
    def _as_float(value, default: float = 0.0) -> float:
        """Convertit de façon sécurisée une entrée de configuration en float."""
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise TypeError(f"Valeur numérique attendue mais reçue : {value!r}") from exc
        raise TypeError(f"Valeur numérique attendue mais reçue : {type(value).__name__}")

    def __init__(self, model: nn.Module, dataloader, config: Dict[str, Any], device: str = 'cuda'):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')

        training_cfg = self.config.get('training', {})
        self.lr = self._as_float(training_cfg.get('lr', 1e-4))
        self.weight_decay = self._as_float(training_cfg.get('weight_decay', 1e-5))
        self.lambda_sem = self._as_float(training_cfg.get('lambda_sem', 1.0))
        self.lambda_geom = self._as_float(training_cfg.get('lambda_geom', 0.5))
        self.lambda_top = self._as_float(training_cfg.get('lambda_top', 0.1))
        self.lambda_rob = self._as_float(training_cfg.get('lambda_rob', 0.1))
        self.lambda_temp = self._as_float(training_cfg.get('lambda_temp', 0.1))
        self.max_grad_norm = self._as_float(training_cfg.get('max_grad_norm', 1.0))

        # Optimiseur AdamW avec weight decay sélectif
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Critère multi-pertes composite
        self.criterion = CompositeMultimodalLoss(
            lambda_sem=self.lambda_sem,
            lambda_geom=self.lambda_geom,
            lambda_top=self.lambda_top,
            lambda_rob=self.lambda_rob,
            lambda_temp=self.lambda_temp
        )

        # Historiques d'apprentissage
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.metrics: Dict[str, List[float]] = {
            'Recall@1': [],
            'Recall@5': [],
            'MAP': [],
            'Isometry': []
        }
        self.detailed_losses: Dict[str, List[float]] = {
            'L_sem': [],
            'L_geom': [],
            'L_top': [],
            'L_robust': [],
            'L_temp': []
        }

    def train_epoch(self, epoch: int, phase: int = 1) -> float:
        """
        Exécute une époque d'entraînement pour la phase spécifiée.
        """
        self.model.train()
        total_epoch_loss = 0.0
        phase_loss_accum: Dict[str, float] = {}

        if not self.dataloader:
            return 0.0

        pbar = tqdm(self.dataloader, desc=f"Phase {phase} - Époque {epoch}")

        for step, batch in enumerate(pbar):
            # Déplacement des tenseurs vers le device
            image = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(image, input_ids, attention_mask, audio, video)

            # Calcul des pertes pour la phase
            loss, loss_breakdown = self.criterion(outputs, phase=phase)

            # Backward pass
            loss.backward()

            # Gradient clipping pour la stabilité de l'homologie persistante
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_epoch_loss += loss.item()
            for k, v in loss_breakdown.items():
                phase_loss_accum[k] = phase_loss_accum.get(k, 0.0) + v

            # Mise à jour de la barre de progression
            postfix = {'Loss': f"{loss.item():.4f}"}
            if 'L_sem' in loss_breakdown:
                postfix['L_sem'] = f"{loss_breakdown['L_sem']:.3f}"
            if 'L_top' in loss_breakdown and phase >= 2:
                postfix['L_top'] = f"{loss_breakdown['L_top']:.3f}"
            pbar.set_postfix(postfix)

        num_batches = max(len(self.dataloader), 1)
        avg_loss = total_epoch_loss / num_batches
        self.train_losses.append(avg_loss)

        for k in self.detailed_losses:
            if k in phase_loss_accum:
                self.detailed_losses[k].append(phase_loss_accum[k] / num_batches)

        return avg_loss

    def validate(self, phase: int = 1) -> Tuple[float, float, float]:
        """
        Évalue le modèle sur le dataset de validation.
        Calcule les métriques d'alignement cross-modal (Recall@1, Recall@5, mAP).
        Évite les fuites de mémoire en transférant immédiatement les tenseurs d'évaluation sur CPU.
        """
        self.model.eval()
        total_val_loss = 0.0

        all_image_embeds = []
        all_text_embeds = []
        all_shared_img = []
        all_shared_txt = []

        if not self.dataloader:
            return 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in self.dataloader:
                image = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)

                outputs = self.model(image, input_ids, attention_mask, audio, video)
                loss, _ = self.criterion(outputs, phase=phase)
                total_val_loss += loss.item()

                # Stockage CPU détaché pour éviter l'accumulation mémoire
                all_image_embeds.append(outputs['image_embed'].detach().cpu())
                all_text_embeds.append(outputs['text_embed'].detach().cpu())
                all_shared_img.append(outputs['shared_image'].detach().cpu())
                all_shared_txt.append(outputs['shared_text'].detach().cpu())

        num_batches = max(len(self.dataloader), 1)
        avg_val_loss = total_val_loss / num_batches
        self.val_losses.append(avg_val_loss)

        # Calcul des métriques globales
        image_embeds = torch.cat(all_image_embeds, dim=0)
        text_embeds = torch.cat(all_text_embeds, dim=0)

        r1 = recall_at_k(image_embeds, text_embeds, k=1)
        r5 = recall_at_k(image_embeds, text_embeds, k=5)
        map_score = mean_average_precision(image_embeds, text_embeds)

        shared_img_sample = torch.cat(all_shared_img[:2], dim=0) if all_shared_img else torch.empty(0)
        shared_txt_sample = torch.cat(all_shared_txt[:2], dim=0) if all_shared_txt else torch.empty(0)
        isometry = topological_isometry_score(shared_img_sample, shared_txt_sample) if shared_img_sample.numel() > 0 else 0.0

        self.metrics['Recall@1'].append(r1)
        self.metrics['Recall@5'].append(r5)
        self.metrics['MAP'].append(map_score)
        self.metrics['Isometry'].append(isometry)

        return avg_val_loss, r1, map_score