# -*- coding: utf-8 -*-
"""
Encodeurs spécialisés pour chaque modalité (Bloc 1 du schéma architectural).
- Image: ViT / ResNet / ConvNeXt -> Tokens locaux I (N_I x d)
- Texte: Transformer (BERT, RoBERTa, ...) -> Tokens locaux T (N_T x d)
- Audio: Conformer / Audio Transformer (AST) -> Tokens locaux A (N_A x d)
- Vidéo: Vision Transformer spatio-temporel / VideoMAE -> Tokens locaux V (N_V x d)
- Projecteurs de modalité P_m: projection vers l'espace latent commun de dimension d.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, BertModel, ASTModel, VideoMAEModel


class ModalityProjector(nn.Module):
    """
    Projecteur de modalité P_m (Bloc 1).
    Aligne la dimension de sortie de chaque encodeur spécifique vers la dimension latente commune d.
    """
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tenseur de tokens locaux (batch_size, seq_len, input_dim)
        Returns:
            Tenseur projeté (batch_size, seq_len, latent_dim)
        """
        return self.net(x)


class ImageEncoder(nn.Module):
    """Encodeur d'images basé sur Vision Transformer (ViT)."""
    def __init__(self, model_name: str = "google/vit-base-patch16-224", freeze: bool = True):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
        self.output_dim = self.vit.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch_size, 3, H, W)
        Returns:
            Tokens locaux I: (batch_size, num_patches, hidden_dim)
        """
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.last_hidden_state


class TextEncoder(nn.Module):
    """Encodeur de texte basé sur BERT / Transformer."""
    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.output_dim = self.bert.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            Tokens locaux T: (batch_size, seq_len, hidden_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class AudioEncoder(nn.Module):
    """
    Encodeur audio basé sur Audio Spectrogram Transformer (AST) ou encodeur temporel léger.
    Gère à la fois les vrais spectrogrammes et les descripteurs de séquences audio.
    """
    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", freeze: bool = True):
        super().__init__()
        self.ast = ASTModel.from_pretrained(model_name)
        if freeze:
            for param in self.ast.parameters():
                param.requires_grad = False
        self.output_dim = self.ast.config.hidden_size
        # Adaptateur léger pour séquences de descripteurs audio
        self.feature_adapter = nn.Sequential(
            nn.Linear(512, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU()
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Soit spectrogramme (batch_size, 128, time_frames),
                   soit descripteurs temporels audio (batch_size, time_steps, feature_dim).
        Returns:
            Tokens locaux A: (batch_size, seq_len, output_dim)
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Si l'entrée est une séquence de caractéristiques (ex: (B, 128, 512))
        if audio.dim() == 3 and audio.shape[-1] == 512:
            return self.feature_adapter(audio)

        # Format spectrogramme attendu par AST (B, 128, 1024)
        if audio.shape[1] != 128:
            if audio.shape[-1] == 128:
                audio = audio.transpose(1, 2)
            else:
                audio = F.interpolate(
                    audio.unsqueeze(1),
                    size=(128, 1024),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

        if audio.shape[-1] < 1024:
            audio = F.pad(audio, (0, 1024 - audio.shape[-1]))
        elif audio.shape[-1] > 1024:
            audio = audio[..., :1024]

        outputs = self.ast(audio)
        return outputs.last_hidden_state


class SpatioTemporalVideoTransformer(nn.Module):
    """
    Transformer spatio-temporel léger pour l'encodage de séquences vidéo de tokens.
    Préserve la causalité et la structure spatio-temporelle avec encodage positionnel sinusoïdal.
    """
    def __init__(self, input_dim: int = 1024, output_dim: int = 768, num_layers: int = 2, nhead: int = 8):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=output_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.proj_in(x)
        # Encodage positionnel temporel
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        div_term = torch.exp(torch.arange(0, h.size(-1), 2, device=x.device).float() * (-math.log(10000.0) / h.size(-1)))
        pe = torch.zeros(b, t, h.size(-1), device=x.device)
        pe[:, :, 0::2] = torch.sin(positions.unsqueeze(-1).float() * div_term)
        pe[:, :, 1::2] = torch.cos(positions.unsqueeze(-1).float() * div_term)
        h = self.norm(self.transformer(h + pe))
        return h


class VideoEncoder(nn.Module):
    """
    Encodeur vidéo spatio-temporel (Bloc 1).
    Supporte:
    1. Clips vidéo bruts (batch_size, frames, channels, H, W) via VideoMAEModel
    2. Séquences vidéo prétraitées / descripteurs (batch_size, frames, feature_dim) via SpatioTemporalVideoTransformer
    """
    def __init__(self, model_name: str = "MCG-NJU/videomae-base", freeze: bool = True):
        super().__init__()
        self.output_dim = 768
        self.videomae = None
        self.model_name = model_name
        self.freeze = freeze
        # Encodeur spatio-temporel pour séquences vidéo
        self.temporal_encoder = SpatioTemporalVideoTransformer(
            input_dim=1024,
            output_dim=self.output_dim,
            num_layers=2,
            nhead=8
        )

    def _get_videomae(self):
        if self.videomae is None:
            self.videomae = VideoMAEModel.from_pretrained(self.model_name)
            if self.freeze:
                for param in self.videomae.parameters():
                    param.requires_grad = False
        return self.videomae

    def forward(self, video_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_input: Soit tenseur 5D de frames (batch_size, 16, 3, 224, 224),
                         soit tenseur 3D de tokens/features vidéo (batch_size, seq_len, feature_dim).
        Returns:
            Tokens locaux V: (batch_size, seq_len, output_dim)
        """
        if video_input.dim() == 2:
            video_input = video_input.unsqueeze(0)

        # Cas 1: Descripteurs spatio-temporels ou features (ex: (B, 64, 1024))
        if video_input.dim() == 3:
            return self.temporal_encoder(video_input)

        # Cas 2: Tenseur vidéo complet (B, T, C, H, W)
        if video_input.dim() == 5:
            videomae = self._get_videomae()
            outputs = videomae(pixel_values=video_input)
            return outputs.last_hidden_state

        raise ValueError(f"Format vidéo inattendu: {video_input.shape}. Attendu: 3D (features) ou 5D (frames).")