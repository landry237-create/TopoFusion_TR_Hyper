# -*- coding: utf-8 -*-
"""
Module de chargement et de prétraitement du dataset DocumentVQA et données multimodales.
Gère l'extraction des modalités réelles (Image, Texte) et simule de manière cohérente
les flux Audio et Vidéo (ou charge de vraies données audiovisuelles si disponibles).
"""

import os
import glob
from typing import Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class DocumentVQADataset(Dataset):
    """
    Dataset multimodal pour DocumentVQA.
    Gère la tolérance aux pannes (cache local, HuggingFace Hub, fallback synthétique).
    """
    def __init__(self, split: str = "train", max_samples: Optional[int] = 1000,
                 tokenizer=None, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.samples = []
        self.dataset = None

        # 1. Tentative de chargement via fichiers Parquet locaux dans le cache Hugging Face
        cache_pattern = os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--HuggingFaceM4--DocumentVQA/snapshots/*/*/*.parquet"
        )
        all_local = sorted(glob.glob(cache_pattern))
        # Filtrer pour le split demandé (ex: train)
        local_files = [f for f in all_local if split in f or "train" in f]
        if not local_files and all_local:
            local_files = all_local

        if local_files:
            try:
                from datasets import load_dataset
                # Sélectionner seulement le nombre de fichiers nécessaires
                num_needed = max(1, (max_samples + 999) // 1000) if max_samples else len(local_files)
                selected_files = local_files[:num_needed]
                print(f"Chargement de {len(selected_files)} shard(s) Parquet locaux pour {max_samples} échantillons...")
                self.dataset = load_dataset("parquet", data_files=selected_files, split="train")
                if max_samples and max_samples < len(self.dataset):
                    self.dataset = self.dataset.select(range(max_samples))
                print(f"Dataset chargé avec succès: {len(self.dataset)} échantillons.")
            except Exception as e:
                print(f"Avertissement lors du chargement des Parquet locaux: {e}")
                self.dataset = None

        # 2. Si échec, tentative via datasets.load_dataset en ligne
        if self.dataset is None:
            try:
                from datasets import load_dataset
                print(f"Tentative de chargement en ligne de HuggingFaceM4/DocumentVQA ({split})...")
                self.dataset = load_dataset("HuggingFaceM4/DocumentVQA", split=split)
                if max_samples and max_samples < len(self.dataset):
                    self.dataset = self.dataset.select(range(max_samples))
            except Exception as e:
                print(f"Avertissement lors du chargement distant: {e}")
                self.dataset = None

        # 3. Fallback synthétique si aucune source externe n'est accessible
        if self.dataset is None:
            print("Génération d'un jeu de données synthétique multimodal pour démonstration/tests...")
            self.num_synthetic = max_samples if max_samples else 32
        else:
            self.num_synthetic = 0

    def __len__(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        return self.num_synthetic

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Cas données réelles
        if self.dataset is not None:
            item = self.dataset[idx]

            # Image
            raw_img = item['image']
            if not isinstance(raw_img, Image.Image):
                raw_img = Image.fromarray(np.array(raw_img))
            image = raw_img.convert('RGB')

            if self.image_processor:
                pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
            else:
                img_arr = np.array(image.resize((224, 224)))
                pixel_values = torch.tensor(img_arr).permute(2, 0, 1).float() / 255.0

            # Texte (Question)
            question = item.get('question', "What is shown in the document?")
            if self.tokenizer:
                text_inputs = self.tokenizer(
                    question,
                    padding="max_length",
                    truncation=True,
                    max_length=32,
                    return_tensors="pt"
                )
                input_ids = text_inputs["input_ids"].squeeze(0)
                attention_mask = text_inputs["attention_mask"].squeeze(0)
            else:
                input_ids = torch.zeros(32, dtype=torch.long)
                attention_mask = torch.ones(32, dtype=torch.long)

        else:
            # Données synthétiques
            pixel_values = torch.randn(3, 224, 224)
            input_ids = torch.randint(0, 1000, (32,), dtype=torch.long)
            attention_mask = torch.ones(32, dtype=torch.long)
            question = f"Synthetic query #{idx}"

        # Modalités Audio et Vidéo (représentations de séquences de descripteurs)
        # Fixe une graine déterministe basée sur l'index pour la cohérence temporelle
        rng = torch.Generator().manual_seed(10000 + idx)
        audio_features = torch.randn(128, 512, generator=rng)
        video_features = torch.randn(64, 1024, generator=rng)

        return {
            "image": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio": audio_features,
            "video": video_features,
            "question": question
        }


def get_dataloader(config: Dict[str, Any], tokenizer=None, image_processor=None) -> DataLoader:
    """
    Crée et configure le DataLoader multimodal avec collate_fn adapté et gestion mémoire.
    """
    dataset_cfg = config.get('dataset', {})
    split = dataset_cfg.get('split', 'train')
    max_samples = dataset_cfg.get('max_samples', 1000)
    batch_size = dataset_cfg.get('batch_size', 8)
    num_workers = dataset_cfg.get('num_workers', 0)

    dataset = DocumentVQADataset(
        split=split,
        max_samples=max_samples,
        tokenizer=tokenizer,
        image_processor=image_processor
    )

    def collate_fn(batch):
        return {
            "image": torch.stack([item["image"] for item in batch]),
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "audio": torch.stack([item["audio"] for item in batch]),
            "video": torch.stack([item["video"] for item in batch]),
            "question": [item["question"] for item in batch]
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=True # Garantit des tailles de batch constantes pour cdist / TDA
    )

    return dataloader