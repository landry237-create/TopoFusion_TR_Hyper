# -*- coding: utf-8 -*-
"""
Point d'entrée principal de l'architecture Topo-Multimodal.
Orchestre:
1. Le chargement des données et des processeurs de vision/texte
2. L'instanciation du modèle TopoMultimodalFramework (conforme à 'archi.png')
3. L'entraînement modulaire en 3 phases avec supervision topologique
4. La génération des visualisations analytiques (diagrammes de persistance, attention, espace latent)
"""

import os
import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoImageProcessor

from data.dataset_loader import get_dataloader
from models.multimodal_framework import TopoMultimodalFramework
from training.trainer import Trainer
from utils.visualization import (
    visualize_dataset_samples,
    visualize_latent_space,
    visualize_persistence_diagram,
    visualize_topological_attention,
    plot_training_curves
)


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement de l'architecture Topo-Multimodal")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin du fichier de configuration")
    parser.add_argument("--quick_demo", action="store_true", help="Mode démo rapide avec 1 époque par phase")
    parser.add_argument("--max_samples", type=int, default=None, help="Nombre maximal d'échantillons du dataset")
    parser.add_argument("--batch_size", type=int, default=None, help="Taille des batchs")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Dossier de sauvegarde des graphiques")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Chargement de la configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.max_samples is not None:
        config['dataset']['max_samples'] = args.max_samples
    elif args.quick_demo:
        config['dataset']['max_samples'] = min(config['dataset'].get('max_samples', 32), 32)

    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size

    # 2. Configuration du device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Initialisation Topo-Multimodal (Device: {device}) ===")

    # 3. Chargement des processeurs
    model_cfg = config.get('models', {})
    text_model_name = model_cfg.get('encoder_text', 'bert-base-uncased')
    image_model_name = model_cfg.get('encoder_image', 'google/vit-base-patch16-224')

    print(f"Chargement du tokenizer texte ({text_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    print(f"Chargement de l'image processor ({image_model_name})...")
    image_processor = AutoImageProcessor.from_pretrained(image_model_name)

    # 4. Création du DataLoader
    dataloader = get_dataloader(config, tokenizer, image_processor)
    print(f"DataLoader initialisé: {len(dataloader)} batchs de taille {config['dataset']['batch_size']}.")

    # Visualisation des premiers échantillons
    sample_path = os.path.join(args.output_dir, "dataset_samples.png")
    print(f"Visualisation d'échantillons du dataset -> {sample_path}")
    visualize_dataset_samples(dataloader, num_samples=4, save_path=sample_path)

    # 5. Instanciation du modèle complet TopoMultimodalFramework
    print("Instanciation de l'architecture TopoMultimodalFramework...")
    model = TopoMultimodalFramework(config).to(device)

    # 6. Instanciation du Trainer
    trainer = Trainer(model, dataloader, config, device=str(device))

    # Gestion des époques (démo rapide ou configuration standard)
    e_phase1 = 1 if args.quick_demo else config['training'].get('epochs_phase1', 5)
    e_phase2 = 1 if args.quick_demo else config['training'].get('epochs_phase2', 10)
    e_phase3 = 1 if args.quick_demo else config['training'].get('epochs_phase3', 5)

    # -----------------------------------------------------------------
    # ENTRAÎNEMENT EN 3 PHASES
    # -----------------------------------------------------------------
    # PHASE I : Amorçage Sémantique et Géométrique (L_sem + L_geom)
    print("\n" + "=" * 60)
    print("DÉMARRAGE PHASE I : Amorçage Sémantique et Géométrique")
    print("=" * 60)
    for epoch in range(e_phase1):
        train_loss = trainer.train_epoch(epoch, phase=1)
        val_loss, r1, map_score = trainer.validate(phase=1)
        print(f"[Phase I - Epoch {epoch+1}/{e_phase1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R@1: {r1:.4f} | MAP: {map_score:.4f}")

    # PHASE II : Stabilisation Structurelle et Topologique (+ L_top)
    print("\n" + "=" * 60)
    print("DÉMARRAGE PHASE II : Stabilisation Structurelle et Topologique")
    print("=" * 60)
    for epoch in range(e_phase2):
        train_loss = trainer.train_epoch(epoch, phase=2)
        val_loss, r1, map_score = trainer.validate(phase=2)
        print(f"[Phase II - Epoch {epoch+1}/{e_phase2}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R@1: {r1:.4f} | MAP: {map_score:.4f}")

    # PHASE III : Robustesse et Continuité Temporelle (+ L_rob + L_temp)
    print("\n" + "=" * 60)
    print("DÉMARRAGE PHASE III : Robustesse Multimodale et Branche Temporelle")
    print("=" * 60)
    for epoch in range(e_phase3):
        train_loss = trainer.train_epoch(epoch, phase=3)
        val_loss, r1, map_score = trainer.validate(phase=3)
        print(f"[Phase III - Epoch {epoch+1}/{e_phase3}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | R@1: {r1:.4f} | MAP: {map_score:.4f}")

    # -----------------------------------------------------------------
    # GÉNÉRATION DES VISUALISATIONS ANALYTIQUES FINALES
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GÉNÉRATION DES RAPPORTS ET VISUALISATIONS ANALYTIQUES")
    print("=" * 60)

    # 1. Courbes d'entraînement
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(trainer.train_losses, trainer.val_losses, trainer.metrics, save_path=curves_path)

    # 2. Inférence sur un batch pour inspection topologique
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        outputs = model(
            batch['image'].to(device),
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['audio'].to(device),
            batch['video'].to(device)
        )

        # 3. Espace latent Shared vs Private
        latent_path = os.path.join(args.output_dir, "latent_space.png")
        visualize_latent_space(outputs['shared_image'], outputs['private_image'], save_path=latent_path)

        # 4. Diagramme de persistance H0/H1
        if outputs['diagrams_fused']:
            diag_path = os.path.join(args.output_dir, "persistence_diagram.png")
            visualize_persistence_diagram(outputs['diagrams_fused'][0], title="Diagramme de Persistance Fusionné ($H_0$ et $H_1$)", save_path=diag_path)

        # 5. Attention topologique
        attn_path = os.path.join(args.output_dir, "topological_attention.png")
        visualize_topological_attention(outputs['topological_attention_weights'], title="Attention Topologique ($A_{ij}$)", save_path=attn_path)

    print("\n Entraînement, évaluation et visualisations terminés avec succès!")


if __name__ == "__main__":
    main()