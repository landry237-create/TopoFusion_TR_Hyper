# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Prépare les données réelles : manifeste, images COCO et caractéristiques gelées."""

import argparse  # Lit les arguments de la ligne de commande.
import sys  # Permet d'ajouter la racine du projet au chemin d'importation.
from pathlib import Path  # Manipule les chemins indépendamment du système.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Rend les paquets du projet importables.

import torch  # Permet de régler le nombre de fils de calcul.

from data.corruptions import list_scenarios  # Énumère les scénarios de corruption.
from data.dataset_loader import prepare_manifest  # Prépare le manifeste réel.
from data.feature_cache import FeatureExtractor  # Pilote l'extraction des caractéristiques.
from utils.config import load_config  # Charge la configuration validée.


def main() -> None:  # Définit le point d'entrée du script.
    """Exécute la préparation complète des données."""
    analyseur = argparse.ArgumentParser(description="Prépare les paires image-texte réelles et leurs caractéristiques.")  # Crée l'analyseur d'arguments.
    analyseur.add_argument("--config", default="config.yaml", help="Chemin du fichier de configuration.")  # Déclare le fichier de configuration.
    analyseur.add_argument("--threads", type=int, default=8, help="Nombre de fils de calcul PyTorch.")  # Déclare le nombre de fils.
    analyseur.add_argument("--sans-corruptions", action="store_true", help="Ignore les scénarios de corruption.")  # Déclare l'option d'exclusion.
    arguments = analyseur.parse_args()  # Lit les arguments reçus.
    torch.set_num_threads(arguments.threads)  # Applique le nombre de fils demandé.
    configuration = load_config(arguments.config)  # Charge la configuration.
    manifeste = prepare_manifest(configuration)  # Prépare le manifeste réel.
    extracteur = FeatureExtractor(configuration, manifeste)  # Prépare l'extracteur.
    extracteur.extract_clean()  # Extrait les caractéristiques propres.
    if not arguments.sans_corruptions:  # Vérifie si les corruptions sont demandées.
        extracteur.extract_corrupted(list_scenarios(configuration))  # Extrait les caractéristiques corrompues.
    print("PREPARATION_TERMINEE", flush=True)  # Signale la fin de la préparation.


if __name__ == "__main__":  # Vérifie que le fichier est exécuté directement.
    main()  # Lance le script.
