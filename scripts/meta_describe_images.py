# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Demande à Muse Spark de décrire de vraies images, si l'utilisateur fournit une clé et confirme l'envoi."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import argparse  # Lit les arguments de la ligne de commande.
import json  # Écrit les descriptions obtenues.
import sys  # Fournit le code de sortie et le chemin d'importation.
from pathlib import Path  # Manipule les chemins indépendamment du système.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Rend les paquets du projet importables.

from data.dataset_loader import ImageStore, prepare_manifest  # Donne accès aux images réelles.
from integrations.meta_model_api import MetaApiUnavailable, MetaModelApiClient  # Importe l'adaptateur.
from utils.config import load_config  # Charge la configuration.


def main() -> int:  # Définit le point d'entrée du script.
    """Envoie quelques images de test à l'API, uniquement après confirmation explicite."""
    analyseur = argparse.ArgumentParser(description="Décrit de vraies images avec la Meta Model API.")  # Crée l'analyseur d'arguments.
    analyseur.add_argument("--config", default="config.yaml", help="Chemin du fichier de configuration.")  # Déclare le fichier de configuration.
    analyseur.add_argument("--nombre", type=int, default=8, help="Nombre d'images à décrire.")  # Déclare le nombre d'images.
    analyseur.add_argument("--confirmer-envoi", action="store_true", help="Confirme l'envoi de ces images à un service externe, potentiellement facturé.")  # Déclare la confirmation.
    analyseur.add_argument("--sortie", default="outputs/meta/descriptions.jsonl", help="Fichier de sortie.")  # Déclare le fichier de sortie.
    arguments = analyseur.parse_args()  # Lit les arguments.
    if not arguments.confirmer_envoi:  # Vérifie la confirmation.
        print("Aucun envoi effectué : ajoutez --confirmer-envoi pour envoyer des images à la Meta Model API.")  # Explique le refus.
        return 2  # Retourne un code d'erreur.
    cfg = load_config(arguments.config)  # Charge la configuration.
    try:  # Protège la création du client.
        client = MetaModelApiClient(cfg["meta_model_api"], Path(cfg["dataset"]["cache_dir"]) / "meta_cache")  # Crée le client.
    except MetaApiUnavailable as exc:  # Intercepte l'absence de clé.
        print(exc)  # Affiche le message explicite.
        return 2  # Retourne un code d'erreur.
    manifeste = prepare_manifest(cfg)  # Relit le manifeste réel.
    store = ImageStore(cfg)  # Prépare l'accès aux images réelles.
    chemin = Path(arguments.sortie)  # Convertit le fichier de sortie.
    chemin.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier de sortie.
    with chemin.open("w", encoding="utf-8") as flux:  # Ouvre le fichier de sortie.
        for entree in manifeste["splits"]["test"][: arguments.nombre]:  # Parcourt les images demandées.
            description = client.describe_image(store.open(entree["file"]))  # Demande la description.
            flux.write(json.dumps({"fichier": entree["file"], "description": description.texte, "modele": description.modele, "provenance": description.provenance, "legendes_reelles": entree["captions"]}, ensure_ascii=False) + "\n")  # Écrit la description avec sa provenance.
    print(f"Descriptions écrites dans {chemin}.")  # Annonce la fin.
    return 0  # Retourne un code de succès.


if __name__ == "__main__":  # Vérifie que le fichier est exécuté directement.
    sys.exit(main())  # Termine avec le code calculé.
