# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Compare des réglages sur la validation uniquement : ce script n'ouvre jamais le jeu de test."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import argparse  # Lit les arguments de la ligne de commande.
import json  # Lit les réglages et écrit les résultats.
import shutil  # Supprime les points de contrôle temporaires.
import sys  # Permet d'ajouter la racine du projet au chemin d'importation.
import time  # Mesure les durées.
from pathlib import Path  # Manipule les chemins indépendamment du système.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Rend les paquets du projet importables.

import torch  # Permet de régler le nombre de fils de calcul.

from data.dataset_loader import prepare_manifest  # Relit le manifeste réel.
from data.feature_cache import FeatureExtractor, FeatureSplit  # Donne accès aux caractéristiques.
from evaluation.embedding import evaluate_retrieval  # Importe l'évaluation.
from models.multimodal_framework import get_variant  # Importe les variantes.
from training.trainer import Trainer  # Importe la boucle d'entraînement.
from utils.config import load_config  # Charge la configuration.


def main() -> None:  # Définit le point d'entrée du script.
    """Entraîne chaque réglage demandé et enregistre son meilleur score de validation."""
    analyseur = argparse.ArgumentParser(description="Compare des réglages sur la validation uniquement.")  # Crée l'analyseur d'arguments.
    analyseur.add_argument("--config", default="config.yaml", help="Chemin du fichier de configuration.")  # Déclare le fichier de configuration.
    analyseur.add_argument("--variante", required=True, help="Variante à régler.")  # Déclare la variante.
    analyseur.add_argument("--reglages", required=True, help="Dictionnaire JSON : nom du réglage -> remplacements de configuration.")  # Déclare les réglages.
    analyseur.add_argument("--graine", type=int, default=42, help="Graine utilisée.")  # Déclare la graine.
    analyseur.add_argument("--threads", type=int, default=2, help="Nombre de fils de calcul.")  # Déclare le nombre de fils.
    analyseur.add_argument("--sortie", default="outputs/reglages", help="Dossier des résultats.")  # Déclare le dossier des résultats.
    arguments = analyseur.parse_args()  # Lit les arguments.
    torch.set_num_threads(arguments.threads)  # Applique le nombre de fils.
    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choisit le device.
    sortie = Path(arguments.sortie)  # Convertit le dossier des résultats.
    sortie.mkdir(parents=True, exist_ok=True)  # Crée le dossier des résultats.
    for nom, remplacements in json.loads(arguments.reglages).items():  # Parcourt les réglages demandés.
        fichier = sortie / f"{arguments.variante}__{nom}__g{arguments.graine}.json"  # Calcule le fichier du résultat.
        if fichier.is_file():  # Vérifie si ce réglage a déjà été évalué.
            print(f"Déjà évalué : {fichier}", flush=True)  # Annonce la réutilisation.
            continue  # Passe au réglage suivant.
        cfg = load_config(arguments.config, remplacements)  # Charge la configuration remplacée.
        manifeste = prepare_manifest(cfg)  # Relit le manifeste.
        racine = FeatureExtractor(cfg, manifeste).racine  # Calcule le dossier du cache.
        train = FeatureSplit(racine, manifeste["splits"]["train"], "train")  # Ouvre l'entraînement.
        val = FeatureSplit(racine, manifeste["splits"]["val"], "val")  # Ouvre la validation.
        temporaire = sortie / "tmp" / f"{arguments.variante}__{nom}"  # Calcule le dossier temporaire.
        debut = time.time()  # Démarre le chronomètre.
        entraineur = Trainer(cfg, get_variant(arguments.variante), train, val, temporaire, arguments.graine, appareil)  # Prépare l'entraînement.
        resume = entraineur.fit()  # Entraîne le modèle.
        scores = evaluate_retrieval(entraineur.modele, val, int(cfg["evaluation"]["batch_size"]), appareil, cfg["evaluation"]["recall_ks"])  # Évalue la validation.
        fichier.write_text(json.dumps({"variante": arguments.variante, "reglage": nom, "remplacements": remplacements, "graine": arguments.graine, "duree_s": time.time() - debut, "meilleure_epoque": resume["meilleure_epoque"], "val": scores, "premiere_epoque_val": entraineur.historique["val"][0]["mean_recall_at_5"]}, ensure_ascii=False, indent=1), encoding="utf-8")  # Écrit le résultat.
        shutil.rmtree(temporaire, ignore_errors=True)  # Supprime les points de contrôle temporaires.
        print(f"REGLAGE {arguments.variante} {nom} val_R@1={scores['mean_recall_at_1']:.4f} val_R@5={scores['mean_recall_at_5']:.4f} val_mAP={scores['mean_map']:.4f} durée={time.time() - debut:.0f}s époque={resume['meilleure_epoque']}", flush=True)  # Journalise le résultat.


if __name__ == "__main__":  # Vérifie que le fichier est exécuté directement.
    main()  # Lance le script.
