# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Point d'entrée de TopoFusion-TR Hyper : chaque étape du protocole est une sous-commande."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import argparse  # Lit les arguments de la ligne de commande.
import subprocess  # Lance les scripts du protocole.
import sys  # Fournit l'interpréteur courant et le code de sortie.
from pathlib import Path  # Manipule les chemins indépendamment du système.

ETAPES = {  # Associe chaque sous-commande à son script.
    "preparer": "scripts/prepare_data.py",  # Prépare les données réelles et les caractéristiques.
    "experiences": "scripts/run_experiments.py",  # Entraîne et évalue les variantes.
    "resultats": "scripts/aggregate_results.py",  # Agrège les résultats sur les graines.
    "figures": "scripts/make_figures.py",  # Génère les figures scientifiques.
    "commentaires": "scripts/verifier_commentaires.py",  # Vérifie les commentaires du code.
}  # Termine le registre des étapes.


def main() -> int:  # Définit le point d'entrée.
    """Transmet les arguments au script de la sous-commande demandée."""
    analyseur = argparse.ArgumentParser(description="TopoFusion-TR Hyper : protocole expérimental complet.")  # Crée l'analyseur d'arguments.
    analyseur.add_argument("etape", choices=sorted(ETAPES), help="Étape du protocole à exécuter.")  # Déclare la sous-commande.
    arguments, reste = analyseur.parse_known_args()  # Sépare la sous-commande de ses arguments.
    script = Path(__file__).resolve().parent / ETAPES[arguments.etape]  # Calcule le chemin du script.
    return subprocess.call([sys.executable, str(script), *reste])  # Lance le script et retourne son code de sortie.


if __name__ == "__main__":  # Vérifie que le fichier est exécuté directement.
    sys.exit(main())  # Termine avec le code du script lancé.
