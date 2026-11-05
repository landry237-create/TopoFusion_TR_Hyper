# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Gestion centralisée des graines aléatoires et des états des générateurs."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import os  # Donne accès aux variables d'environnement.
import random  # Fournit le générateur standard de Python.
from typing import Any, Dict  # Importe les types utilisés dans les signatures.

import numpy as np  # Fournit le générateur de NumPy.
import torch  # Fournit les générateurs de PyTorch.


def set_global_seed(graine: int) -> None:  # Fixe toutes les graines du processus.
    """Fixe les graines de Python, NumPy et PyTorch pour un entraînement reproductible."""
    os.environ["PYTHONHASHSEED"] = str(graine)  # Stabilise le hachage pour les sous-processus.
    random.seed(graine)  # Initialise le générateur standard.
    np.random.seed(graine)  # Initialise le générateur NumPy.
    torch.manual_seed(graine)  # Initialise le générateur PyTorch sur CPU.
    if torch.cuda.is_available():  # Vérifie la présence d'un GPU.
        torch.cuda.manual_seed_all(graine)  # Initialise les générateurs de tous les GPU.
        torch.backends.cudnn.deterministic = True  # Demande des convolutions déterministes.
        torch.backends.cudnn.benchmark = False  # Désactive la sélection variable des algorithmes.


def capture_rng_state() -> Dict[str, Any]:  # Capture l'état de tous les générateurs.
    """Retourne l'état courant des générateurs pour reprendre un entraînement à l'identique."""
    etat: Dict[str, Any] = {  # Regroupe les états dans un dictionnaire.
        "python": random.getstate(),  # État du générateur standard.
        "numpy": np.random.get_state(),  # État du générateur NumPy.
        "torch": torch.get_rng_state(),  # État du générateur PyTorch sur CPU.
    }  # Termine le dictionnaire des états.
    if torch.cuda.is_available():  # Vérifie la présence d'un GPU.
        etat["cuda"] = torch.cuda.get_rng_state_all()  # Ajoute l'état de chaque GPU.
    return etat  # Retourne les états capturés.


def restore_rng_state(etat: Dict[str, Any]) -> None:  # Restaure l'état de tous les générateurs.
    """Restaure les états capturés par capture_rng_state."""
    random.setstate(etat["python"])  # Restaure le générateur standard.
    np.random.set_state(etat["numpy"])  # Restaure le générateur NumPy.
    torch.set_rng_state(etat["torch"])  # Restaure le générateur PyTorch sur CPU.
    if "cuda" in etat and torch.cuda.is_available():  # Vérifie que l'état GPU est utilisable.
        torch.cuda.set_rng_state_all(etat["cuda"])  # Restaure les générateurs GPU.
