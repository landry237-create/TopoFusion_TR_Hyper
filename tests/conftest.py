# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Montages de test. Ces tenseurs servent uniquement aux tests et n'alimentent aucun résultat."""

from pathlib import Path  # Manipule les chemins indépendamment du système.

import numpy as np  # Crée les tableaux de test.
import pytest  # Fournit les montages de test.
import torch  # Limite le nombre de fils pendant les tests.

from data.feature_cache import FeatureSplit  # Importe l'accès aux caractéristiques.
from utils.config import load_config  # Charge la configuration.

RACINE = Path(__file__).resolve().parents[1]  # Calcule la racine du projet.


@pytest.fixture(autouse=True)  # Applique le montage à tous les tests.
def _limiter_fils():  # Limite le nombre de fils.
    """Limite les fils de calcul, pour que les tests restent rapides même si une expérience tourne."""
    torch.set_num_threads(2)  # Fixe deux fils de calcul.


@pytest.fixture()  # Déclare un montage.
def cfg():  # Fournit une configuration réduite.
    """Retourne la configuration de référence, réduite pour des tests rapides."""
    return load_config(str(RACINE / "config.yaml"), {  # Charge la configuration avec des remplacements.
        "model": {"latent_dim": 32, "num_latent_tokens": 10, "num_shared_tokens": 6, "num_heads": 2, "graph_hidden_dim": 8},  # Réduit le modèle.
        "topology": {"filtration_dim": 8, "landscape_steps": 8, "loss_points": 16},  # Réduit le moteur topologique.
        "training": {"epochs": 3, "batch_size": 8},  # Réduit l'entraînement.
        "evaluation": {"batch_size": 8, "topology_points": 16, "token_stability_samples": 4},  # Réduit l'évaluation.
    })  # Termine les remplacements.


def fabriquer_split(dossier: Path, prefixe: str, nb_images: int, graine: int) -> FeatureSplit:  # Fabrique un petit sous-ensemble.
    """Écrit un petit cache de test : deux légendes par image, avec un signal commun image-texte."""
    generateur = np.random.default_rng(graine)  # Crée le générateur de test.
    signal = generateur.normal(size=(nb_images, 16))  # Tire le signal commun à chaque paire.
    images = np.repeat(signal[:, None, :], 5, axis=1) + 0.1 * generateur.normal(size=(nb_images, 5, 16))  # Construit les tokens d'images.
    legendes = np.repeat(signal, 2, axis=0)  # Répète le signal pour les deux légendes.
    textes = np.repeat(legendes[:, None, :], 6, axis=1) + 0.1 * generateur.normal(size=(nb_images * 2, 6, 16))  # Construit les tokens de textes.
    masques = np.ones((nb_images * 2, 6), dtype=bool)  # Construit les masques.
    masques[:, 4:] = False  # Simule du padding sur les deux dernières positions.
    np.save(dossier / f"{prefixe}.image.npy", images.astype(np.float16))  # Écrit les tokens d'images.
    np.save(dossier / f"{prefixe}.text.npy", textes.astype(np.float16))  # Écrit les tokens de textes.
    np.save(dossier / f"{prefixe}.text_mask.npy", masques)  # Écrit les masques.
    entrees = [{"file": f"f{i}", "image_id": i, "captions": [f"{prefixe} image {i} a", f"{prefixe} image {i} b"]} for i in range(nb_images)]  # Construit le manifeste de test.
    return FeatureSplit(dossier, entrees, prefixe)  # Retourne le sous-ensemble.


@pytest.fixture()  # Déclare un montage.
def splits(tmp_path):  # Fournit de petits sous-ensembles.
    """Retourne de petits sous-ensembles d'entraînement et de validation."""
    return fabriquer_split(tmp_path, "train", 32, 0), fabriquer_split(tmp_path, "val", 16, 1)  # Retourne les deux sous-ensembles.
