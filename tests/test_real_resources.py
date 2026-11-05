# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Tests d'intégration sur les vraies ressources : ignorés s'ils sont indisponibles, jamais simulés."""

import json  # Lit le manifeste réel.
from pathlib import Path  # Manipule les chemins indépendamment du système.

import numpy as np  # Effectue les calculs matriciels.
import pytest  # Fournit le cadre de test.
import torch  # Fournit les tenseurs.
from PIL import Image  # Ouvre les images réelles.

from data.dataset_loader import ImageStore, load_caption_table, manifest_path, verify_no_leakage  # Importe le chargeur réel.
from models.encoders import FrozenImageEncoder, FrozenTextEncoder, feature_origin  # Importe les encodeurs gelés.
from utils.config import load_config  # Charge la configuration.

RACINE = Path(__file__).resolve().parents[1]  # Calcule la racine du projet.


def _config_reelle():  # Charge la configuration réelle.
    """Retourne la configuration de référence, avec des chemins absolus."""
    cfg = load_config(str(RACINE / "config.yaml"))  # Charge la configuration.
    cfg["dataset"]["cache_dir"] = str(RACINE / cfg["dataset"]["cache_dir"])  # Rend le cache indépendant du dossier courant.
    return cfg  # Retourne la configuration.


def _manifeste_ou_ignorer(cfg):  # Lit le manifeste réel.
    """Retourne le manifeste réel, ou ignore le test s'il n'a pas encore été préparé."""
    chemin = manifest_path(cfg)  # Calcule le chemin du manifeste.
    if not chemin.is_file():  # Vérifie la présence du fichier.
        pytest.skip("Manifeste absent : exécuter d'abord scripts/prepare_data.py.")  # Ignore le test.
    return json.loads(chemin.read_text(encoding="utf-8"))  # Retourne le manifeste.


@pytest.mark.network  # Signale le besoin éventuel du réseau.
@pytest.mark.slow  # Signale un test long.
def test_encodeur_image_reel_gele_et_deterministe():  # Corrige le défaut H1.
    """L'encodeur d'image réel doit rester gelé, déterministe et en mode évaluation."""
    cfg = _config_reelle()  # Charge la configuration.
    try:  # Protège le chargement du modèle.
        encodeur = FrozenImageEncoder(cfg["encoders"]["image"], cfg["encoders"]["image_grid"])  # Charge l'encodeur réel.
    except OSError as exc:  # Intercepte un modèle indisponible.
        pytest.skip(f"Modèle indisponible : {exc}")  # Ignore le test.
    encodeur.train()  # Simule l'appel fait par model.train().
    assert encodeur.modele.training is False  # Vérifie que le dropout reste désactivé.
    assert all(not p.requires_grad for p in encodeur.parameters())  # Vérifie le gel des poids.
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))  # Crée une image de test.
    a = encodeur([image, image])  # Encode une première fois.
    b = encodeur([image, image])  # Encode une seconde fois.
    assert a.shape == (2, 1 + cfg["encoders"]["image_grid"] ** 2, encodeur.dim_sortie)  # Vérifie la forme.
    assert torch.equal(a, b)  # Vérifie le déterminisme.
    assert feature_origin(cfg["encoders"]["image"]).startswith("Poids ouverts Meta")  # Vérifie la provenance déclarée.


@pytest.mark.network  # Signale le besoin éventuel du réseau.
@pytest.mark.slow  # Signale un test long.
def test_encodeur_texte_reel_masque_le_padding():  # Vérifie le masque réel.
    """L'encodeur de texte réel doit produire un masque cohérent et rester déterministe."""
    cfg = _config_reelle()  # Charge la configuration.
    try:  # Protège le chargement du modèle.
        encodeur = FrozenTextEncoder(cfg["encoders"]["text"], cfg["encoders"]["max_text_tokens"])  # Charge l'encodeur réel.
    except OSError as exc:  # Intercepte un modèle indisponible.
        pytest.skip(f"Modèle indisponible : {exc}")  # Ignore le test.
    encodeur.train()  # Simule l'appel fait par model.train().
    assert encodeur.modele.training is False  # Vérifie que le dropout reste désactivé.
    tokens, masque = encodeur(["a dog", "a man riding a wave on top of a surfboard"])  # Encode deux légendes.
    assert tokens.shape == (2, cfg["encoders"]["max_text_tokens"], encodeur.dim_sortie)  # Vérifie la forme.
    assert masque[0].sum() < masque[1].sum()  # Vérifie que la légende courte possède plus de padding.
    assert torch.equal(tokens, encodeur(["a dog", "a man riding a wave on top of a surfboard"])[0])  # Vérifie le déterminisme.


@pytest.mark.network  # Signale le besoin éventuel du réseau.
@pytest.mark.slow  # Signale un test long.
def test_jeu_cible_reel_contient_les_paires_attendues():  # Vérifie le jeu réel.
    """Le chargeur réel doit retrouver les 82 783 images et récupérer la première ligne du CSV."""
    cfg = _config_reelle()  # Charge la configuration.
    try:  # Protège le chargement du jeu de données.
        table = load_caption_table(cfg)  # Charge le jeu réel.
    except Exception as exc:  # Intercepte l'indisponibilité du dépôt.
        pytest.skip(f"Jeu de données indisponible : {type(exc).__name__}")  # Ignore le test.
    assert table.report["images_distinctes"] == 82783  # Vérifie le nombre d'images.
    assert table.report["legendes_valides"] == 413253  # Vérifie le nombre de légendes valides.
    assert table.report["en_tete_recupere"] is True  # Vérifie la récupération de la première ligne.
    assert "COCO_train2014_000000000009.jpg" in table.captions_by_image  # Vérifie l'échantillon récupéré.
    assert "a white vase filled with different colored flowers." in table.captions_by_image["COCO_train2014_000000000030.jpg"]  # Vérifie une correspondance connue.


def test_manifeste_reel_sans_fuite_et_images_lisibles():  # Vérifie le découpage réel.
    """Le manifeste réel ne doit contenir aucune fuite, et ses images doivent être lisibles."""
    cfg = _config_reelle()  # Charge la configuration.
    manifeste = _manifeste_ou_ignorer(cfg)  # Lit le manifeste.
    verify_no_leakage(manifeste)  # Vérifie l'absence de fuite.
    assert manifeste["meta"]["comptes"]["train"]["images"] == cfg["dataset"]["num_train_images"]  # Vérifie le quota d'entraînement.
    assert manifeste["meta"]["comptes"]["test"]["images"] == cfg["dataset"]["num_test_images"]  # Vérifie le quota de test.
    store = ImageStore(cfg)  # Prépare l'accès aux images.
    for entree in manifeste["splits"]["test"][:5]:  # Parcourt quelques images de test.
        assert store.open(entree["file"]).mode == "RGB"  # Vérifie que l'image est lisible.
        assert entree["captions"]  # Vérifie la présence de légendes réelles.


def test_cache_reel_coherent_avec_le_manifeste():  # Vérifie le cache réel.
    """Les caractéristiques en cache doivent correspondre au manifeste, et rester différentes selon la modalité."""
    from data.feature_cache import FeatureExtractor, FeatureSplit  # Importe l'accès au cache.

    cfg = _config_reelle()  # Charge la configuration.
    manifeste = _manifeste_ou_ignorer(cfg)  # Lit le manifeste.
    racine = FeatureExtractor(cfg, manifeste).racine  # Calcule le dossier du cache.
    if not (racine / "test.text_mask.npy").is_file():  # Vérifie la présence du cache de test.
        pytest.skip("Caractéristiques absentes : exécuter d'abord scripts/prepare_data.py.")  # Ignore le test.
    test = FeatureSplit(racine, manifeste["splits"]["test"], "test")  # Ouvre le test.
    assert test.nb_images == cfg["dataset"]["num_test_images"]  # Vérifie le nombre d'images.
    assert test.nb_legendes == manifeste["meta"]["comptes"]["test"]["legendes"]  # Vérifie le nombre de légendes.
    assert test.images.dtype == np.float16 and np.isfinite(np.asarray(test.images[:32], dtype=np.float32)).all()  # Vérifie les valeurs des images.
    assert np.asarray(test.images[:8], dtype=np.float32).std() > 0.0  # Vérifie que les caractéristiques ne sont pas constantes.
    assert test.masques_texte[:, 0].all()  # Vérifie que le premier token est toujours valide.
