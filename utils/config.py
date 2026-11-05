# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Chargement et validation stricte de la configuration du projet."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import copy  # Fournit la copie profonde des dictionnaires de configuration.
import hashlib  # Fournit le hachage stable de la configuration.
import json  # Sérialise la configuration de façon canonique avant hachage.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Any, Dict, Iterable, Optional  # Importe les types utilisés dans les signatures.

import yaml  # Lit le fichier de configuration YAML.

CHEMINS_FLOTTANTS = [  # Liste les clés numériques que PyYAML peut lire comme des chaînes.
    ("training", "lr"),  # Taux d'apprentissage initial.
    ("training", "weight_decay"),  # Décroissance des poids d'AdamW.
    ("training", "warmup_ratio"),  # Part des étapes consacrée à l'échauffement.
    ("training", "min_lr_ratio"),  # Rapport final du taux d'apprentissage.
    ("training", "max_grad_norm"),  # Norme maximale du gradient.
    ("training", "feature_noise_std"),  # Écart-type du bruit sur les caractéristiques.
    ("training", "feature_mask_prob"),  # Probabilité de masquage d'un token.
    ("training", "modality_dropout_prob"),  # Probabilité d'abandon d'une modalité.
    ("model", "dropout"),  # Taux de dropout du modèle.
    ("model", "initial_temperature"),  # Température initiale d'InfoNCE.
    ("model", "initial_structural_gate"),  # Valeur initiale de la porte structurelle.
    ("topology", "soft_betti_temperature"),  # Température des courbes de Betti douces.
]  # Termine la liste des chemins numériques.

SECTIONS_REQUISES = [  # Liste les sections qui doivent exister dans le fichier.
    "dataset",  # Section décrivant les données.
    "modalities",  # Section décrivant les modalités actives.
    "encoders",  # Section décrivant les encodeurs gelés.
    "model",  # Section décrivant l'architecture.
    "topology",  # Section décrivant le moteur topologique.
    "training",  # Section décrivant l'entraînement.
    "loss_weights",  # Section décrivant les poids des pertes.
    "evaluation",  # Section décrivant le protocole d'évaluation.
    "experiments",  # Section décrivant le plan expérimental.
    "corruptions",  # Section décrivant les perturbations.
    "meta_model_api",  # Section décrivant l'adaptateur Meta.
    "outputs",  # Section décrivant les sorties.
]  # Termine la liste des sections requises.


class ConfigError(ValueError):  # Définit l'erreur levée pour une configuration invalide.
    """Signale une configuration invalide avec un message exploitable."""


def _lire(cfg: Dict[str, Any], chemin: Iterable[str]) -> Any:  # Lit une valeur à partir d'un chemin de clés.
    """Retourne la valeur située au chemin indiqué ou lève une erreur explicite."""
    courant: Any = cfg  # Démarre le parcours à la racine.
    for cle in chemin:  # Parcourt chaque niveau du chemin.
        if not isinstance(courant, dict) or cle not in courant:  # Vérifie que le niveau existe.
            raise ConfigError(f"Clé de configuration manquante : {'.'.join(chemin)}")  # Nomme la clé absente.
        courant = courant[cle]  # Descend d'un niveau.
    return courant  # Retourne la valeur trouvée.


def _exiger(condition: bool, message: str) -> None:  # Factorise la vérification d'une condition.
    """Lève une erreur de configuration si la condition est fausse."""
    if not condition:  # Teste la condition reçue.
        raise ConfigError(message)  # Remonte le message explicatif.


def _convertir_flottants(cfg: Dict[str, Any]) -> None:  # Convertit les nombres lus comme des chaînes.
    """Convertit en flottants les valeurs numériques écrites sous forme de chaînes."""
    for section, cle in CHEMINS_FLOTTANTS:  # Parcourt les clés numériques connues.
        valeur = _lire(cfg, (section, cle))  # Lit la valeur actuelle.
        try:  # Tente la conversion numérique.
            cfg[section][cle] = float(valeur)  # Remplace la valeur par un flottant.
        except (TypeError, ValueError) as exc:  # Intercepte une valeur non numérique.
            raise ConfigError(f"{section}.{cle} doit être numérique, reçu : {valeur!r}") from exc  # Explique le problème.


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:  # Valide toute la configuration.
    """Valide la configuration en place et la retourne."""
    for section in SECTIONS_REQUISES:  # Vérifie chaque section attendue.
        _exiger(section in cfg, f"Section de configuration manquante : {section}")  # Nomme la section absente.
    _convertir_flottants(cfg)  # Normalise les valeurs numériques.
    mod = cfg["modalities"]  # Récupère les modalités déclarées.
    _exiger(bool(mod.get("image")) and bool(mod.get("text")), "Le benchmark exige les modalités réelles image et texte.")  # Impose image et texte.
    _exiger(  # Refuse l'activation de modalités sans données réelles.
        not mod.get("audio") and not mod.get("video"),  # Vérifie que l'audio et la vidéo restent inactifs.
        "Audio et vidéo sont des extensions expérimentales : aucune donnée réelle n'est intégrée, "  # Explique la raison du refus.
        "et ce projet ne génère jamais de caractéristiques artificielles pour les simuler.",  # Rappelle la règle du projet.
    )  # Termine la vérification des modalités.
    m = cfg["model"]  # Récupère la section du modèle.
    _exiger(m["num_shared_tokens"] < m["num_latent_tokens"], "model.num_shared_tokens doit être < model.num_latent_tokens pour conserver un espace privé.")  # Impose K_c < K.
    _exiger(m["latent_dim"] % m["num_heads"] == 0, "model.latent_dim doit être divisible par model.num_heads.")  # Impose la divisibilité de l'attention.
    _exiger(0.0 < m["initial_temperature"] < 1.0, "model.initial_temperature doit appartenir à ]0, 1[.")  # Borne la température.
    _exiger(0.0 < m["initial_structural_gate"] < 1.0, "model.initial_structural_gate doit appartenir à ]0, 1[.")  # Borne la porte.
    t = cfg["topology"]  # Récupère la section topologique.
    _exiger(set(t["loss_homology_dims"]).issubset({0, 1}), "topology.loss_homology_dims ne peut contenir que 0 et 1.")  # Limite aux dimensions calculées.
    _exiger(t["loss_points"] >= 8, "topology.loss_points doit valoir au moins 8.")  # Impose un nuage exploitable.
    _exiger(t["soft_betti_temperature"] > 0.0, "topology.soft_betti_temperature doit être strictement positive.")  # Évite la division par zéro.
    tr = cfg["training"]  # Récupère la section d'entraînement.
    _exiger(tr["batch_size"] >= 2, "training.batch_size doit valoir au moins 2 pour InfoNCE.")  # Impose au moins un négatif.
    _exiger(tr["epochs"] >= 1, "training.epochs doit valoir au moins 1.")  # Impose au moins une époque.
    _exiger(abs(sum(tr["phase_fractions"]) - 1.0) < 1e-6 and len(tr["phase_fractions"]) == 3, "training.phase_fractions doit contenir 3 valeurs de somme 1.")  # Valide les phases.
    d = cfg["dataset"]  # Récupère la section des données.
    _exiger(abs(sum(d["split_fractions"]) - 1.0) < 1e-6 and len(d["split_fractions"]) == 3, "dataset.split_fractions doit contenir 3 valeurs de somme 1.")  # Valide le découpage.
    _exiger("{file}" in d["image_url_template"], "dataset.image_url_template doit contenir le motif {file}.")  # Valide le gabarit d'URL.
    for nom, valeur in cfg["loss_weights"].items():  # Parcourt les poids des pertes.
        cfg["loss_weights"][nom] = float(valeur)  # Convertit le poids en flottant.
        _exiger(cfg["loss_weights"][nom] >= 0.0, f"loss_weights.{nom} doit être positif ou nul.")  # Refuse un poids négatif.
    return cfg  # Retourne la configuration validée.


def deep_update(base: Dict[str, Any], nouveautes: Optional[Dict[str, Any]]) -> Dict[str, Any]:  # Fusionne deux dictionnaires.
    """Retourne une copie de base mise à jour récursivement."""
    resultat = copy.deepcopy(base)  # Copie la base pour ne pas la modifier.
    for cle, valeur in (nouveautes or {}).items():  # Parcourt les valeurs à fusionner.
        if isinstance(valeur, dict) and isinstance(resultat.get(cle), dict):  # Détecte un sous-dictionnaire.
            resultat[cle] = deep_update(resultat[cle], valeur)  # Fusionne récursivement.
        else:  # Traite le cas d'une valeur simple.
            resultat[cle] = copy.deepcopy(valeur)  # Remplace la valeur.
    return resultat  # Retourne le dictionnaire fusionné.


def load_config(chemin: str = "config.yaml", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # Charge la configuration.
    """Charge, fusionne et valide un fichier de configuration YAML."""
    fichier = Path(chemin)  # Convertit le chemin reçu.
    _exiger(fichier.is_file(), f"Fichier de configuration introuvable : {fichier.resolve()}")  # Vérifie l'existence du fichier.
    with fichier.open("r", encoding="utf-8") as flux:  # Ouvre le fichier en UTF-8.
        cfg = yaml.safe_load(flux)  # Analyse le YAML sans exécuter de code.
    _exiger(isinstance(cfg, dict), f"La configuration {fichier} doit être un dictionnaire YAML.")  # Vérifie la structure racine.
    return validate_config(deep_update(cfg, overrides))  # Fusionne puis valide.


def config_hash(cfg: Dict[str, Any]) -> str:  # Calcule une empreinte de configuration.
    """Retourne une empreinte courte et stable de la configuration."""
    canon = json.dumps(cfg, sort_keys=True, ensure_ascii=True, default=str)  # Sérialise de façon canonique.
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]  # Tronque l'empreinte SHA-256.
