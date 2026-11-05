# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Diagnostics des représentations : alignement, variance, dimension effective, effondrement, séparation."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from typing import Any, Dict  # Importe les types des signatures.

import numpy as np  # Effectue les calculs matriciels.
import torch  # Fournit les tenseurs.
import torch.nn.functional as F  # Fournit la normalisation.

from data.feature_cache import FeatureSplit  # Importe l'accès aux caractéristiques.
from models.multimodal_framework import modality_inputs  # Importe la conversion des lots.


def effective_rank(x: np.ndarray) -> float:  # Calcule la dimension effective.
    """Retourne le rapport de participation (somme lambda)² / somme lambda², où lambda sont les variances principales."""
    centre = x - x.mean(axis=0, keepdims=True)  # Centre les données.
    valeurs = np.linalg.svd(centre, compute_uv=False) ** 2  # Calcule les variances principales.
    return float(valeurs.sum() ** 2 / max((valeurs ** 2).sum(), 1e-12))  # Retourne le rapport de participation.


def collapse_flags(z: np.ndarray, seuil_std: float = 1e-3, seuil_rang: float = 2.0) -> Dict[str, Any]:  # Détecte l'effondrement.
    """Mesure la variance moyenne et la dimension effective, puis indique un effondrement éventuel."""
    ecart = float(z.std(axis=0).mean())  # Calcule l'écart-type moyen par dimension.
    rang = effective_rank(z)  # Calcule la dimension effective.
    return {"ecart_type_moyen": ecart, "rang_effectif": rang, "effondre": bool(ecart < seuil_std or rang < seuil_rang)}  # Retourne le diagnostic.


def alignment_report(z_image: np.ndarray, z_texte: np.ndarray) -> Dict[str, float]:  # Mesure l'alignement inter-modal.
    """Compare la similarité des vraies paires à celle de paires non correspondantes."""
    similarites = np.matmul(z_image, z_texte.T)  # Calcule toutes les similarités.
    positives = np.diag(similarites)  # Lit les similarités des vraies paires.
    masque = ~np.eye(similarites.shape[0], dtype=bool)  # Prépare le masque des paires non correspondantes.
    negatives = similarites[masque]  # Lit les similarités des paires non correspondantes.
    return {"cosinus_paires_vraies": float(positives.mean()), "cosinus_paires_fausses": float(negatives.mean()), "marge_alignement": float(positives.mean() - negatives.mean())}  # Retourne le rapport.


@torch.no_grad()  # Désactive le calcul du gradient.
def representation_diagnostics(modele: torch.nn.Module, split: FeatureSplit, nombre: int, appareil: torch.device) -> Dict[str, Any]:  # Exécute tous les diagnostics.
    """Exécute les diagnostics sur un échantillon apparié et déterministe du sous-ensemble fourni."""
    modele.eval()  # Désactive le dropout.
    n = min(nombre, split.nb_images)  # Limite le nombre d'échantillons.
    premieres = np.asarray([liste[0] for liste in split.legendes_par_image[:n]], dtype=np.int64)  # Sélectionne la première légende de chaque image.
    lot = split.batch_from_indices(np.arange(n), premieres, appareil)  # Assemble le lot apparié.
    sorties = modele(modality_inputs(lot))  # Exécute le modèle.
    z_i = sorties["embeddings"]["image"].cpu().numpy()  # Lit les embeddings d'images.
    z_t = sorties["embeddings"]["text"].cpu().numpy()  # Lit les embeddings de textes.
    rapport: Dict[str, Any] = {  # Construit le rapport de base.
        "alignement": alignment_report(z_i, z_t),  # Mesure l'alignement inter-modal.
        "image": collapse_flags(z_i),  # Mesure l'effondrement des images.
        "texte": collapse_flags(z_t),  # Mesure l'effondrement des textes.
        "temperature": float(modele.temperature.item()),  # Lit la température apprise.
        "porte_structurelle": float(modele.porte.item()),  # Lit le poids appris de la voie structurelle.
    }  # Termine le rapport de base.
    if sorties["shared"]:  # Vérifie la présence de la banque.
        separation = {}  # Prépare le rapport de séparation.
        for nom in sorties["shared"]:  # Parcourt les modalités.
            s = F.normalize(sorties["shared"][nom], dim=-1)  # Normalise les tokens partagés.
            p = F.normalize(sorties["private"][nom], dim=-1)  # Normalise les tokens privés.
            separation[f"cosinus_absolu_partage_prive_{nom}"] = float(torch.matmul(s, p.transpose(1, 2)).abs().mean().item())  # Mesure le recouvrement partagé/privé.
            separation[f"effondrement_prive_{nom}"] = collapse_flags(sorties["private"][nom].mean(dim=1).cpu().numpy())  # Mesure l'effondrement de l'espace privé.
            separation[f"effondrement_partage_{nom}"] = collapse_flags(sorties["shared"][nom].mean(dim=1).cpu().numpy())  # Mesure l'effondrement de l'espace partagé.
            reconstruction = F.cosine_similarity(sorties["reconstruction"][nom], sorties["pooled_input"][nom], dim=-1).mean()  # Mesure la qualité de la reconstruction.
            separation[f"cosinus_reconstruction_{nom}"] = float(reconstruction.item())  # Enregistre la reconstruction.
        partage_i = F.normalize(sorties["shared"]["image"].mean(dim=1), dim=-1)  # Résume les tokens partagés d'images.
        partage_t = F.normalize(sorties["shared"]["text"].mean(dim=1), dim=-1)  # Résume les tokens partagés de textes.
        prive_i = F.normalize(sorties["private"]["image"].mean(dim=1), dim=-1)  # Résume les tokens privés d'images.
        prive_t = F.normalize(sorties["private"]["text"].mean(dim=1), dim=-1)  # Résume les tokens privés de textes.
        separation["alignement_partage_inter_modal"] = float((partage_i * partage_t).sum(dim=-1).mean().item())  # Mesure l'accord entre espaces partagés.
        separation["alignement_prive_inter_modal"] = float((prive_i * prive_t).sum(dim=-1).mean().item())  # Mesure l'accord entre espaces privés.
        rapport["partage_prive"] = separation  # Ajoute la séparation au rapport.
        rapport["poids_fusion_moyens"] = {nom: float(sorties["alphas"][:, rang].mean().item()) for rang, nom in enumerate(sorties["alpha_names"])}  # Lit les poids de fusion moyens.
    if hasattr(modele, "filtration"):  # Vérifie la présence de la filtration.
        rapport["lipschitz"] = modele.filtration.lipschitz_bound()  # Lit la constante de Lipschitz mesurée.
        rapport["echelles_apprises"] = [float(v) for v in modele.filtration.echelles.cpu().numpy()]  # Lit les échelles apprises.
    if hasattr(modele, "bloc"):  # Vérifie la présence du bloc d'attention.
        rapport["lambda_attention"] = [float(v) for v in modele.bloc.attention.lambda_biais.flatten().cpu().numpy()]  # Lit les coefficients du biais.
    return rapport  # Retourne le rapport complet.
