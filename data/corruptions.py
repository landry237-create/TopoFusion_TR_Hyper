# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Perturbations contrôlées et déterministes des vraies entrées image et texte."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import hashlib  # Dérive une graine stable pour chaque échantillon.
import string  # Fournit l'alphabet utilisé pour remplacer des caractères.
from typing import Callable, Dict, List, Tuple  # Importe les types des signatures.

import numpy as np  # Effectue les opérations sur les pixels.
from PIL import Image, ImageFilter  # Manipule les images et le flou gaussien.


def _generateur(graine: int, cle: str) -> np.random.Generator:  # Crée un générateur propre à un échantillon.
    """Retourne un générateur reproductible qui dépend de la graine et de l'échantillon."""
    condensat = hashlib.sha256(f"{graine}:{cle}".encode("utf-8")).digest()  # Hache la graine et la clé.
    return np.random.default_rng(int.from_bytes(condensat[:8], "little"))  # Initialise le générateur.


def gaussian_noise(image: Image.Image, sigma: float, graine: int, cle: str) -> Image.Image:  # Ajoute un bruit gaussien.
    """Ajoute un bruit gaussien d'écart-type sigma, exprimé sur des pixels normalisés dans [0, 1]."""
    pixels = np.asarray(image, dtype=np.float32) / 255.0  # Normalise les pixels.
    bruit = _generateur(graine, cle).normal(0.0, sigma, pixels.shape)  # Tire le bruit.
    return Image.fromarray((np.clip(pixels + bruit, 0.0, 1.0) * 255.0).astype(np.uint8))  # Retourne l'image bruitée.


def gaussian_blur(image: Image.Image, rayon: float, graine: int, cle: str) -> Image.Image:  # Applique un flou.
    """Applique un flou gaussien de rayon donné ; l'opération n'est pas aléatoire."""
    return image.filter(ImageFilter.GaussianBlur(radius=rayon))  # Retourne l'image floutée.


def random_mask(image: Image.Image, fraction: float, graine: int, cle: str, cote_bloc: int = 32) -> Image.Image:  # Masque des blocs.
    """Remplace par du gris une fraction des blocs carrés de l'image."""
    generateur = _generateur(graine, cle)  # Crée le générateur de l'échantillon.
    pixels = np.array(image, dtype=np.uint8)  # Copie les pixels pour pouvoir les modifier.
    hauteur, largeur = pixels.shape[:2]  # Lit les dimensions de l'image.
    lignes = int(np.ceil(hauteur / cote_bloc))  # Calcule le nombre de lignes de blocs.
    colonnes = int(np.ceil(largeur / cote_bloc))  # Calcule le nombre de colonnes de blocs.
    nb_blocs = lignes * colonnes  # Calcule le nombre total de blocs.
    choisis = generateur.choice(nb_blocs, size=int(round(fraction * nb_blocs)), replace=False)  # Tire les blocs à masquer.
    for indice in choisis:  # Parcourt les blocs tirés.
        ligne, colonne = np.unravel_index(int(indice), (lignes, colonnes))  # Convertit l'indice en position.
        pixels[ligne * cote_bloc: (ligne + 1) * cote_bloc, colonne * cote_bloc: (colonne + 1) * cote_bloc] = 127  # Remplace le bloc par du gris.
    return Image.fromarray(pixels)  # Retourne l'image masquée.


def token_deletion(texte: str, probabilite: float, graine: int, cle: str) -> str:  # Supprime des mots.
    """Supprime chaque mot avec la probabilité donnée et conserve toujours au moins un mot."""
    mots = texte.split()  # Découpe le texte en mots.
    tirages = _generateur(graine, cle).random(len(mots))  # Tire une valeur par mot.
    restants = [mot for mot, tirage in zip(mots, tirages) if tirage >= probabilite]  # Conserve les mots non supprimés.
    if not restants:  # Détecte une suppression totale.
        restants = [mots[int(np.argmax(tirages))]]  # Conserve le mot associé au plus grand tirage.
    return " ".join(restants)  # Recompose le texte.


def char_corruption(texte: str, probabilite: float, graine: int, cle: str) -> str:  # Corrompt des caractères.
    """Remplace chaque lettre par une autre avec la probabilité donnée."""
    generateur = _generateur(graine, cle)  # Crée le générateur de l'échantillon.
    lettres = string.ascii_lowercase  # Définit l'alphabet de remplacement.
    sortie: List[str] = []  # Prépare les caractères de sortie.
    for caractere in texte:  # Parcourt les caractères du texte.
        if caractere.isalpha() and generateur.random() < probabilite:  # Décide si cette lettre est corrompue.
            sortie.append(lettres[int(generateur.integers(0, len(lettres)))])  # Ajoute une lettre tirée au hasard.
        else:  # Traite un caractère conservé.
            sortie.append(caractere)  # Conserve le caractère d'origine.
    return "".join(sortie)  # Recompose le texte.


CORRUPTIONS_IMAGE: Dict[str, Callable[..., Image.Image]] = {  # Associe chaque nom à une corruption d'image.
    "image_gaussian_noise": gaussian_noise,  # Bruit gaussien.
    "image_blur_radius": gaussian_blur,  # Flou gaussien.
    "image_random_mask": random_mask,  # Masquage aléatoire.
}  # Termine le registre des corruptions d'image.

CORRUPTIONS_TEXTE: Dict[str, Callable[..., str]] = {  # Associe chaque nom à une corruption de texte.
    "text_token_deletion": token_deletion,  # Suppression de mots.
    "text_char_corruption": char_corruption,  # Corruption de caractères.
}  # Termine le registre des corruptions de texte.


def list_scenarios(cfg: Dict) -> List[Tuple[str, str, float]]:  # Énumère les scénarios de corruption.
    """Retourne les scénarios configurés sous la forme (modalité, nom, sévérité)."""
    scenarios: List[Tuple[str, str, float]] = []  # Prépare la liste.
    for nom, severites in cfg["corruptions"].items():  # Parcourt les corruptions configurées.
        if nom not in CORRUPTIONS_IMAGE and nom not in CORRUPTIONS_TEXTE:  # Vérifie que le nom est connu.
            raise ValueError(f"Corruption inconnue : {nom}. Attendu : {sorted(CORRUPTIONS_IMAGE) + sorted(CORRUPTIONS_TEXTE)}")  # Explique l'erreur.
        modalite = "image" if nom in CORRUPTIONS_IMAGE else "text"  # Détermine la modalité touchée.
        for severite in severites:  # Parcourt les sévérités.
            scenarios.append((modalite, nom, float(severite)))  # Ajoute le scénario.
    return scenarios  # Retourne les scénarios.
