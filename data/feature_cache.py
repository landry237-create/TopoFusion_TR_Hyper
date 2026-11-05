# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Extraction et cache des caractéristiques produites par les encodeurs gelés.

Mettre ces caractéristiques en cache est scientifiquement valide : les encodeurs
sont gelés et déterministes, donc leur sortie ne dépend pas de l'entraînement.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import json  # Écrit les métadonnées de provenance.
import re  # Construit des noms de dossier sûrs.
import time  # Mesure le débit d'extraction.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple  # Importe les types des signatures.

import numpy as np  # Fournit les tableaux projetés en mémoire.
import torch  # Fournit les tenseurs.

from data.corruptions import CORRUPTIONS_IMAGE, CORRUPTIONS_TEXTE  # Importe les registres de corruptions.
from data.dataset_loader import ImageStore, SPLITS, relevance_from_manifest  # Importe l'accès aux images et l'index des légendes.
from models.encoders import ClipGlobalEncoder, FrozenImageEncoder, FrozenTextEncoder, feature_origin  # Importe les encodeurs gelés.


def _nom_sur(nom: str) -> str:  # Convertit un identifiant en nom de dossier.
    """Remplace les caractères non sûrs d'un identifiant de modèle."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", nom)  # Retourne le nom nettoyé.


def scenario_name(nom: str, severite: float) -> str:  # Construit le nom d'un scénario.
    """Retourne un nom de scénario lisible, par exemple image_blur_radius_2.0."""
    return f"{nom}_{float(severite)}"  # Assemble le nom et la sévérité.


def _ecrire_par_lots(chemin: Path, nombre: int, lots: Iterator[np.ndarray], dtype: np.dtype) -> None:  # Écrit un tableau par morceaux.
    """Écrit un tableau sur disque par lots, puis le publie de façon atomique."""
    temporaire = chemin.with_name(chemin.name + ".part")  # Prépare un fichier temporaire.
    tableau: Optional[np.memmap] = None  # Initialise le tableau projeté.
    position = 0  # Initialise la position d'écriture.
    for lot in lots:  # Parcourt les lots produits.
        if tableau is None:  # Vérifie si le fichier doit être créé.
            tableau = np.lib.format.open_memmap(temporaire, mode="w+", dtype=dtype, shape=(nombre,) + lot.shape[1:])  # Crée le fichier à la bonne taille.
        tableau[position: position + len(lot)] = lot  # Copie le lot dans le fichier.
        position += len(lot)  # Avance la position d'écriture.
    if tableau is None or position != nombre:  # Vérifie que toutes les lignes ont été écrites.
        raise RuntimeError(f"Extraction incomplète pour {chemin} : {position}/{nombre}.")  # Refuse un fichier partiel.
    tableau.flush()  # Force l'écriture sur le disque.
    del tableau  # Ferme le tableau projeté, ce qui est nécessaire sous Windows.
    temporaire.replace(chemin)  # Publie le fichier complet.


def _lots(elements: Sequence[Any], taille: int) -> Iterator[Sequence[Any]]:  # Découpe une séquence en lots.
    """Parcourt une séquence par lots successifs."""
    for debut in range(0, len(elements), taille):  # Parcourt les positions de début.
        yield elements[debut: debut + taille]  # Retourne un lot.


class FeatureExtractor:  # Pilote l'extraction des caractéristiques.
    """Calcule et met en cache les caractéristiques des encodeurs gelés."""

    def __init__(self, cfg: Dict[str, Any], manifeste: Dict[str, Any]) -> None:  # Prépare l'extracteur.
        """Prépare les chemins et reporte le chargement des encodeurs."""
        self.cfg = cfg  # Conserve la configuration.
        self.manifeste = manifeste  # Conserve le manifeste.
        self.store = ImageStore(cfg)  # Prépare l'accès aux images.
        enc = cfg["encoders"]  # Récupère la section des encodeurs.
        self.taille_lot = int(enc["batch_size"])  # Fixe la taille des lots d'extraction.
        self.racine = Path(cfg["dataset"]["cache_dir"]) / "features" / _nom_sur(f"{enc['image']}__{enc['text']}__g{enc['image_grid']}__t{enc['max_text_tokens']}")  # Fixe le dossier des tokens.
        self.racine_clip = Path(cfg["dataset"]["cache_dir"]) / "features" / _nom_sur(f"clip__{cfg['clip_baseline']['model']}") if cfg.get("clip_baseline", {}).get("enabled") else None  # Fixe le dossier de CLIP.
        self.racine.mkdir(parents=True, exist_ok=True)  # Crée le dossier des tokens.
        if self.racine_clip is not None:  # Vérifie que CLIP est activé.
            self.racine_clip.mkdir(parents=True, exist_ok=True)  # Crée le dossier de CLIP.
        self._image: Optional[FrozenImageEncoder] = None  # Reporte le chargement de l'encodeur d'image.
        self._texte: Optional[FrozenTextEncoder] = None  # Reporte le chargement de l'encodeur de texte.
        self._clip: Optional[ClipGlobalEncoder] = None  # Reporte le chargement de CLIP.

    @property  # Expose le chargement paresseux sous forme d'attribut.
    def encodeur_image(self) -> FrozenImageEncoder:  # Fournit l'encodeur d'image.
        """Charge l'encodeur d'image lors de la première utilisation."""
        if self._image is None:  # Vérifie si le chargement est nécessaire.
            self._image = FrozenImageEncoder(self.cfg["encoders"]["image"], self.cfg["encoders"]["image_grid"])  # Charge l'encodeur.
        return self._image  # Retourne l'encodeur.

    @property  # Expose le chargement paresseux sous forme d'attribut.
    def encodeur_texte(self) -> FrozenTextEncoder:  # Fournit l'encodeur de texte.
        """Charge l'encodeur de texte lors de la première utilisation."""
        if self._texte is None:  # Vérifie si le chargement est nécessaire.
            self._texte = FrozenTextEncoder(self.cfg["encoders"]["text"], self.cfg["encoders"]["max_text_tokens"])  # Charge l'encodeur.
        return self._texte  # Retourne l'encodeur.

    @property  # Expose le chargement paresseux sous forme d'attribut.
    def encodeur_clip(self) -> ClipGlobalEncoder:  # Fournit la référence CLIP.
        """Charge CLIP lors de la première utilisation."""
        if self._clip is None:  # Vérifie si le chargement est nécessaire.
            self._clip = ClipGlobalEncoder(self.cfg["clip_baseline"]["model"])  # Charge le modèle.
        return self._clip  # Retourne le modèle.

    def _images_sequence(self, fichiers: Sequence[str], transformation: Optional[Callable] = None) -> Iterator[Sequence[Any]]:  # Produit des lots d'images.
        """Produit des lots d'images réelles, éventuellement corrompues de façon déterministe."""
        for lot in _lots(list(fichiers), self.taille_lot):  # Parcourt les fichiers par lots.
            images = [self.store.open(nom) for nom in lot]  # Ouvre les images du lot.
            if transformation is not None:  # Vérifie si une corruption est demandée.
                images = [transformation(image, nom) for image, nom in zip(images, lot)]  # Applique la corruption.
            yield images  # Retourne le lot.

    def _extraire_images(self, racine: Path, prefixe: str, fichiers: Sequence[str], clip: bool, transformation: Optional[Callable] = None) -> None:  # Extrait les caractéristiques d'images.
        """Extrait et enregistre les caractéristiques d'images, si elles sont absentes."""
        chemin = racine / f"{prefixe}.image.npy"  # Calcule le chemin de sortie.
        if chemin.is_file():  # Vérifie si le travail est déjà fait.
            return  # Réutilise le fichier existant.
        debut = time.time()  # Mémorise l'heure de début.

        def produire() -> Iterator[np.ndarray]:  # Définit le générateur de lots.
            """Encode les images et convertit les résultats en demi-précision."""
            for images in self._images_sequence(fichiers, transformation):  # Parcourt les lots d'images.
                sortie = self.encodeur_clip.encode_images(images) if clip else self.encodeur_image(images)  # Encode le lot.
                yield sortie.cpu().numpy().astype(np.float16)  # Convertit pour économiser le disque.

        _ecrire_par_lots(chemin, len(fichiers), produire(), np.dtype(np.float16))  # Écrit le tableau.
        print(f"Images encodées : {chemin.name} ({len(fichiers)} en {time.time() - debut:.0f} s)", flush=True)  # Affiche le débit.

    def _extraire_textes(self, racine: Path, prefixe: str, textes: Sequence[str], clip: bool) -> None:  # Extrait les caractéristiques de textes.
        """Extrait et enregistre les caractéristiques de textes, si elles sont absentes."""
        chemin = racine / f"{prefixe}.text.npy"  # Calcule le chemin des tokens.
        chemin_masque = racine / f"{prefixe}.text_mask.npy"  # Calcule le chemin du masque.
        if chemin.is_file() and (clip or chemin_masque.is_file()):  # Vérifie si le travail est déjà fait.
            return  # Réutilise les fichiers existants.
        debut = time.time()  # Mémorise l'heure de début.
        ordre = np.argsort([len(texte) for texte in textes])  # Regroupe des textes de longueur voisine pour accélérer CLIP.
        inverse = np.argsort(ordre)  # Calcule la permutation inverse.
        if clip:  # Traite le cas de CLIP.
            morceaux = [self.encodeur_clip.encode_texts([textes[int(i)] for i in lot]).cpu().numpy() for lot in _lots(ordre, self.taille_lot * 2)]  # Encode tous les lots.
            sortie = np.concatenate(morceaux, axis=0)[inverse].astype(np.float16)  # Rétablit l'ordre d'origine.
            np.save(chemin, sortie)  # Écrit les embeddings.
        else:  # Traite le cas de l'encodeur de texte.
            masques: List[np.ndarray] = []  # Prépare les masques.

            def produire() -> Iterator[np.ndarray]:  # Définit le générateur de lots.
                """Encode les textes dans l'ordre d'origine et conserve leurs masques."""
                for lot in _lots(list(textes), self.taille_lot * 2):  # Parcourt les textes par lots.
                    tokens, masque = self.encodeur_texte(lot)  # Encode le lot.
                    masques.append(masque.cpu().numpy())  # Conserve le masque.
                    yield tokens.cpu().numpy().astype(np.float16)  # Convertit pour économiser le disque.

            _ecrire_par_lots(chemin, len(textes), produire(), np.dtype(np.float16))  # Écrit les tokens.
            np.save(chemin_masque, np.concatenate(masques, axis=0))  # Écrit les masques.
        print(f"Textes encodés : {chemin.name} ({len(textes)} en {time.time() - debut:.0f} s)", flush=True)  # Affiche le débit.

    def extract_clean(self) -> None:  # Extrait les caractéristiques des données propres.
        """Extrait les caractéristiques propres de tous les sous-ensembles."""
        for nom in SPLITS:  # Parcourt les sous-ensembles.
            entrees = self.manifeste["splits"][nom]  # Récupère les entrées du manifeste.
            fichiers = [entree["file"] for entree in entrees]  # Récupère les fichiers image.
            legendes, _, _ = relevance_from_manifest(entrees)  # Aplatit les légendes.
            self._extraire_images(self.racine, nom, fichiers, clip=False)  # Extrait les tokens d'images.
            self._extraire_textes(self.racine, nom, legendes, clip=False)  # Extrait les tokens de textes.
            if self.racine_clip is not None and nom == "test":  # La référence CLIP n'est évaluée que sur le test.
                self._extraire_images(self.racine_clip, nom, fichiers, clip=True)  # Extrait les embeddings d'images CLIP.
                self._extraire_textes(self.racine_clip, nom, legendes, clip=True)  # Extrait les embeddings de textes CLIP.
        self._ecrire_provenance()  # Enregistre la provenance des caractéristiques.

    def extract_corrupted(self, scenarios: Sequence[Tuple[str, str, float]]) -> None:  # Extrait les scénarios corrompus.
        """Extrait les caractéristiques de test pour chaque scénario de corruption."""
        graine = int(self.cfg["evaluation"]["corruption_seed"])  # Récupère la graine des corruptions.
        entrees = self.manifeste["splits"]["test"]  # Récupère les entrées de test.
        fichiers = [entree["file"] for entree in entrees]  # Récupère les fichiers image.
        legendes, _, _ = relevance_from_manifest(entrees)  # Aplatit les légendes.
        for modalite, nom, severite in scenarios:  # Parcourt les scénarios.
            prefixe = f"test__{scenario_name(nom, severite)}"  # Calcule le préfixe de sortie.
            if modalite == "image":  # Traite une corruption d'image.
                fonction = CORRUPTIONS_IMAGE[nom]  # Récupère la corruption.
                transformation = lambda image, cle, f=fonction, s=severite: f(image, s, graine, cle)  # Lie la sévérité et la graine.
                self._extraire_images(self.racine, prefixe, fichiers, clip=False, transformation=transformation)  # Extrait les tokens corrompus.
                if self.racine_clip is not None:  # Vérifie que CLIP est activé.
                    self._extraire_images(self.racine_clip, prefixe, fichiers, clip=True, transformation=transformation)  # Extrait les embeddings CLIP corrompus.
            else:  # Traite une corruption de texte.
                fonction = CORRUPTIONS_TEXTE[nom]  # Récupère la corruption.
                textes = [fonction(texte, severite, graine, f"{rang}") for rang, texte in enumerate(legendes)]  # Applique la corruption à chaque légende.
                self._extraire_textes(self.racine, prefixe, textes, clip=False)  # Extrait les tokens corrompus.
                if self.racine_clip is not None:  # Vérifie que CLIP est activé.
                    self._extraire_textes(self.racine_clip, prefixe, textes, clip=True)  # Extrait les embeddings CLIP corrompus.

    def _ecrire_provenance(self) -> None:  # Enregistre l'origine des caractéristiques.
        """Écrit un fichier indiquant quel modèle a produit quelles caractéristiques."""
        enc = self.cfg["encoders"]  # Récupère la section des encodeurs.
        contenu = {  # Construit la description.
            "image": {"modele": enc["image"], "origine": feature_origin(enc["image"]), "grille": enc["image_grid"]},  # Décrit l'encodeur d'image.
            "texte": {"modele": enc["text"], "origine": feature_origin(enc["text"]), "tokens_max": enc["max_text_tokens"]},  # Décrit l'encodeur de texte.
            "format": "float16, encodeurs gelés en mode évaluation",  # Décrit le format de stockage.
        }  # Termine la description.
        (self.racine / "provenance.json").write_text(json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")  # Écrit le fichier.


class FeatureSplit:  # Donne accès à un sous-ensemble de caractéristiques.
    """Sous-ensemble en mémoire projetée : tokens, masques et correspondance image-texte."""

    def __init__(self, racine: Path, entrees: Sequence[Dict[str, Any]], prefixe: str, prefixe_image: Optional[str] = None, prefixe_texte: Optional[str] = None) -> None:  # Ouvre les fichiers.
        """Ouvre les tableaux projetés en mémoire et construit l'index des légendes."""
        racine = Path(racine)  # Normalise le chemin reçu.
        self.images = np.load(racine / f"{prefixe_image or prefixe}.image.npy", mmap_mode="r")  # Ouvre les tokens d'images.
        self.textes = np.load(racine / f"{prefixe_texte or prefixe}.text.npy", mmap_mode="r")  # Ouvre les tokens de textes.
        self.masques_texte = np.load(racine / f"{prefixe_texte or prefixe}.text_mask.npy", mmap_mode="r")  # Ouvre les masques de textes.
        self.legendes, sources, self.images_par_texte = relevance_from_manifest(entrees)  # Construit l'index des légendes.
        self.image_de_legende = np.asarray(sources, dtype=np.int64)  # Associe chaque légende à son image source.
        self.nb_images = int(self.images.shape[0])  # Compte les images.
        self.nb_legendes = int(self.textes.shape[0])  # Compte les légendes.
        if self.nb_legendes != len(self.legendes):  # Vérifie la cohérence avec le manifeste.
            raise RuntimeError(f"Incohérence du cache : {self.nb_legendes} tokens, {len(self.legendes)} légendes.")  # Refuse un cache incohérent.
        identifiants: Dict[str, int] = {}  # Prépare l'attribution d'un identifiant par texte.
        self.id_texte = np.asarray([identifiants.setdefault(texte, len(identifiants)) for texte in self.legendes], dtype=np.int64)  # Associe un identifiant à chaque texte.
        self.legendes_par_image: List[np.ndarray] = [np.where(self.image_de_legende == rang)[0] for rang in range(self.nb_images)]  # Liste les légendes de chaque image.

    def relevance_matrix(self) -> np.ndarray:  # Construit la matrice de pertinence.
        """Retourne la matrice image × légende ; un texte identique reste pertinent pour chaque image qu'il décrit."""
        matrice = np.zeros((self.nb_images, self.nb_legendes), dtype=bool)  # Initialise la matrice.
        for rang, texte in enumerate(self.legendes):  # Parcourt les légendes.
            matrice[self.images_par_texte[texte], rang] = True  # Marque toutes les images décrites par ce texte.
        return matrice  # Retourne la matrice.

    def batch_from_indices(self, indices_images: np.ndarray, indices_legendes: np.ndarray, appareil: torch.device) -> Dict[str, torch.Tensor]:  # Assemble un lot.
        """Assemble un lot apparié de tokens image et texte à partir d'indices."""
        return {  # Construit le lot.
            "image_tokens": torch.from_numpy(np.asarray(self.images[indices_images], dtype=np.float32)).to(appareil),  # Tokens d'images.
            "text_tokens": torch.from_numpy(np.asarray(self.textes[indices_legendes], dtype=np.float32)).to(appareil),  # Tokens de textes.
            "text_mask": torch.from_numpy(np.asarray(self.masques_texte[indices_legendes], dtype=bool)).to(appareil),  # Masque de padding.
            "text_ids": torch.from_numpy(self.id_texte[indices_legendes]).to(appareil),  # Identifiants de texte pour exclure les faux négatifs.
        }  # Termine le lot.

    def iter_train_batches(self, taille_lot: int, generateur: np.random.Generator, appareil: torch.device) -> Iterator[Dict[str, torch.Tensor]]:  # Parcourt l'entraînement.
        """Parcourt les images au hasard en tirant une légende réelle de chacune ; une image n'apparaît qu'une fois par époque."""
        permutation = generateur.permutation(self.nb_images)  # Mélange les images.
        for debut in range(0, self.nb_images - taille_lot + 1, taille_lot):  # Ignore le dernier lot incomplet, pour garder des négatifs comparables.
            lot_images = permutation[debut: debut + taille_lot]  # Sélectionne les images du lot.
            lot_legendes = np.asarray([generateur.choice(self.legendes_par_image[int(rang)]) for rang in lot_images], dtype=np.int64)  # Tire une légende de chaque image.
            yield self.batch_from_indices(lot_images, lot_legendes, appareil)  # Retourne le lot.

    def iter_eval_pairs(self, taille_lot: int, appareil: torch.device) -> Iterator[Dict[str, torch.Tensor]]:  # Parcourt une évaluation appariée.
        """Parcourt dans l'ordre chaque image et sa première légende, de manière déterministe."""
        premieres = np.asarray([liste[0] for liste in self.legendes_par_image], dtype=np.int64)  # Sélectionne la première légende de chaque image.
        for debut in range(0, self.nb_images, taille_lot):  # Parcourt les images par lots.
            fin = min(debut + taille_lot, self.nb_images)  # Calcule la fin du lot.
            if fin - debut >= 2:  # Ignore un dernier lot trop petit pour InfoNCE.
                yield self.batch_from_indices(np.arange(debut, fin), premieres[debut:fin], appareil)  # Retourne le lot.


def open_split(cfg: Dict[str, Any], manifeste: Dict[str, Any], nom: str) -> FeatureSplit:  # Ouvre un sous-ensemble propre.
    """Ouvre les caractéristiques propres d'un sous-ensemble."""
    extracteur = FeatureExtractor(cfg, manifeste)  # Calcule les chemins du cache.
    return FeatureSplit(extracteur.racine, manifeste["splits"][nom], nom)  # Retourne le sous-ensemble.
