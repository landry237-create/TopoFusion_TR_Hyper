# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Encodeurs gelés des modalités et projecteurs appris.

Image et texte constituent le benchmark. Audio et vidéo sont des extensions
expérimentales : elles acceptent uniquement de vraies entrées et ne sont
jamais appelées par le pipeline, car aucune donnée réelle n'est intégrée.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import math  # Fournit la racine carrée entière pour la grille de patchs.
from typing import Dict, List, Sequence, Tuple  # Importe les types des signatures.

import torch  # Fournit les tenseurs.
import torch.nn as nn  # Fournit les modules de réseaux de neurones.
import torch.nn.functional as F  # Fournit le pooling adaptatif.
from PIL import Image  # Déclare le type des images en entrée.


class ModalityUnavailableError(RuntimeError):  # Définit l'erreur liée à une modalité sans vraie donnée.
    """Signale qu'une modalité expérimentale a été appelée sans véritable entrée."""


def feature_origin(nom_modele: str) -> str:  # Décrit la provenance d'un encodeur.
    """Indique si un encodeur provient de poids ouverts Meta ou d'un autre fournisseur."""
    if nom_modele.startswith(("facebook/", "meta/", "meta-llama/")):  # Détecte un dépôt publié par Meta.
        return f"Poids ouverts Meta, exécution locale : {nom_modele}"  # Indique l'origine Meta.
    return f"Encodeur local tiers : {nom_modele}"  # Indique une autre origine.


class _EncodeurGele(nn.Module):  # Factorise le gel strict des encodeurs.
    """Classe de base d'un encodeur gelé, maintenu en mode évaluation."""

    def _geler(self, modele: nn.Module) -> None:  # Gèle les poids et désactive le dropout.
        """Gèle un modèle et place ses couches en mode évaluation."""
        modele.eval()  # Désactive le dropout interne.
        for parametre in modele.parameters():  # Parcourt les poids du modèle.
            parametre.requires_grad_(False)  # Interdit leur mise à jour.

    def train(self, mode: bool = True) -> "_EncodeurGele":  # Remplace le comportement standard de train().
        """Ignore l'appel à train : un encodeur gelé ne doit jamais réactiver son dropout."""
        return self  # Retourne le module sans changer son mode.


class FrozenImageEncoder(_EncodeurGele):  # Encode les images en tokens locaux.
    """Encodeur d'image gelé qui produit le jeton global et une grille réduite de patchs."""

    def __init__(self, nom_modele: str, grille: int = 4) -> None:  # Charge le modèle et son préprocesseur.
        """Charge l'encodeur et son préprocesseur depuis Hugging Face."""
        super().__init__()  # Initialise le module parent.
        from transformers import AutoImageProcessor, AutoModel  # Importe transformers seulement lorsque c'est nécessaire.

        self.nom_modele = nom_modele  # Conserve le nom du modèle.
        self.grille = int(grille)  # Conserve la taille de la grille de patchs.
        self.preprocesseur = AutoImageProcessor.from_pretrained(nom_modele)  # Charge le préprocesseur officiel.
        self.modele = AutoModel.from_pretrained(nom_modele)  # Charge les poids pré-entraînés.
        self._geler(self.modele)  # Gèle les poids et désactive le dropout.
        self.dim_sortie = int(self.modele.config.hidden_size)  # Lit la dimension des tokens.
        self.nb_registres = int(getattr(self.modele.config, "num_register_tokens", 0) or 0)  # Lit le nombre de jetons de registre.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def forward(self, images: Sequence[Image.Image]) -> torch.Tensor:  # Encode une liste d'images.
        """Retourne un tenseur de forme (B, 1 + grille², D)."""
        entrees = self.preprocesseur(images=list(images), return_tensors="pt")  # Applique le prétraitement officiel.
        sorties = self.modele(pixel_values=entrees["pixel_values"].to(next(self.modele.parameters()).device))  # Exécute l'encodeur.
        etats = sorties.last_hidden_state  # Récupère tous les tokens.
        jeton_global = etats[:, :1]  # Extrait le jeton global.
        patchs = etats[:, 1 + self.nb_registres:]  # Extrait les tokens de patchs.
        cote = int(math.isqrt(patchs.size(1)))  # Calcule le côté de la grille de patchs.
        if cote * cote != patchs.size(1):  # Vérifie que les patchs forment un carré.
            raise ValueError(f"Grille de patchs non carrée : {patchs.size(1)} tokens.")  # Explique l'erreur.
        carte = patchs.transpose(1, 2).reshape(patchs.size(0), self.dim_sortie, cote, cote)  # Reforme la carte spatiale.
        reduits = F.adaptive_avg_pool2d(carte, self.grille).flatten(2).transpose(1, 2)  # Réduit la grille de patchs.
        return torch.cat([jeton_global, reduits], dim=1)  # Assemble le jeton global et les patchs.


class FrozenTextEncoder(_EncodeurGele):  # Encode les légendes en tokens locaux.
    """Encodeur de texte gelé qui conserve les tokens et le masque de padding."""

    def __init__(self, nom_modele: str, max_tokens: int = 24) -> None:  # Charge le modèle et son tokeniseur.
        """Charge l'encodeur et son tokeniseur depuis Hugging Face."""
        super().__init__()  # Initialise le module parent.
        from transformers import AutoModel, AutoTokenizer  # Importe transformers seulement lorsque c'est nécessaire.

        self.nom_modele = nom_modele  # Conserve le nom du modèle.
        self.max_tokens = int(max_tokens)  # Conserve la longueur maximale.
        self.tokeniseur = AutoTokenizer.from_pretrained(nom_modele)  # Charge le tokeniseur officiel.
        self.modele = AutoModel.from_pretrained(nom_modele)  # Charge les poids pré-entraînés.
        self._geler(self.modele)  # Gèle les poids et désactive le dropout.
        self.dim_sortie = int(self.modele.config.hidden_size)  # Lit la dimension des tokens.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def forward(self, textes: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:  # Encode une liste de légendes.
        """Retourne les tokens (B, T, D) et le masque booléen (B, T) des tokens réels."""
        entrees = self.tokeniseur(list(textes), padding="max_length", truncation=True, max_length=self.max_tokens, return_tensors="pt")  # Tokenise avec une longueur fixe.
        appareil = next(self.modele.parameters()).device  # Récupère le device du modèle.
        sorties = self.modele(input_ids=entrees["input_ids"].to(appareil), attention_mask=entrees["attention_mask"].to(appareil))  # Exécute l'encodeur.
        return sorties.last_hidden_state, entrees["attention_mask"].to(appareil).bool()  # Retourne les tokens et le masque.


class ClipGlobalEncoder(_EncodeurGele):  # Fournit la référence CLIP pré-entraînée.
    """Référence CLIP pré-entraînée, utilisée telle quelle pour la comparaison."""

    def __init__(self, nom_modele: str) -> None:  # Charge le modèle CLIP.
        """Charge le modèle CLIP et son processeur depuis Hugging Face."""
        super().__init__()  # Initialise le module parent.
        from transformers import AutoProcessor, CLIPModel  # Importe transformers seulement lorsque c'est nécessaire.

        self.nom_modele = nom_modele  # Conserve le nom du modèle.
        self.processeur = AutoProcessor.from_pretrained(nom_modele)  # Charge le processeur officiel.
        self.modele = CLIPModel.from_pretrained(nom_modele, use_safetensors=True)  # Charge les poids au format safetensors.
        self._geler(self.modele)  # Gèle les poids et désactive le dropout.
        self.dim_sortie = int(self.modele.config.projection_dim)  # Lit la dimension de l'espace commun.

    @staticmethod  # Déclare une méthode sans état.
    def _tenseur_projete(sortie: object) -> torch.Tensor:  # Extrait l'embedding projeté d'une sortie.
        """Retourne l'embedding, que transformers renvoie un tenseur ou un objet de sortie."""
        if isinstance(sortie, torch.Tensor):  # Détecte un tenseur direct.
            return sortie  # Retourne le tenseur.
        return sortie.pooler_output  # Retourne l'embedding projeté de l'objet de sortie.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:  # Encode des images.
        """Retourne les embeddings globaux d'images normalisés."""
        entrees = self.processeur(images=list(images), return_tensors="pt")  # Applique le prétraitement officiel.
        return F.normalize(self._tenseur_projete(self.modele.get_image_features(pixel_values=entrees["pixel_values"])), dim=-1)  # Retourne les embeddings normalisés.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def encode_texts(self, textes: Sequence[str]) -> torch.Tensor:  # Encode des légendes.
        """Retourne les embeddings globaux de textes normalisés."""
        entrees = self.processeur(text=list(textes), return_tensors="pt", padding=True, truncation=True)  # Tokenise les légendes.
        return F.normalize(self._tenseur_projete(self.modele.get_text_features(input_ids=entrees["input_ids"], attention_mask=entrees["attention_mask"])), dim=-1)  # Retourne les embeddings normalisés.


class ExperimentalAudioEncoder(_EncodeurGele):  # Déclare l'extension audio.
    """Extension expérimentale : encode un vrai spectrogramme de forme (B, 128, T)."""

    STATUT = "experimental"  # Indique que cette modalité n'appartient pas au benchmark.

    def __init__(self, nom_modele: str) -> None:  # Mémorise le modèle sans le charger.
        """Mémorise le modèle ; il est chargé uniquement lors d'un véritable appel."""
        super().__init__()  # Initialise le module parent.
        self.nom_modele = nom_modele  # Conserve le nom du modèle.
        self.modele = None  # Reporte le chargement des poids.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def forward(self, spectrogrammes: torch.Tensor) -> torch.Tensor:  # Encode de vrais spectrogrammes.
        """Valide strictement la forme de l'entrée, puis exécute le modèle audio."""
        if spectrogrammes is None:  # Refuse l'absence d'entrée.
            raise ModalityUnavailableError("Audio absent : aucune caractéristique artificielle n'est générée.")  # Explique le refus.
        if spectrogrammes.dim() != 3 or spectrogrammes.size(1) != 128:  # Vérifie le contrat de forme.
            raise ValueError(f"Spectrogramme attendu de forme (B, 128, T), reçu {tuple(spectrogrammes.shape)}.")  # Explique l'erreur.
        if self.modele is None:  # Vérifie si le modèle reste à charger.
            from transformers import ASTModel  # Importe la classe seulement lorsque c'est nécessaire.

            self.modele = ASTModel.from_pretrained(self.nom_modele).to(spectrogrammes.device)  # Charge le modèle sur le bon device.
            self._geler(self.modele)  # Gèle les poids et désactive le dropout.
        return self.modele(spectrogrammes.transpose(1, 2)).last_hidden_state  # AST attend les dimensions (B, T, 128).


class ExperimentalVideoEncoder(_EncodeurGele):  # Déclare l'extension vidéo.
    """Extension expérimentale : encode de vraies images vidéo de forme (B, T, 3, H, L)."""

    STATUT = "experimental"  # Indique que cette modalité n'appartient pas au benchmark.

    def __init__(self, nom_modele: str) -> None:  # Mémorise le modèle sans le charger.
        """Mémorise le modèle ; il est chargé uniquement lors d'un véritable appel."""
        super().__init__()  # Initialise le module parent.
        self.nom_modele = nom_modele  # Conserve le nom du modèle.
        self.modele = None  # Reporte le chargement des poids.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def forward(self, images_video: torch.Tensor) -> torch.Tensor:  # Encode de vraies images vidéo.
        """Valide strictement la forme de l'entrée, puis exécute le modèle vidéo."""
        if images_video is None:  # Refuse l'absence d'entrée.
            raise ModalityUnavailableError("Vidéo absente : aucune caractéristique artificielle n'est générée.")  # Explique le refus.
        if images_video.dim() != 5 or images_video.size(2) != 3:  # Vérifie le contrat de forme.
            raise ValueError(f"Vidéo attendue de forme (B, T, 3, H, L), reçue {tuple(images_video.shape)}.")  # Explique l'erreur.
        if self.modele is None:  # Vérifie si le modèle reste à charger.
            from transformers import VideoMAEModel  # Importe la classe seulement lorsque c'est nécessaire.

            self.modele = VideoMAEModel.from_pretrained(self.nom_modele).to(images_video.device)  # Charge le modèle sur le bon device.
            self._geler(self.modele)  # Gèle les poids et désactive le dropout.
        return self.modele(pixel_values=images_video).last_hidden_state  # Retourne les tokens vidéo.


class ModalityProjector(nn.Module):  # Projette les tokens d'une modalité.
    """Projecteur appris P_m qui aligne chaque modalité sur la dimension latente commune."""

    def __init__(self, dim_entree: int, dim_latente: int, dropout: float = 0.1) -> None:  # Construit le projecteur.
        """Construit un projecteur à deux couches suivi d'une normalisation."""
        super().__init__()  # Initialise le module parent.
        self.reseau = nn.Sequential(  # Assemble les couches du projecteur.
            nn.Linear(dim_entree, dim_latente),  # Change la dimension des tokens.
            nn.LayerNorm(dim_latente),  # Stabilise l'échelle des activations.
            nn.GELU(),  # Ajoute la non-linéarité.
            nn.Dropout(dropout),  # Régularise le projecteur.
            nn.Linear(dim_latente, dim_latente),  # Affine la représentation.
            nn.LayerNorm(dim_latente),  # Normalise la sortie.
        )  # Termine l'assemblage.

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # Applique le projecteur.
        """Projette des tokens (B, T, D_m) vers (B, T, d)."""
        return self.reseau(tokens)  # Retourne les tokens projetés.


def pool_tokens(tokens: torch.Tensor, masque: torch.Tensor) -> torch.Tensor:  # Moyenne les tokens valides.
    """Calcule la moyenne des tokens en ignorant les positions de padding."""
    poids = masque.unsqueeze(-1).to(tokens.dtype)  # Convertit le masque en poids.
    return (tokens * poids).sum(dim=1) / poids.sum(dim=1).clamp_min(1.0)  # Retourne la moyenne pondérée.


def describe_feature_sources(noms: Dict[str, str]) -> List[Dict[str, str]]:  # Résume l'origine des caractéristiques.
    """Sépare explicitement les caractéristiques Meta, locales et propres à TopoFusion."""
    lignes = [{"modalite": modalite, "origine": feature_origin(nom)} for modalite, nom in noms.items()]  # Décrit les encodeurs gelés.
    lignes.append({"modalite": "commune", "origine": "Modules TopoFusion entraînés dans ce projet (projecteurs, banque, topologie)."})  # Décrit les modules du projet.
    lignes.append({"modalite": "Muse Spark", "origine": "Non utilisé dans les résultats : l'adaptateur exige une clé et n'expose pas d'embeddings."})  # Décrit le statut de Muse Spark.
    return lignes  # Retourne la description complète.
