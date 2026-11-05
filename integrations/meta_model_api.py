# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Adaptateur de la Meta Model API (Muse Spark), limité aux interfaces documentées.

État réel : cet adaptateur n'a jamais été exécuté contre l'API, faute de clé. Il est testé contre
un serveur local de substitution qui vérifie le format documenté, ce qui ne prouve pas l'accès réel.
Aucun résultat de ce dépôt ne provient de Muse Spark.

La documentation consultée le 5 novembre 2026 ne décrit ni point de terminaison d'embeddings ni
accès aux états cachés, et logprobs renvoie une erreur 400. L'adaptateur exploite donc la seule sortie
disponible : un texte décrivant l'image. Ce texte peut ensuite passer par un encodeur local.
Les caractéristiques obtenues sont dérivées d'un texte : elles ne sont pas des représentations internes.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import base64  # Encode les images en URL de données.
import hashlib  # Calcule la clé du cache local.
import io  # Sérialise une image en mémoire.
import json  # Lit et écrit les requêtes et le cache.
import os  # Lit la clé dans les variables d'environnement.
import random  # Ajoute la gigue recommandée entre les tentatives.
import time  # Attend entre les tentatives.
import urllib.error  # Expose les erreurs HTTP de la bibliothèque standard.
import urllib.request  # Envoie les requêtes sans dépendance supplémentaire.
from dataclasses import dataclass  # Définit le conteneur de réponse.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Any, Dict, Optional  # Importe les types des signatures.

from PIL import Image  # Déclare le type des images en entrée.

INVITE_DEFAUT = "Describe the visible content of this image in one factual English sentence."  # Fixe la consigne par défaut.
PROVENANCE = "meta_model_api_text_output"  # Identifie l'origine des textes produits.


class MetaApiUnavailable(RuntimeError):  # Définit l'erreur d'indisponibilité.
    """Signale que l'API ne peut pas être utilisée, par exemple faute de clé."""


class MetaApiError(RuntimeError):  # Définit l'erreur renvoyée par l'API.
    """Signale une réponse d'erreur de l'API."""


class EmbeddingsNotExposed(NotImplementedError):  # Définit l'erreur liée aux embeddings.
    """Signale que la documentation ne décrit aucun accès aux représentations internes."""


@dataclass(frozen=True)  # Rend le résultat immuable.
class VisualDescription:  # Regroupe une description et sa provenance.
    """Description textuelle d'une image, avec sa provenance explicite."""

    texte: str  # Texte renvoyé par le modèle.
    modele: str  # Modèle indiqué dans la réponse de l'API.
    provenance: str = PROVENANCE  # Origine de la description.
    depuis_cache: bool = False  # Indique si le résultat provient du cache local.


class MetaModelApiClient:  # Encapsule les appels à l'API.
    """Client minimal de /chat/completions, conforme à la documentation officielle."""

    def __init__(self, cfg_meta: Dict[str, Any], dossier_cache: Optional[Path] = None) -> None:  # Prépare le client.
        """Lit la clé dans l'environnement et refuse explicitement de fonctionner sans elle."""
        self.url = cfg_meta["base_url"].rstrip("/") + "/chat/completions"  # Construit l'URL documentée.
        self.modele = cfg_meta["model"]  # Conserve l'identifiant du modèle.
        self.max_tokens = int(cfg_meta["max_completion_tokens"])  # Conserve la longueur maximale de la réponse.
        self.max_essais = int(cfg_meta["max_retries"])  # Conserve le nombre maximal de tentatives.
        self.delai = float(cfg_meta["timeout_seconds"])  # Conserve le délai d'attente.
        self.cle = os.environ.get(cfg_meta["api_key_env"], "").strip()  # Lit la clé sans jamais la stocker sur le disque.
        if not self.cle:  # Vérifie la présence de la clé.
            raise MetaApiUnavailable(f"Meta Model API inactive : la variable {cfg_meta['api_key_env']} est absente. Aucun repli ni résultat simulé n'est utilisé.")  # Explique le refus.
        self.cache = Path(dossier_cache) if dossier_cache else None  # Conserve le dossier du cache.
        if self.cache is not None:  # Vérifie qu'un cache est demandé.
            self.cache.mkdir(parents=True, exist_ok=True)  # Crée le dossier du cache.

    @staticmethod  # Déclare une méthode sans état.
    def _url_donnees(image: Image.Image) -> str:  # Encode une image.
        """Encode une image en URL de données JPEG, format accepté par la documentation."""
        tampon = io.BytesIO()  # Prépare un tampon mémoire.
        image.convert("RGB").save(tampon, format="JPEG", quality=90)  # Écrit l'image en JPEG.
        return "data:image/jpeg;base64," + base64.b64encode(tampon.getvalue()).decode("ascii")  # Retourne l'URL de données.

    def _appeler(self, corps: Dict[str, Any]) -> Dict[str, Any]:  # Envoie une requête.
        """Envoie une requête et réessaie sur 429 et 504, avec recul exponentiel et gigue."""
        donnees = json.dumps(corps).encode("utf-8")  # Sérialise le corps de la requête.
        entetes = {"Authorization": f"Bearer {self.cle}", "Content-Type": "application/json"}  # Prépare les en-têtes documentés.
        for essai in range(self.max_essais):  # Parcourt les tentatives.
            requete = urllib.request.Request(self.url, data=donnees, headers=entetes, method="POST")  # Prépare la requête.
            try:  # Protège l'appel réseau.
                with urllib.request.urlopen(requete, timeout=self.delai) as reponse:  # Envoie la requête.
                    return json.loads(reponse.read().decode("utf-8"))  # Retourne la réponse décodée.
            except urllib.error.HTTPError as exc:  # Intercepte une erreur HTTP.
                if exc.code in (429, 504) and essai + 1 < self.max_essais:  # Vérifie si l'erreur est temporaire.
                    time.sleep((2.0 ** essai) + random.random())  # Attend avant de réessayer.
                    continue  # Passe à la tentative suivante.
                raise MetaApiError(f"Erreur HTTP {exc.code} de la Meta Model API : {exc.read().decode('utf-8', 'replace')[:300]}") from exc  # Remonte l'erreur sans exposer la clé.
            except urllib.error.URLError as exc:  # Intercepte une erreur réseau.
                raise MetaApiUnavailable(f"Meta Model API injoignable : {exc.reason}") from exc  # Remonte l'erreur réseau.
        raise MetaApiError("Nombre maximal de tentatives atteint.")  # Signale l'échec final.

    def describe_image(self, image: Image.Image, invite: str = INVITE_DEFAUT) -> VisualDescription:  # Décrit une image.
        """Demande une description textuelle au format de requête documenté."""
        url = self._url_donnees(image)  # Encode l'image.
        cle_cache = hashlib.sha256(f"{self.modele}|{invite}|{url}".encode("utf-8")).hexdigest()  # Calcule la clé du cache.
        chemin = self.cache / f"{cle_cache}.json" if self.cache is not None else None  # Calcule le chemin du cache.
        if chemin is not None and chemin.is_file():  # Vérifie la présence d'un résultat en cache.
            contenu = json.loads(chemin.read_text(encoding="utf-8"))  # Relit le résultat.
            return VisualDescription(contenu["texte"], contenu["modele"], contenu["provenance"], True)  # Retourne le résultat du cache.
        corps = {"model": self.modele, "max_completion_tokens": self.max_tokens, "messages": [{"role": "user", "content": [{"type": "text", "text": invite}, {"type": "image_url", "image_url": {"url": url}}]}]}  # Construit la requête documentée.
        reponse = self._appeler(corps)  # Envoie la requête.
        try:  # Protège la lecture de la réponse.
            texte = reponse["choices"][0]["message"]["content"]  # Lit le texte produit.
            modele = reponse.get("model", self.modele)  # Lit le modèle indiqué par l'API.
        except (KeyError, IndexError, TypeError) as exc:  # Intercepte une réponse inattendue.
            raise MetaApiError(f"Réponse inattendue de l'API : {str(reponse)[:300]}") from exc  # Explique l'erreur.
        if chemin is not None:  # Vérifie qu'un cache est demandé.
            chemin.write_text(json.dumps({"texte": texte, "modele": modele, "provenance": PROVENANCE}, ensure_ascii=False), encoding="utf-8")  # Écrit le résultat dans le cache.
        return VisualDescription(texte, modele, PROVENANCE, False)  # Retourne la description.

    def internal_embeddings(self, *_: Any, **__: Any) -> None:  # Rend explicite une capacité absente.
        """Refuse explicitement : l'API documentée n'expose ni embeddings ni états cachés."""
        raise EmbeddingsNotExposed("La documentation de la Meta Model API ne décrit aucun embedding : aucune dimension ni représentation interne n'est inventée ici.")  # Explique le refus.
