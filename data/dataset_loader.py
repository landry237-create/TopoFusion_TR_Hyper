# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Chargement des vraies paires image-texte du benchmark.

Le dépôt Hugging Face fournit uniquement des fichiers CSV : nom de fichier COCO,
étiquettes visuelles et légende. Les images sont donc résolues depuis COCO train2014.
Aucun repli synthétique n'existe : une donnée absente est exclue et comptée.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import hashlib  # Fournit le hachage utilisé pour le découpage déterministe.
import io  # Permet de décoder une image reçue en mémoire.
import json  # Écrit et relit le manifeste du découpage.
import re  # Valide le format des noms de fichiers COCO.
import time  # Espace les nouvelles tentatives de téléchargement.
import urllib.error  # Expose les erreurs réseau de la bibliothèque standard.
import urllib.request  # Télécharge les images sans dépendance supplémentaire.
from collections import Counter  # Compte les rejets par motif.
from concurrent.futures import ThreadPoolExecutor  # Parallélise les téléchargements.
from dataclasses import dataclass, field  # Définit des conteneurs de données simples.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple  # Importe les types des signatures.

from PIL import Image  # Décode et redimensionne les images.

MOTIF_COCO = re.compile(r"^COCO_train2014_(\d{12})\.jpg$")  # Décrit un nom de fichier COCO train2014 valide.
SPLITS = ("train", "val", "test")  # Fixe l'ordre officiel des trois sous-ensembles.


class DatasetError(RuntimeError):  # Définit l'erreur levée quand les données réelles manquent.
    """Signale l'impossibilité d'obtenir des données réelles."""


@dataclass  # Génère le constructeur du conteneur.
class CaptionTable:  # Regroupe les légendes valides et le bilan de validation.
    """Table des légendes réelles indexées par fichier image."""

    captions_by_image: Dict[str, List[str]]  # Associe chaque fichier image à ses légendes.
    report: Dict[str, Any] = field(default_factory=dict)  # Conserve le bilan de chargement.


def image_id_from_file(nom_fichier: str) -> int:  # Extrait l'identifiant COCO d'un nom de fichier.
    """Retourne l'identifiant numérique contenu dans un nom de fichier COCO."""
    resultat = MOTIF_COCO.match(nom_fichier)  # Applique le motif attendu.
    if resultat is None:  # Détecte un nom non conforme.
        raise DatasetError(f"Nom de fichier COCO invalide : {nom_fichier!r}")  # Nomme la valeur fautive.
    return int(resultat.group(1))  # Convertit l'identifiant en entier.


def _charger_depuis_hub(cfg_donnees: Dict[str, Any]) -> Tuple[Iterable[Sequence[Any]], Tuple[str, ...]]:  # Lit le dépôt Hugging Face.
    """Tente l'appel strict demandé, puis lit le CSV de référence si cet appel échoue."""
    from datasets import load_dataset  # Importe datasets seulement lorsque c'est nécessaire.

    depot = cfg_donnees["hf_repo"]  # Récupère le nom du dépôt.
    split = cfg_donnees["hf_split"]  # Récupère le nom du split.
    jeu = None  # Initialise le jeu chargé.
    if cfg_donnees.get("try_strict_call", True):  # Vérifie si l'appel strict doit être tenté.
        try:  # Protège l'appel strict.
            jeu = load_dataset(depot, split=split)  # Exécute exactement l'appel demandé.
            print(f"Appel strict load_dataset({depot!r}, split={split!r}) réussi.")  # Confirme le succès.
        except Exception as exc:  # Intercepte l'échec dû aux colonnes hétérogènes du dépôt.
            print(f"Appel strict load_dataset({depot!r}, split={split!r}) impossible : {type(exc).__name__}.")  # Explique l'échec.
            print("Cause connue : les fichiers du dépôt n'ont pas tous les mêmes colonnes.")  # Donne la cause identifiée.
    if jeu is None or MOTIF_COCO.match(str(jeu.column_names[0])) is None:  # Vérifie le résultat de l'appel strict.
        fichier = cfg_donnees["fallback_data_file"]  # Récupère le CSV de référence.
        print(f"Lecture du fichier de référence {fichier!r} du même dépôt.")  # Annonce la lecture du CSV.
        jeu = load_dataset(depot, data_files=fichier, split=split)  # Charge uniquement le CSV de référence.
    colonnes = tuple(str(nom) for nom in jeu.column_names)  # Récupère les noms de colonnes.
    if len(colonnes) != 3:  # Vérifie la structure attendue.
        raise DatasetError(f"Trois colonnes attendues (image, étiquettes, légende), reçues : {colonnes}")  # Explique l'écart.
    lignes = zip(jeu[colonnes[0]], jeu[colonnes[1]], jeu[colonnes[2]])  # Assemble les lignes sans copier chaque enregistrement.
    return lignes, colonnes  # Retourne les lignes et les noms de colonnes.


def load_caption_table(cfg: Dict[str, Any], source: Optional[Callable[[Dict[str, Any]], Tuple[Iterable[Sequence[Any]], Tuple[str, ...]]]] = None) -> CaptionTable:  # Charge et valide les légendes.
    """Charge les légendes réelles, récupère la ligne prise pour un en-tête et valide chaque ligne."""
    donnees = cfg["dataset"]  # Récupère la section des données.
    print(f"Chargement du jeu de données Hugging Face : {donnees['hf_repo']} (split={donnees['hf_split']})")  # Affiche le message de chargement demandé.
    lignes, colonnes = (source or _charger_depuis_hub)(donnees)  # Lit la source réelle ou la source injectée par un test.
    rejets: Counter = Counter()  # Prépare le comptage des rejets.
    par_image: Dict[str, List[str]] = {}  # Prépare le regroupement par image.
    max_mots = int(donnees["max_caption_words"])  # Récupère la longueur maximale tolérée.
    lues = 0  # Initialise le compteur de lignes lues.
    en_tete_recupere = MOTIF_COCO.match(colonnes[0]) is not None  # Détecte une première ligne de données prise pour un en-tête.

    def traiter(fichier: Any, _etiquettes: Any, legende: Any) -> None:  # Valide une ligne ; les étiquettes sont ignorées volontairement.
        """Valide une ligne et l'ajoute à la table si elle est correcte."""
        if not isinstance(fichier, str) or MOTIF_COCO.match(fichier) is None:  # Vérifie le nom de fichier.
            rejets["nom_fichier_invalide"] += 1  # Compte le rejet.
            return  # Abandonne cette ligne.
        if not isinstance(legende, str) or not legende.strip():  # Vérifie la présence d'une légende.
            rejets["legende_vide"] += 1  # Compte le rejet.
            return  # Abandonne cette ligne.
        texte = " ".join(legende.split())  # Normalise les espaces sans modifier les mots.
        if len(texte.split()) > max_mots:  # Vérifie la longueur de la légende.
            rejets["legende_trop_longue"] += 1  # Compte le rejet.
            return  # Abandonne cette ligne.
        existantes = par_image.setdefault(fichier, [])  # Récupère les légendes déjà connues pour cette image.
        if texte in existantes:  # Détecte un doublon strict pour la même image.
            rejets["doublon_image_legende"] += 1  # Compte le rejet.
            return  # Abandonne cette ligne.
        existantes.append(texte)  # Enregistre la légende valide.

    if en_tete_recupere:  # Vérifie si la première ligne a été prise pour un en-tête.
        traiter(*colonnes)  # Traite cet échantillon réel au lieu de le perdre.
        lues += 1  # Compte cette ligne récupérée.
    for ligne in lignes:  # Parcourt toutes les lignes du CSV.
        traiter(ligne[0], ligne[1], ligne[2])  # Valide la ligne courante.
        lues += 1  # Met à jour le compteur.
    if not par_image:  # Vérifie qu'au moins un échantillon réel existe.
        raise DatasetError("Aucune paire valide trouvée : aucun repli synthétique n'est autorisé.")  # Refuse de continuer sans données réelles.
    nb_legendes = sum(len(liste) for liste in par_image.values())  # Compte les légendes retenues.
    rapport = {  # Construit le bilan de chargement.
        "lignes_lues": lues,  # Nombre de lignes lues.
        "legendes_valides": nb_legendes,  # Nombre de légendes conservées.
        "images_distinctes": len(par_image),  # Nombre d'images distinctes.
        "rejets": dict(rejets),  # Détail des rejets par motif.
        "en_tete_recupere": en_tete_recupere,  # Indique si la première ligne a été récupérée.
        "colonne_etiquettes_utilisee": False,  # Rappelle que les étiquettes dérivées de l'image sont exclues.
    }  # Termine le bilan.
    print(f"Échantillons réels chargés : {nb_legendes} légendes pour {len(par_image)} images (lignes lues : {lues}, rejets : {dict(rejets)}).")  # Affiche le nombre réel d'échantillons.
    return CaptionTable(par_image, rapport)  # Retourne la table validée.


def _uniforme_stable(graine: int, nom_fichier: str) -> float:  # Associe un réel stable à un fichier image.
    """Retourne un réel de [0, 1[ qui dépend seulement de la graine et du fichier."""
    condensat = hashlib.sha256(f"{graine}:{nom_fichier}".encode("utf-8")).hexdigest()  # Calcule un hachage stable entre systèmes.
    return int(condensat[:16], 16) / float(16 ** 16)  # Convertit le préfixe en réel uniforme.


def assign_split(graine: int, nom_fichier: str, fractions: Sequence[float]) -> str:  # Attribue un sous-ensemble à une image.
    """Attribue une image à un sous-ensemble ; toutes ses légendes le suivent, ce qui évite la fuite."""
    valeur = _uniforme_stable(graine, nom_fichier)  # Calcule la position stable de l'image.
    if valeur < fractions[0]:  # Teste l'appartenance à l'entraînement.
        return "train"  # Retourne le sous-ensemble d'entraînement.
    if valeur < fractions[0] + fractions[1]:  # Teste l'appartenance à la validation.
        return "val"  # Retourne le sous-ensemble de validation.
    return "test"  # Retourne le sous-ensemble de test.


def select_candidates(table: CaptionTable, cfg: Dict[str, Any]) -> Dict[str, List[str]]:  # Sélectionne les images candidates.
    """Choisit de façon déterministe les images de chaque sous-ensemble, avec une réserve."""
    donnees = cfg["dataset"]  # Récupère la section des données.
    graine = int(donnees["split_seed"])  # Récupère la graine du découpage.
    fractions = donnees["split_fractions"]  # Récupère les fractions du découpage.
    seaux: Dict[str, List[Tuple[float, str]]] = {nom: [] for nom in SPLITS}  # Prépare un seau par sous-ensemble.
    for fichier in table.captions_by_image:  # Parcourt toutes les images valides.
        seaux[assign_split(graine, fichier, fractions)].append((_uniforme_stable(graine + 1, fichier), fichier))  # Range l'image avec une clé d'ordre stable.
    quotas = {"train": donnees["num_train_images"], "val": donnees["num_val_images"], "test": donnees["num_test_images"]}  # Regroupe les quotas demandés.
    candidats: Dict[str, List[str]] = {}  # Prépare le résultat.
    for nom in SPLITS:  # Traite chaque sous-ensemble.
        reserve = int(quotas[nom] * (1.0 + float(donnees["reserve_ratio"]))) + 1  # Ajoute une réserve pour remplacer les images indisponibles.
        ordonnes = sorted(seaux[nom])  # Trie selon la clé stable.
        if len(ordonnes) < quotas[nom]:  # Vérifie que le quota peut être atteint.
            raise DatasetError(f"Quota impossible pour {nom} : {quotas[nom]} demandées, {len(ordonnes)} disponibles.")  # Explique le manque.
        candidats[nom] = [fichier for _, fichier in ordonnes[:reserve]]  # Conserve les premiers candidats.
    return candidats  # Retourne les candidats de chaque sous-ensemble.


class ImageStore:  # Gère le cache local des images réelles.
    """Résout les noms de fichiers COCO vers des images réelles stockées localement."""

    def __init__(self, cfg: Dict[str, Any]) -> None:  # Initialise le cache d'images.
        """Prépare le dossier de cache et les paramètres de téléchargement."""
        donnees = cfg["dataset"]  # Récupère la section des données.
        self.cote = int(donnees["image_cache_size"])  # Fixe le petit côté des images stockées.
        self.dossier = Path(donnees["cache_dir"]) / f"coco_images_{self.cote}"  # Calcule le dossier du cache.
        self.dossier.mkdir(parents=True, exist_ok=True)  # Crée le dossier si nécessaire.
        self.gabarit = donnees["image_url_template"]  # Conserve le gabarit d'URL.
        self.racine_coco = Path(donnees["coco_root"]) if donnees.get("coco_root") else None  # Conserve un éventuel dossier COCO local.
        self.travailleurs = int(donnees["download_workers"])  # Fixe le nombre de téléchargements simultanés.

    def chemin(self, nom_fichier: str) -> Path:  # Calcule le chemin d'une image du cache.
        """Retourne le chemin local d'une image dans le cache."""
        return self.dossier / nom_fichier  # Assemble le chemin.

    def _lire_octets(self, nom_fichier: str) -> bytes:  # Lit les octets bruts d'une image.
        """Lit l'image dans le dossier COCO local ou la télécharge, avec nouvelles tentatives."""
        if self.racine_coco is not None and (self.racine_coco / nom_fichier).is_file():  # Vérifie la présence d'une copie locale.
            return (self.racine_coco / nom_fichier).read_bytes()  # Lit la copie locale.
        derniere: Optional[Exception] = None  # Mémorise la dernière erreur rencontrée.
        for essai in range(3):  # Autorise trois tentatives.
            try:  # Protège la requête réseau.
                requete = urllib.request.Request(self.gabarit.format(file=nom_fichier), headers={"User-Agent": "TopoFusion-TR-Hyper-data-loader"})  # Prépare la requête avec un client générique.
                with urllib.request.urlopen(requete, timeout=30) as reponse:  # Envoie la requête avec un délai maximal.
                    return reponse.read()  # Retourne les octets reçus.
            except (urllib.error.URLError, TimeoutError, OSError) as exc:  # Intercepte une erreur réseau.
                derniere = exc  # Mémorise l'erreur.
                time.sleep(1.5 * (essai + 1))  # Attend avant de réessayer.
        raise DatasetError(f"Image indisponible : {nom_fichier} ({derniere})")  # Signale l'échec définitif.

    def _preparer(self, nom_fichier: str) -> Tuple[str, Optional[str]]:  # Prépare une image du cache.
        """Télécharge, décode et valide une image ; retourne une erreur plutôt qu'une image factice."""
        cible = self.chemin(nom_fichier)  # Calcule le chemin de destination.
        if cible.is_file():  # Vérifie si l'image est déjà disponible.
            try:  # Protège la vérification de l'image existante.
                with Image.open(cible) as image:  # Ouvre le fichier existant.
                    image.verify()  # Contrôle l'intégrité du fichier.
                return nom_fichier, None  # Signale le succès.
            except Exception:  # Détecte un fichier du cache corrompu.
                cible.unlink(missing_ok=True)  # Supprime le fichier corrompu pour le récupérer.
        try:  # Protège le téléchargement et le décodage.
            octets = self._lire_octets(nom_fichier)  # Récupère les octets de l'image.
            with Image.open(io.BytesIO(octets)) as image:  # Décode l'image reçue.
                rgb = image.convert("RGB")  # Force l'espace de couleur attendu.
            if min(rgb.size) < 32:  # Refuse une image trop petite pour être informative.
                return nom_fichier, "image_trop_petite"  # Signale le rejet.
            echelle = self.cote / float(min(rgb.size))  # Calcule le facteur de réduction.
            if echelle < 1.0:  # Réduit seulement les images plus grandes que la cible.
                rgb = rgb.resize((round(rgb.width * echelle), round(rgb.height * echelle)), Image.Resampling.BICUBIC)  # Redimensionne pour économiser le disque.
            temporaire = cible.with_suffix(".part")  # Prépare une écriture atomique.
            rgb.save(temporaire, format="JPEG", quality=92)  # Écrit l'image réduite.
            temporaire.replace(cible)  # Publie le fichier une fois complet.
            return nom_fichier, None  # Signale le succès.
        except Exception as exc:  # Intercepte toute erreur liée à cet échantillon.
            return nom_fichier, f"{type(exc).__name__}: {exc}"  # Retourne l'erreur pour la comptabiliser.

    def ensure_all(self, fichiers: Sequence[str]) -> Dict[str, str]:  # Prépare toutes les images demandées.
        """Prépare toutes les images et retourne celles qui sont indisponibles, avec le motif."""
        echecs: Dict[str, str] = {}  # Prépare le bilan des échecs.
        debut = time.time()  # Mémorise l'heure de début.
        with ThreadPoolExecutor(max_workers=self.travailleurs) as executeur:  # Crée le groupe de téléchargement.
            for rang, (nom, erreur) in enumerate(executeur.map(self._preparer, fichiers), start=1):  # Parcourt les résultats dans l'ordre.
                if erreur is not None:  # Détecte un échec.
                    echecs[nom] = erreur  # Enregistre le motif.
                if rang % 500 == 0 or rang == len(fichiers):  # Limite la fréquence des messages.
                    print(f"Images prêtes : {rang}/{len(fichiers)} ({len(echecs)} échecs, {time.time() - debut:.0f} s)", flush=True)  # Affiche la progression.
        return echecs  # Retourne le bilan des échecs.

    def open(self, nom_fichier: str) -> Image.Image:  # Ouvre une image du cache.
        """Ouvre une image réelle du cache au format RGB."""
        with Image.open(self.chemin(nom_fichier)) as image:  # Ouvre le fichier du cache.
            return image.convert("RGB")  # Retourne une copie décodée.


def _selection_legendes(graine: int, fichier: str, legendes: Sequence[str], nombre: int) -> List[str]:  # Choisit les légendes d'une image.
    """Choisit de façon stable un sous-ensemble des légendes d'une image."""
    ordonnees = sorted(legendes, key=lambda texte: _uniforme_stable(graine, f"{fichier}|{texte}"))  # Trie selon une clé stable.
    return ordonnees[:nombre]  # Conserve le nombre demandé.


def build_manifest(table: CaptionTable, store: ImageStore, cfg: Dict[str, Any]) -> Dict[str, Any]:  # Construit le manifeste final.
    """Construit le manifeste du découpage en ne retenant que des images réellement disponibles."""
    donnees = cfg["dataset"]  # Récupère la section des données.
    candidats = select_candidates(table, cfg)  # Sélectionne les candidats de chaque sous-ensemble.
    quotas = {"train": donnees["num_train_images"], "val": donnees["num_val_images"], "test": donnees["num_test_images"]}  # Regroupe les quotas.
    tous = [fichier for nom in SPLITS for fichier in candidats[nom]]  # Aplatit la liste des candidats.
    print(f"Vérification de {len(tous)} images COCO candidates dans {store.dossier}")  # Annonce l'étape de résolution.
    echecs = store.ensure_all(tous)  # Prépare les images et récupère les échecs.
    graine = int(donnees["split_seed"])  # Récupère la graine du découpage.
    manifeste: Dict[str, Any] = {"splits": {}, "meta": {}}  # Prépare le manifeste.
    for nom in SPLITS:  # Traite chaque sous-ensemble.
        valides = [fichier for fichier in candidats[nom] if fichier not in echecs][: quotas[nom]]  # Conserve les premières images disponibles.
        if len(valides) < quotas[nom]:  # Vérifie que le quota est atteint.
            raise DatasetError(f"Quota non atteint pour {nom} : {len(valides)}/{quotas[nom]} images disponibles.")  # Explique le manque.
        nombre = int(donnees["train_captions_per_image"]) if nom == "train" else int(donnees["eval_captions_per_image"])  # Choisit le nombre de légendes par image.
        manifeste["splits"][nom] = [  # Construit les entrées du sous-ensemble.
            {  # Décrit une image et ses légendes.
                "file": fichier,  # Nom de fichier COCO d'origine.
                "image_id": image_id_from_file(fichier),  # Identifiant COCO numérique.
                "captions": _selection_legendes(graine, fichier, table.captions_by_image[fichier], nombre),  # Légendes réelles de cette image.
            }  # Termine l'entrée.
            for fichier in valides  # Parcourt les images retenues.
        ]  # Termine la liste du sous-ensemble.
    empreintes = {nom: hashlib.sha256(",".join(e["file"] for e in manifeste["splits"][nom]).encode()).hexdigest()[:16] for nom in SPLITS}  # Calcule l'empreinte de chaque liste.
    manifeste["meta"] = {  # Renseigne les métadonnées du manifeste.
        "source_legendes": donnees["hf_repo"],  # Dépôt d'origine des légendes.
        "source_images": donnees["image_url_template"],  # Origine des images.
        "strategie_decoupage": "hachage SHA-256 de l'image, toutes ses légendes dans le même sous-ensemble",  # Stratégie du découpage.
        "graine_decoupage": graine,  # Graine du découpage.
        "empreintes": empreintes,  # Empreintes de vérification.
        "images_indisponibles": len(echecs),  # Nombre d'images exclues.
        "motifs_indisponibles": dict(list(echecs.items())[:20]),  # Premiers motifs d'exclusion pour le diagnostic.
        "bilan_chargement": table.report,  # Bilan de validation des légendes.
        "comptes": {nom: {"images": len(manifeste["splits"][nom]), "legendes": sum(len(e["captions"]) for e in manifeste["splits"][nom])} for nom in SPLITS},  # Comptes réels.
    }  # Termine les métadonnées.
    verify_no_leakage(manifeste)  # Vérifie l'absence de fuite avant de retourner le manifeste.
    for nom in SPLITS:  # Parcourt chaque sous-ensemble.
        print(f"Sous-ensemble {nom} : {manifeste['meta']['comptes'][nom]['images']} images, {manifeste['meta']['comptes'][nom]['legendes']} légendes.")  # Affiche les comptes réels.
    return manifeste  # Retourne le manifeste complet.


def verify_no_leakage(manifeste: Dict[str, Any]) -> None:  # Contrôle l'étanchéité du découpage.
    """Lève une erreur si une image apparaît dans deux sous-ensembles."""
    ensembles = {nom: {entree["file"] for entree in manifeste["splits"][nom]} for nom in SPLITS}  # Construit l'ensemble des images par sous-ensemble.
    for gauche, droite in (("train", "val"), ("train", "test"), ("val", "test")):  # Parcourt les paires de sous-ensembles.
        commun = ensembles[gauche] & ensembles[droite]  # Calcule l'intersection.
        if commun:  # Détecte une fuite.
            raise DatasetError(f"Fuite de données : {len(commun)} images communes entre {gauche} et {droite}.")  # Refuse de continuer.


def manifest_path(cfg: Dict[str, Any]) -> Path:  # Calcule le chemin du manifeste.
    """Retourne le chemin du manifeste, qui dépend des paramètres du découpage."""
    donnees = cfg["dataset"]  # Récupère la section des données.
    cle = json.dumps([donnees["split_seed"], donnees["split_fractions"], donnees["num_train_images"], donnees["num_val_images"], donnees["num_test_images"], donnees["train_captions_per_image"], donnees["eval_captions_per_image"], donnees["max_caption_words"]])  # Rassemble les paramètres déterminants.
    return Path(donnees["cache_dir"]) / f"manifest_{hashlib.sha256(cle.encode()).hexdigest()[:10]}.json"  # Assemble le chemin.


def prepare_manifest(cfg: Dict[str, Any], forcer: bool = False) -> Dict[str, Any]:  # Prépare ou relit le manifeste.
    """Retourne le manifeste existant ou le construit à partir des données réelles."""
    chemin = manifest_path(cfg)  # Calcule le chemin du manifeste.
    if chemin.is_file() and not forcer:  # Vérifie si le manifeste existe déjà.
        manifeste = json.loads(chemin.read_text(encoding="utf-8"))  # Relit le manifeste.
        verify_no_leakage(manifeste)  # Revérifie l'absence de fuite.
        print(f"Manifeste réutilisé : {chemin} ({manifeste['meta']['comptes']})")  # Annonce la réutilisation.
        return manifeste  # Retourne le manifeste relu.
    table = load_caption_table(cfg)  # Charge les légendes réelles.
    store = ImageStore(cfg)  # Prépare le cache d'images.
    manifeste = build_manifest(table, store, cfg)  # Construit le manifeste.
    chemin.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier parent.
    chemin.write_text(json.dumps(manifeste, ensure_ascii=False, indent=1), encoding="utf-8")  # Écrit le manifeste.
    return manifeste  # Retourne le manifeste construit.


def relevance_from_manifest(entrees: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[int], Dict[str, List[int]]]:  # Aplatit les légendes d'un sous-ensemble.
    """Retourne les légendes aplaties, l'image source de chacune et les images associées à chaque texte."""
    legendes: List[str] = []  # Prépare la liste aplatie.
    sources: List[int] = []  # Prépare l'image source de chaque légende.
    images_par_texte: Dict[str, List[int]] = {}  # Prépare l'index des textes partagés.
    for rang, entree in enumerate(entrees):  # Parcourt les images du sous-ensemble.
        for legende in entree["captions"]:  # Parcourt les légendes de l'image.
            legendes.append(legende)  # Ajoute la légende.
            sources.append(rang)  # Ajoute l'image source.
            images_par_texte.setdefault(legende, []).append(rang)  # Enregistre le lien entre ce texte et cette image.
    return legendes, sources, images_par_texte  # Retourne les trois structures.
