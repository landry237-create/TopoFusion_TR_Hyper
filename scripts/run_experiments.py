# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Entraîne et évalue les variantes du protocole sur les mêmes données et les mêmes conditions."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import argparse  # Lit les arguments de la ligne de commande.
import json  # Écrit les résultats.
import sys  # Permet d'ajouter la racine du projet au chemin d'importation.
import time  # Mesure les durées.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Any, Dict  # Importe les types des signatures.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Rend les paquets du projet importables.

import numpy as np  # Fournit les tableaux de résultats.
import psutil  # Mesure la mémoire sur tous les systèmes, y compris Windows.
import torch  # Fournit les tenseurs.

from data.corruptions import list_scenarios  # Énumère les scénarios de corruption.
from data.dataset_loader import prepare_manifest  # Prépare le manifeste réel.
from data.feature_cache import FeatureExtractor, FeatureSplit  # Donne accès aux caractéristiques.
from evaluation.diagnostics import representation_diagnostics  # Importe les diagnostics.
from evaluation.embedding import encode_split  # Importe le calcul des embeddings.
from evaluation.robustness_eval import RobustnessEvaluator  # Importe l'évaluation sur le test.
from evaluation.topology_eval import structure_report  # Importe les mesures de structure.
from models.encoders import describe_feature_sources  # Décrit l'origine des caractéristiques.
from models.multimodal_framework import get_variant, modality_inputs  # Importe le modèle.
from training.trainer import Trainer  # Importe la boucle d'entraînement.
from utils.config import config_hash, load_config  # Importe la gestion de configuration.
from utils.metrics import retrieval_metrics  # Importe les métriques de recherche.
from utils.seed import set_global_seed  # Importe la gestion des graines.


def _pic_memoire_mo() -> float:  # Mesure le pic de mémoire.
    """Retourne le pic de mémoire du processus en mégaoctets, ou la mémoire courante si le pic n'est pas exposé."""
    memoire = psutil.Process().memory_info()  # Lit les compteurs de mémoire du processus.
    try:  # Protège l'accès au module propre à Unix.
        import resource  # Importe le module qui n'existe pas sous Windows.

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Convertit les kilo-octets de Linux en mégaoctets.
    except ImportError:  # Traite le cas de Windows.
        return getattr(memoire, "peak_wset", memoire.rss) / (1024.0 ** 2)  # Utilise le compteur de pic de Windows.


def _sans_tableaux(resultats: Dict[str, Any]) -> Dict[str, Any]:  # Retire les tableaux d'un résultat.
    """Retire les tableaux volumineux avant l'écriture du fichier JSON."""
    return {cle: valeur for cle, valeur in resultats.items() if cle not in ("par_requete", "z_image", "z_texte")}  # Filtre les clés.


@torch.no_grad()  # Désactive le calcul du gradient.
def measure_costs(modele: torch.nn.Module, test: FeatureSplit, appareil: torch.device) -> Dict[str, float]:  # Mesure les coûts.
    """Mesure le temps d'inférence et le coût du calcul topologique sur de vrais lots de test."""
    modele.eval()  # Désactive le dropout.
    n = min(128, test.nb_images)  # Fixe la taille de l'échantillon mesuré.
    premieres = np.asarray([liste[0] for liste in test.legendes_par_image[:n]], dtype=np.int64)  # Sélectionne la première légende de chaque image.
    lot = test.batch_from_indices(np.arange(n), premieres, appareil)  # Assemble le lot.
    debut = time.perf_counter()  # Démarre le chronomètre.
    sorties = modele(modality_inputs(lot), fusionner=False)  # Exécute le modèle sur les deux modalités.
    par_paire = (time.perf_counter() - debut) / n  # Calcule la durée par paire.
    rapport = {"inference_s_par_paire_image_texte": par_paire}  # Prépare le rapport.
    vue = sorties["views"].get("image")  # Récupère la vue structurelle réelle des images.
    if vue is not None and vue.filtration is not None:  # Vérifie la présence d'une filtration réelle.
        from models.topology import differentiable_persistence, topological_bias  # Importe les étapes à chronométrer.

        debut = time.perf_counter()  # Démarre le chronomètre.
        differentiable_persistence(vue.filtration)  # Calcule la persistance sur les vrais tokens.
        rapport["persistance_s_par_nuage"] = (time.perf_counter() - debut) / n  # Calcule la durée par nuage.
        debut = time.perf_counter()  # Démarre le chronomètre.
        topological_bias(vue.filtration)  # Calcule l'ultramétrique sur les vrais tokens.
        rapport["ultrametrique_s_par_nuage"] = (time.perf_counter() - debut) / n  # Calcule la durée par nuage.
    return rapport  # Retourne le rapport.


def run_one(cfg: Dict[str, Any], manifeste: Dict[str, Any], nom: str, graine: int, racine: Path, scenarios: Any, appareil: torch.device, reprise: bool) -> Dict[str, Any]:  # Exécute une expérience.
    """Entraîne une variante, puis l'évalue une seule fois sur le jeu de test."""
    dossier = racine / nom / f"graine_{graine}"  # Calcule le dossier de l'expérience.
    fichier = dossier / "resultats.json"  # Calcule le fichier de résultats.
    if fichier.is_file():  # Vérifie si l'expérience est déjà terminée.
        existants = json.loads(fichier.read_text(encoding="utf-8"))  # Relit les résultats existants.
        if existants.get("empreinte_config") == config_hash(cfg):  # Vérifie qu'ils proviennent de la même configuration.
            print(f"Déjà terminé : {fichier}", flush=True)  # Annonce la réutilisation.
            return existants  # Retourne les résultats existants.
        print(f"Résultats obsolètes ignorés (autre configuration) : {fichier}", flush=True)  # Refuse de réutiliser silencieusement d'anciens résultats.
    variante = get_variant(nom)  # Récupère la description de la variante.
    extracteur = FeatureExtractor(cfg, manifeste)  # Calcule les chemins du cache.
    train = FeatureSplit(extracteur.racine, manifeste["splits"]["train"], "train")  # Ouvre l'entraînement.
    val = FeatureSplit(extracteur.racine, manifeste["splits"]["val"], "val")  # Ouvre la validation.
    entraineur = Trainer(cfg, variante, train, val, dossier, graine, appareil)  # Prépare l'entraînement.
    evaluateur = RobustnessEvaluator(cfg, manifeste, scenarios, appareil)  # Prépare l'évaluation sur le test.
    avant_i, avant_t = encode_split(entraineur.modele, evaluateur.propre, int(cfg["evaluation"]["batch_size"]), appareil)  # Encode le test avant tout entraînement, pour les figures et le niveau de départ.
    metriques_avant = retrieval_metrics(avant_i, avant_t, evaluateur.pertinence, cfg["evaluation"]["recall_ks"])  # Mesure le modèle non entraîné.
    if reprise:  # Vérifie si la reprise est demandée.
        entraineur.resume()  # Reprend depuis le dernier point de contrôle.
    resume = entraineur.fit()  # Entraîne le modèle.
    print(f"Évaluation unique sur le test : {nom}, graine {graine}", flush=True)  # Annonce l'unique passage sur le test.
    modele = entraineur.modele  # Récupère le meilleur modèle selon la validation.
    resultats = evaluateur.evaluate_model(modele)  # Évalue le test, propre et corrompu.
    n = min(int(cfg["evaluation"]["topology_points"]), evaluateur.propre.nb_images)  # Limite la taille des nuages.
    premieres = np.asarray([liste[0] for liste in evaluateur.propre.legendes_par_image[:n]], dtype=np.int64)  # Sélectionne la première légende de chaque image.
    x_i = np.asarray(evaluateur.propre.images[:n], dtype=np.float32).mean(axis=1)  # Calcule la caractéristique d'entrée des images.
    jetons = np.asarray(evaluateur.propre.textes[premieres], dtype=np.float32)  # Lit les tokens de textes.
    masque = np.asarray(evaluateur.propre.masques_texte[premieres], dtype=np.float32)[..., None]  # Lit le masque de padding.
    x_t = (jetons * masque).sum(axis=1) / masque.sum(axis=1)  # Calcule la caractéristique d'entrée des textes, sans le padding.
    x_i = x_i / np.linalg.norm(x_i, axis=1, keepdims=True).clip(min=1e-9)  # Normalise les entrées des images.
    x_t = x_t / np.linalg.norm(x_t, axis=1, keepdims=True).clip(min=1e-9)  # Normalise les entrées des textes.
    structure = structure_report(resultats["z_image"][:n], resultats["z_texte"][premieres], x_i, x_t)  # Mesure la préservation de structure.
    sortie = _sans_tableaux(resultats)  # Retire les tableaux volumineux.
    sortie.update({"resume_entrainement": resume, "avant_entrainement": metriques_avant, "structure": structure, "diagnostics": representation_diagnostics(modele, evaluateur.propre, 512, appareil), "couts": measure_costs(modele, evaluateur.propre, appareil), "pic_memoire_mo": _pic_memoire_mo(), "empreinte_config": config_hash(cfg)})  # Ajoute les autres résultats.
    if variante.tokens_topo:  # Vérifie la présence de tokens topologiques.
        z_i_sans, z_t_sans = encode_split(modele, evaluateur.propre, int(cfg["evaluation"]["batch_size"]), appareil, desactiver_topo=True)  # Encode le test sans tokens topologiques.
        sortie["inference_sans_tokens_topo"] = retrieval_metrics(z_i_sans, z_t_sans, evaluateur.pertinence, cfg["evaluation"]["recall_ks"])  # Mesure leur contribution à l'inférence.
    np.savez_compressed(dossier / "par_requete.npz", **resultats["par_requete"])  # Enregistre les mesures par requête.
    np.savez_compressed(dossier / "embeddings_test.npz", z_image=resultats["z_image"].astype(np.float16), z_texte=resultats["z_texte"].astype(np.float16), avant_image=avant_i.astype(np.float16), avant_texte=avant_t.astype(np.float16))  # Enregistre les embeddings.
    fichier.write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")  # Écrit les résultats.
    (dossier / "last.pt").unlink(missing_ok=True)  # Supprime le point de reprise devenu inutile, pour économiser le disque.
    return sortie  # Retourne les résultats.


def run_clip(cfg: Dict[str, Any], manifeste: Dict[str, Any], racine: Path, scenarios: Any, appareil: torch.device) -> None:  # Évalue CLIP.
    """Évalue la référence CLIP pré-entraînée, sans entraînement, sur le même test."""
    dossier = racine / "CLIP"  # Calcule le dossier de CLIP.
    dossier.mkdir(parents=True, exist_ok=True)  # Crée le dossier.
    evaluateur = RobustnessEvaluator(cfg, manifeste, scenarios, appareil)  # Prépare l'évaluation.
    resultats = evaluateur.evaluate_clip()  # Évalue CLIP.
    if resultats is None:  # Vérifie la disponibilité de CLIP.
        print("CLIP indisponible : embeddings absents.", flush=True)  # Annonce l'absence de CLIP.
        return  # Abandonne cette étape.
    np.savez_compressed(dossier / "par_requete.npz", **resultats["par_requete"])  # Enregistre les mesures par requête.
    np.savez_compressed(dossier / "embeddings_test.npz", z_image=resultats["z_image"].astype(np.float16), z_texte=resultats["z_texte"].astype(np.float16))  # Enregistre les embeddings.
    sortie = _sans_tableaux(resultats)  # Retire les tableaux volumineux.
    sortie["modele"] = cfg["clip_baseline"]["model"]  # Indique le modèle utilisé.
    (dossier / "resultats.json").write_text(json.dumps(sortie, ensure_ascii=False, indent=1), encoding="utf-8")  # Écrit les résultats.
    print(f"CLIP test : {resultats['propre']['mean_recall_at_1']:.4f} (R@1), {resultats['propre']['mean_recall_at_5']:.4f} (R@5)", flush=True)  # Affiche le résultat.


def main() -> None:  # Définit le point d'entrée du script.
    """Exécute les expériences demandées sur la ligne de commande."""
    analyseur = argparse.ArgumentParser(description="Entraîne et évalue les variantes de TopoFusion-TR Hyper.")  # Crée l'analyseur d'arguments.
    analyseur.add_argument("--config", default="config.yaml", help="Chemin du fichier de configuration.")  # Déclare le fichier de configuration.
    analyseur.add_argument("--variantes", default=None, help="Variantes séparées par des virgules ; toutes par défaut.")  # Déclare les variantes.
    analyseur.add_argument("--graines", default=None, help="Graines séparées par des virgules ; celles de la configuration par défaut.")  # Déclare les graines.
    analyseur.add_argument("--threads", type=int, default=4, help="Nombre de fils de calcul PyTorch.")  # Déclare le nombre de fils.
    analyseur.add_argument("--sortie", default=None, help="Dossier de sortie des expériences.")  # Déclare le dossier de sortie.
    analyseur.add_argument("--epoques", type=int, default=None, help="Remplace le nombre d'époques, pour un essai rapide.")  # Déclare le nombre d'époques.
    analyseur.add_argument("--clip", action="store_true", help="Évalue aussi la référence CLIP.")  # Déclare l'évaluation de CLIP.
    analyseur.add_argument("--reprise", action="store_true", help="Reprend depuis le dernier point de contrôle.")  # Déclare la reprise.
    arguments = analyseur.parse_args()  # Lit les arguments.
    torch.set_num_threads(arguments.threads)  # Applique le nombre de fils.
    remplacements = {"training": {"epochs": arguments.epoques}} if arguments.epoques else None  # Prépare le remplacement éventuel.
    cfg = load_config(arguments.config, remplacements)  # Charge la configuration.
    set_global_seed(int(cfg["seed"]))  # Fixe la graine globale.
    racine = Path(arguments.sortie or Path(cfg["outputs"]["root"]) / "runs")  # Calcule le dossier de sortie.
    racine.mkdir(parents=True, exist_ok=True)  # Crée le dossier de sortie.
    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choisit le device.
    manifeste = prepare_manifest(cfg)  # Relit le manifeste réel.
    print(json.dumps(describe_feature_sources({"image": cfg["encoders"]["image"], "texte": cfg["encoders"]["text"]}), ensure_ascii=False), flush=True)  # Affiche l'origine des caractéristiques.
    scenarios = list_scenarios(cfg)  # Énumère les scénarios.
    variantes = arguments.variantes.split(",") if arguments.variantes else list(cfg["experiments"]["variants"]) + list(cfg["experiments"]["ablations"])  # Sélectionne les variantes.
    graines = [int(g) for g in arguments.graines.split(",")] if arguments.graines else list(cfg["experiments"]["seeds"])  # Sélectionne les graines.
    if arguments.clip:  # Vérifie si CLIP est demandé.
        run_clip(cfg, manifeste, racine, scenarios, appareil)  # Évalue CLIP.
    for nom in variantes:  # Parcourt les variantes.
        for graine in graines:  # Parcourt les graines.
            debut = time.time()  # Démarre le chronomètre.
            sortie = run_one(cfg, manifeste, nom, graine, racine, scenarios, appareil, arguments.reprise)  # Exécute l'expérience.
            print(f"RUN_OK {nom} graine={graine} test_R@1={sortie['propre']['mean_recall_at_1']:.4f} test_R@5={sortie['propre']['mean_recall_at_5']:.4f} durée={time.time() - debut:.0f}s", flush=True)  # Journalise le résultat.


if __name__ == "__main__":  # Vérifie que le fichier est exécuté directement.
    main()  # Lance le script.
