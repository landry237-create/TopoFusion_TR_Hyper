# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Génère toutes les figures à partir des résultats mesurés et du meilleur modèle B4."""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import argparse  # Lit les arguments de la ligne de commande.
import json  # Lit les résultats.
import sys  # Permet d'ajouter la racine du projet au chemin d'importation.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Any, Dict, List, Optional, Tuple  # Importe les types des signatures.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Rend les paquets du projet importables.

import numpy as np  # Effectue les calculs matriciels.
import torch  # Fournit les tenseurs.

from data.corruptions import list_scenarios  # Énumère les scénarios.
from data.dataset_loader import ImageStore, prepare_manifest  # Donne accès aux images réelles.
from data.feature_cache import FeatureExtractor, FeatureSplit, scenario_name  # Donne accès aux caractéristiques.
from evaluation.topology_eval import finite_points, rips_diagrams  # Importe les diagrammes exacts.
from models.multimodal_framework import TopoFusionModel, VARIANTS, get_variant  # Importe le modèle.
from models.topology import exact_diagrams, persistence_landscape, soft_betti_curve  # Importe les outils topologiques.
from utils import visualization as viz  # Importe les fonctions de tracé.
from utils.config import load_config  # Charge la configuration.
from utils.metrics import pairwise_euclidean  # Importe le calcul des distances.


def _historiques(racine: Path) -> Dict[str, List[Dict[str, Any]]]:  # Lit les historiques.
    """Lit les historiques d'entraînement disponibles, par variante."""
    resultats: Dict[str, List[Dict[str, Any]]] = {}  # Prépare le dictionnaire.
    for nom in VARIANTS:  # Parcourt les variantes.
        fichiers = sorted((racine / nom).glob("graine_*/historique.json"))  # Trouve les historiques de la variante.
        if fichiers:  # Vérifie la présence de fichiers.
            resultats[nom] = [json.loads(f.read_text(encoding="utf-8")) for f in fichiers]  # Lit les historiques.
    return resultats  # Retourne les historiques.


def _charger_b4(cfg: Dict[str, Any], racine: Path, graine: int, dims: Dict[str, int], appareil: torch.device) -> Optional[TopoFusionModel]:  # Charge le meilleur modèle.
    """Charge le meilleur modèle B4 selon la validation."""
    chemin = racine / "B4" / f"graine_{graine}" / "best.pt"  # Calcule le chemin du point de contrôle.
    if not chemin.is_file():  # Vérifie la présence du fichier.
        return None  # Signale l'absence du modèle.
    modele = TopoFusionModel(cfg, get_variant("B4"), dims).to(appareil)  # Construit le modèle.
    modele.load_state_dict(torch.load(chemin, map_location=appareil, weights_only=False)["modele"])  # Charge les poids.
    modele.eval()  # Désactive le dropout.
    return modele  # Retourne le modèle.


@torch.no_grad()  # Désactive le calcul du gradient.
def _figures_modele(modele: TopoFusionModel, test: FeatureSplit, extracteur: FeatureExtractor, cfg: Dict[str, Any], manifeste: Dict[str, Any], sortie: Path, appareil: torch.device) -> None:  # Trace les figures liées au modèle.
    """Trace les figures qui nécessitent le modèle : diagrammes, paysages, graphe, attention, stabilité."""
    n = 64  # Fixe le nombre d'échantillons examinés.
    premieres = np.asarray([liste[0] for liste in test.legendes_par_image[:n]], dtype=np.int64)  # Sélectionne la première légende de chaque image.
    lot = test.batch_from_indices(np.arange(n), premieres, appareil)  # Assemble un lot de test.
    vue_i = modele({"image": (lot["image_tokens"], None)}, fusionner=False)["views"]["image"]  # Exécute la voie structurelle des images.
    vue_t = modele({"text": (lot["text_tokens"], lot["text_mask"])}, fusionner=False)["views"]["text"]  # Exécute la voie structurelle des textes.
    f_i = vue_i.filtration[0].cpu().numpy().astype(np.float64)  # Lit la filtration d'une image.
    f_t = vue_t.filtration[0].cpu().numpy().astype(np.float64)  # Lit la filtration de sa légende.
    d_i = exact_diagrams(f_i)  # Calcule ses diagrammes exacts.
    d_t = exact_diagrams(f_t)  # Calcule ses diagrammes exacts.
    z_i = np.load(sortie / "runs" / "B4" / "graine_42" / "embeddings_test.npz")["z_image"].astype(np.float32)[:256]  # Lit les embeddings d'images.
    rips_i = rips_diagrams(pairwise_euclidean(z_i))  # Calcule les diagrammes de Rips exacts du nuage d'images.
    viz.diagrammes_persistance({"Nuage de 256 images de test (Rips)": (rips_i[0], rips_i[1]), "Tokens partagés d'une image": (d_i[0], d_i[1]), "Tokens partagés de sa légende": (d_t[0], d_t[1])}, sortie / "figures" / "06_diagrammes_persistance.png")  # Trace les diagrammes.
    grille_max = max(float(finite_points(d).max()) if finite_points(d).size else 0.0 for d in (d_i[0], d_i[1], d_t[0], d_t[1]))  # Calcule la borne commune de la grille.
    grille = torch.linspace(0.0, grille_max * 1.02, 300)  # Crée une grille commune et fine.
    paysage_i: Dict[int, np.ndarray] = {}  # Prépare les paysages de l'image.
    paysage_t: Dict[int, np.ndarray] = {}  # Prépare les paysages du texte.
    for dimension in (0, 1):  # Parcourt H0 et H1.
        for source, cible in ((d_i, paysage_i), (d_t, paysage_t)):  # Parcourt les deux modalités.
            points = torch.from_numpy(finite_points(source[dimension]).astype(np.float32))  # Convertit le diagramme exact.
            masque = torch.ones(1, max(points.shape[0], 1), dtype=torch.bool)  # Crée le masque.
            if points.shape[0] == 0:  # Traite un diagramme vide.
                points = torch.zeros(1, 2)  # Crée un point nul.
                masque = torch.zeros(1, 1, dtype=torch.bool)  # Invalide ce point.
            cible[dimension] = persistence_landscape(points[:, 0].unsqueeze(0), points[:, 1].unsqueeze(0), masque, grille, 2)[0].numpy()  # Calcule le paysage.
    viz.paysages(grille.numpy(), paysage_i, paysage_t, sortie / "figures" / "07_paysages_persistance.png")  # Trace les paysages.
    viz.graphe_latent(vue_i.adjacency[0].cpu().numpy(), f_i, sortie / "figures" / "08_graphe_latent.png")  # Trace le graphe latent.
    viz.attention_topologique(vue_i.bias[0].cpu().numpy(), vue_i.attention[0].mean(dim=0).cpu().numpy(), sortie / "figures" / "09_attention_topologique.png", int(cfg["model"]["num_shared_tokens"]))  # Trace l'attention.
    echelles = torch.linspace(0.0, float(f_i.max()) * 1.02, 200)  # Crée la grille des échelles.
    temperature = float(cfg["topology"]["soft_betti_temperature"])  # Lit la température.
    exacte: Dict[int, np.ndarray] = {}  # Prépare les courbes exactes.
    douce: Dict[int, np.ndarray] = {}  # Prépare les courbes douces.
    for dimension in (0, 1):  # Parcourt H0 et H1.
        d = d_i[dimension]  # Lit le diagramme exact.
        naissance = torch.from_numpy(d[:, 0].astype(np.float32)).unsqueeze(0)  # Lit les naissances.
        mort = np.where(np.isinf(d[:, 1]), 1e6, d[:, 1])  # Remplace l'infini par une grande valeur, uniquement pour cette évaluation de courbe.
        mort_t = torch.from_numpy(mort.astype(np.float32)).unsqueeze(0)  # Convertit les morts.
        masque = torch.ones_like(naissance, dtype=torch.bool)  # Crée le masque.
        douce[dimension] = soft_betti_curve(naissance, mort_t, masque, echelles, temperature)[0].numpy()  # Calcule la courbe douce.
        exacte[dimension] = np.asarray([float(((d[:, 0] <= e) & (d[:, 1] > e)).sum()) for e in echelles.numpy()])  # Calcule la courbe exacte.
    viz.betti_douce_vs_exacte(echelles.numpy(), exacte, douce, temperature, sortie / "figures" / "13_betti_douce_vs_exacte.png")  # Trace la comparaison.
    borne = modele.filtration.lipschitz_bound()["borne_stabilite"]  # Lit la borne de stabilité.
    deplacements: List[float] = []  # Prépare les déplacements.
    distances: Dict[str, List[float]] = {"H0": [], "H1": []}  # Prépare les distances.
    import gudhi  # Importe la distance de bottleneck.

    propres = vue_i.points.cpu().numpy()  # Lit les points propres.
    filtrations = vue_i.filtration.cpu().numpy().astype(np.float64)  # Lit les filtrations propres.
    for modalite, nom, severite in list_scenarios(cfg):  # Parcourt les scénarios.
        if modalite != "image":  # Limite cette figure aux corruptions d'image.
            continue  # Passe au scénario suivant.
        split = FeatureSplit(extracteur.racine, manifeste["splits"]["test"], "test", prefixe_image=f"test__{scenario_name(nom, severite)}")  # Ouvre le scénario.
        jetons = torch.from_numpy(np.asarray(split.images[:n], dtype=np.float32)).to(appareil)  # Lit les tokens corrompus.
        vue = modele({"image": (jetons, None)}, fusionner=False)["views"]["image"]  # Exécute la voie structurelle.
        points = vue.points.cpu().numpy()  # Lit les points corrompus.
        matrices = vue.filtration.cpu().numpy().astype(np.float64)  # Lit les filtrations corrompues.
        for rang in range(n):  # Parcourt les échantillons.
            deplacements.append(float(np.linalg.norm(propres[rang] - points[rang], axis=-1).max()))  # Calcule le déplacement maximal.
            a = exact_diagrams(filtrations[rang])  # Calcule le diagramme propre.
            b = exact_diagrams(matrices[rang])  # Calcule le diagramme corrompu.
            distances["H0"].append(float(gudhi.bottleneck_distance(finite_points(a[0]), finite_points(b[0]))))  # Calcule la distance de H0.
            distances["H1"].append(float(gudhi.bottleneck_distance(finite_points(a[1]), finite_points(b[1]))))  # Calcule la distance de H1.
    viz.stabilite(np.asarray(deplacements), {c: np.asarray(v) for c, v in distances.items()}, borne, sortie / "figures" / "12_stabilite_borne.png")  # Trace la vérification.


def main() -> None:  # Définit le point d'entrée du script.
    """Génère toutes les figures demandées."""
    analyseur = argparse.ArgumentParser(description="Génère les figures scientifiques.")  # Crée l'analyseur d'arguments.
    analyseur.add_argument("--config", default="config.yaml", help="Chemin du fichier de configuration.")  # Déclare le fichier de configuration.
    analyseur.add_argument("--sortie", default="outputs", help="Dossier racine des sorties.")  # Déclare le dossier de sortie.
    analyseur.add_argument("--threads", type=int, default=4, help="Nombre de fils de calcul.")  # Déclare le nombre de fils.
    arguments = analyseur.parse_args()  # Lit les arguments.
    torch.set_num_threads(arguments.threads)  # Applique le nombre de fils.
    cfg = load_config(arguments.config)  # Charge la configuration.
    manifeste = prepare_manifest(cfg)  # Relit le manifeste.
    sortie = Path(arguments.sortie)  # Convertit le dossier de sortie.
    racine = sortie / "runs"  # Calcule le dossier des expériences.
    figures = sortie / "figures"  # Calcule le dossier des figures.
    viz.appliquer_style()  # Applique le style commun.
    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choisit le device.
    store = ImageStore(cfg)  # Prépare l'accès aux images réelles.
    entrees = manifeste["splits"]["train"][:8]  # Sélectionne quelques paires d'entraînement.
    viz.echantillons_bruts([store.open(e["file"]) for e in entrees], [e["captions"][0] for e in entrees], figures / "01_echantillons_bruts.png")  # Trace les échantillons.
    historiques = _historiques(racine)  # Lit les historiques.
    if historiques:  # Vérifie la présence de résultats.
        viz.courbes_perte(historiques, figures / "02_courbes_perte.png")  # Trace les pertes.
        viz.courbes_recall(historiques, figures / "03_courbes_recall.png")  # Trace les rappels.
    fichier_b4 = racine / "B4" / "graine_42" / "embeddings_test.npz"  # Calcule le fichier des embeddings de B4.
    if fichier_b4.is_file():  # Vérifie la présence des embeddings.
        donnees = np.load(fichier_b4)  # Lit les embeddings.
        extracteur = FeatureExtractor(cfg, manifeste)  # Calcule les chemins du cache.
        test = FeatureSplit(extracteur.racine, manifeste["splits"]["test"], "test")  # Ouvre le test.
        n = 250  # Fixe le nombre d'images affichées.
        premieres = np.asarray([liste[0] for liste in test.legendes_par_image[:n]], dtype=np.int64)  # Sélectionne la première légende de chaque image.
        apres = (donnees["z_image"][:n].astype(np.float32), donnees["z_texte"][premieres].astype(np.float32))  # Lit les embeddings finaux.
        avant = (donnees["avant_image"][:n].astype(np.float32), donnees["avant_texte"][premieres].astype(np.float32))  # Lit les embeddings initiaux.
        viz.latent_avant_apres(avant, apres, figures / "04_latent_avant_apres.png")  # Trace l'espace latent.
        fichier_clip = racine / "CLIP" / "embeddings_test.npz"  # Calcule le fichier des embeddings de CLIP.
        if fichier_clip.is_file():  # Vérifie la présence de CLIP.
            clip = np.load(fichier_clip)  # Lit les embeddings de CLIP.
            viz.clip_vs_topofusion((clip["z_image"][:n].astype(np.float32), clip["z_texte"][premieres].astype(np.float32)), apres, figures / "05_clip_vs_topofusion.png")  # Trace la comparaison.
        dims = {"image": int(test.images.shape[-1]), "text": int(test.textes.shape[-1])}  # Lit les dimensions.
        modele = _charger_b4(cfg, racine, 42, dims, appareil)  # Charge le meilleur modèle.
        if modele is not None:  # Vérifie la présence du modèle.
            _figures_modele(modele, test, extracteur, cfg, manifeste, sortie, appareil)  # Trace les figures liées au modèle.
    fichier_resume = sortie / "resultats" / "resume.json"  # Calcule le fichier du résumé.
    if fichier_resume.is_file():  # Vérifie la présence du résumé.
        resume = json.loads(fichier_resume.read_text(encoding="utf-8"))  # Lit le résumé.
        v = resume["variantes"]  # Lit les variantes.
        ordre = [n for n in ("B0", "B1", "B2", "B3", "B4", "B4_sans_L_top", "B4_sans_tokens_topo", "B4_sans_attention_topo", "B4_sans_robustesse", "B4_sans_topologie") if n in v]  # Fixe l'ordre des modèles.
        valeurs: Dict[str, Dict[str, Tuple[float, float]]] = {n: {m: (v[n]["test"][m]["moyenne"] * 100.0, v[n]["test"][m]["ecart_type"] * 100.0) for m in ("mean_recall_at_1", "mean_recall_at_5", "mean_map")} for n in ordre}  # Assemble les valeurs.
        if resume.get("clip"):  # Vérifie la présence de CLIP.
            valeurs["CLIP"] = {m: (resume["clip"]["propre"][m] * 100.0, 0.0) for m in ("mean_recall_at_1", "mean_recall_at_5", "mean_map")}  # Ajoute CLIP.
        viz.ablation(valeurs, figures / "11_ablation.png", reference="B4")  # Trace l'ablation.
        scenarios = sorted(v[ordre[0]]["scenarios"])  # Liste les scénarios.
        robustesse: Dict[str, Dict[str, Dict[str, float]]] = {}  # Prépare les résultats de robustesse.
        for nom in ("B1", "B4"):  # Parcourt les modèles affichés.
            if nom in v:  # Vérifie la présence du modèle.
                robustesse[nom] = {s: {"r5": v[nom]["scenarios"][s]["mean_recall_at_5"]["moyenne"] * 100.0, "r5_std": v[nom]["scenarios"][s]["mean_recall_at_5"]["ecart_type"] * 100.0, "degradation": v[nom]["scenarios"][s]["degradation_mean_recall_at_5"]["moyenne"] * 100.0, "degradation_std": v[nom]["scenarios"][s]["degradation_mean_recall_at_5"]["ecart_type"] * 100.0} for s in scenarios}  # Assemble les valeurs.
        if resume.get("clip"):  # Vérifie la présence de CLIP.
            robustesse["CLIP"] = {s: {"r5": resume["clip"]["scenarios"][s]["mean_recall_at_5"] * 100.0, "r5_std": 0.0, "degradation": resume["clip"]["scenarios"][s]["degradation_mean_recall_at_5"] * 100.0, "degradation_std": 0.0} for s in scenarios}  # Ajoute CLIP.
        viz.robustesse(robustesse, scenarios, figures / "10_robustesse.png")  # Trace la robustesse.
    print("FIGURES_TERMINEES", flush=True)  # Signale la fin.


if __name__ == "__main__":  # Vérifie que le fichier est exécuté directement.
    main()  # Lance le script.
