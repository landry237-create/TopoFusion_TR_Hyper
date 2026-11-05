# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Agrège les résultats sur les graines et compare les modèles sur les mêmes requêtes de test.

Toutes les valeurs du rapport proviennent des fichiers produits par les expériences :
aucune valeur n'est saisie à la main.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import argparse  # Lit les arguments de la ligne de commande.
import json  # Lit et écrit les résultats.
import sys  # Permet d'ajouter la racine du projet au chemin d'importation.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Any, Dict, List, Optional  # Importe les types des signatures.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Rend les paquets du projet importables.

import numpy as np  # Effectue les calculs statistiques.

from data.dataset_loader import prepare_manifest  # Relit le manifeste réel.
from models.multimodal_framework import VARIANTS  # Importe les variantes.
from utils.config import load_config  # Charge la configuration.
from utils.metrics import chance_level, paired_bootstrap_difference  # Importe les outils statistiques.

METRIQUES = ["mean_recall_at_1", "mean_recall_at_5", "mean_recall_at_10", "mean_map", "i2t_recall_at_1", "i2t_recall_at_5", "t2i_recall_at_1", "t2i_recall_at_5", "i2t_map", "t2i_map", "i2t_median_rank", "t2i_median_rank"]  # Liste les métriques résumées.
COMPARAISONS = [("B1", "B0"), ("B2", "B1"), ("B3", "B2"), ("B4", "B3"), ("B4", "B1"), ("B4", "B4_sans_topologie"), ("B4", "B4_sans_L_top"), ("B4", "B4_sans_tokens_topo"), ("B4", "B4_sans_attention_topo"), ("B4", "B4_sans_robustesse"), ("B4", "CLIP"), ("B1", "CLIP")]  # Liste les comparaisons appariées.


def _lire_runs(racine: Path, nom: str) -> List[Dict[str, Any]]:  # Lit les résultats d'une variante.
    """Lit tous les résultats disponibles d'une variante, une entrée par graine."""
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted((racine / nom).glob("graine_*/resultats.json"))]  # Lit chaque fichier.


def _moyenne_ecart(valeurs: List[float]) -> Dict[str, float]:  # Résume une série.
    """Retourne la moyenne, l'écart-type corrigé et le nombre de valeurs."""
    tableau = np.asarray(valeurs, dtype=np.float64)  # Convertit la série.
    return {"moyenne": float(tableau.mean()), "ecart_type": float(tableau.std(ddof=1)) if tableau.size > 1 else 0.0, "n": int(tableau.size)}  # Retourne le résumé.


def _par_requete_moyen(racine: Path, nom: str) -> Optional[Dict[str, np.ndarray]]:  # Lit les mesures par requête.
    """Moyenne les succès par requête sur les graines, pour comparer des modèles sur les mêmes requêtes."""
    fichiers = sorted((racine / nom).glob("graine_*/par_requete.npz")) or sorted((racine / nom).glob("par_requete.npz"))  # Trouve les fichiers disponibles.
    if not fichiers:  # Vérifie la présence de fichiers.
        return None  # Signale l'absence de données.
    cumuls: Dict[str, List[np.ndarray]] = {}  # Prépare les accumulations.
    for fichier in fichiers:  # Parcourt les graines.
        donnees = np.load(fichier)  # Lit les mesures par requête.
        for sens in ("i2t", "t2i"):  # Parcourt les deux directions.
            rangs = donnees[f"rangs_{sens}"]  # Lit les rangs.
            cumuls.setdefault(f"r1_{sens}", []).append((rangs < 1).astype(np.float64))  # Enregistre les succès au rang 1.
            cumuls.setdefault(f"r5_{sens}", []).append((rangs < 5).astype(np.float64))  # Enregistre les succès au rang 5.
            cumuls.setdefault(f"ap_{sens}", []).append(donnees[f"ap_{sens}"].astype(np.float64))  # Enregistre la précision moyenne.
    return {cle: np.mean(np.stack(valeurs), axis=0) for cle, valeurs in cumuls.items()}  # Retourne les moyennes par requête.


def _resume_variante(runs: List[Dict[str, Any]]) -> Dict[str, Any]:  # Résume une variante.
    """Résume une variante sur les graines : recherche, robustesse, structure et coûts."""
    resume: Dict[str, Any] = {"nb_graines": len(runs), "test": {m: _moyenne_ecart([r["propre"][m] for r in runs]) for m in METRIQUES}}  # Résume le test.
    resume["avant_entrainement"] = {m: _moyenne_ecart([r["avant_entrainement"][m] for r in runs]) for m in ("mean_recall_at_1", "mean_recall_at_5", "mean_map")}  # Résume le modèle non entraîné.
    scenarios = sorted(runs[0]["scenarios"])  # Liste les scénarios.
    resume["scenarios"] = {s: {c: _moyenne_ecart([r["scenarios"][s][c] for r in runs]) for c in runs[0]["scenarios"][s]} for s in scenarios}  # Résume les scénarios.
    resume["structure"] = {c: _moyenne_ecart([r["structure"][c] for r in runs]) for c in runs[0]["structure"]}  # Résume la structure.
    resume["couts"] = {c: _moyenne_ecart([r["couts"][c] for r in runs]) for c in runs[0]["couts"]}  # Résume les coûts.
    resume["duree_entrainement_s"] = _moyenne_ecart([r["resume_entrainement"]["duree_entrainement_s"] for r in runs])  # Résume les durées.
    resume["pic_memoire_mo"] = _moyenne_ecart([r["pic_memoire_mo"] for r in runs])  # Résume la mémoire.
    resume["parametres"] = runs[0]["resume_entrainement"]["entrainables"]  # Lit le nombre de paramètres.
    resume["etapes_invalides"] = sum(r["resume_entrainement"]["etapes_invalides"] for r in runs)  # Compte les étapes invalides.
    resume["meilleures_epoques"] = [r["resume_entrainement"]["meilleure_epoque"] for r in runs]  # Liste les meilleures époques.
    resume["porte_structurelle"] = _moyenne_ecart([r["diagnostics"]["porte_structurelle"] for r in runs])  # Résume la porte structurelle.
    resume["effondrement"] = any(r["diagnostics"]["image"]["effondre"] or r["diagnostics"]["texte"]["effondre"] for r in runs)  # Indique un effondrement éventuel.
    resume["rang_effectif_image"] = _moyenne_ecart([r["diagnostics"]["image"]["rang_effectif"] for r in runs])  # Résume la dimension effective des images.
    resume["rang_effectif_texte"] = _moyenne_ecart([r["diagnostics"]["texte"]["rang_effectif"] for r in runs])  # Résume la dimension effective des textes.
    resume["marge_alignement"] = _moyenne_ecart([r["diagnostics"]["alignement"]["marge_alignement"] for r in runs])  # Résume l'alignement inter-modal.
    if all("inference_sans_tokens_topo" in r for r in runs):  # Vérifie la présence de l'ablation à l'inférence.
        resume["inference_sans_tokens_topo"] = {m: _moyenne_ecart([r["inference_sans_tokens_topo"][m] for r in runs]) for m in ("mean_recall_at_1", "mean_recall_at_5", "mean_map")}  # Résume l'ablation à l'inférence.
    if all(r.get("modalite_manquante") for r in runs):  # Vérifie la présence des modalités manquantes.
        resume["modalite_manquante"] = {c: _moyenne_ecart([r["modalite_manquante"][c] for r in runs]) for c in runs[0]["modalite_manquante"]}  # Résume les modalités manquantes.
    if all(r.get("stabilite_tokens") for r in runs):  # Vérifie la présence de la stabilité des tokens.
        resume["stabilite_tokens"] = {s: {c: _moyenne_ecart([r["stabilite_tokens"][s][c] for r in runs]) for c in runs[0]["stabilite_tokens"][s]} for s in runs[0]["stabilite_tokens"]}  # Résume la stabilité des tokens.
        resume["violations_borne_stabilite"] = sum(sum(r["stabilite_tokens"][s]["violations_de_la_borne"] for s in r["stabilite_tokens"]) for r in runs)  # Compte les violations de la borne.
    if all("partage_prive" in r["diagnostics"] for r in runs):  # Vérifie la présence de la séparation partagé/privé.
        resume["separation"] = {c: _moyenne_ecart([r["diagnostics"]["partage_prive"][c] for r in runs]) for c in runs[0]["diagnostics"]["partage_prive"] if not c.startswith("effondrement")}  # Résume la séparation.
        resume["poids_fusion"] = {c: _moyenne_ecart([r["diagnostics"]["poids_fusion_moyens"][c] for r in runs]) for c in runs[0]["diagnostics"]["poids_fusion_moyens"]}  # Résume les poids de fusion.
    return resume  # Retourne le résumé.


def _comparer(racine: Path, a: str, b: str) -> Optional[Dict[str, Any]]:  # Compare deux modèles.
    """Compare deux modèles requête par requête, par bootstrap apparié."""
    pa = _par_requete_moyen(racine, a)  # Lit les mesures du premier modèle.
    pb = _par_requete_moyen(racine, b)  # Lit les mesures du second modèle.
    if pa is None or pb is None:  # Vérifie la disponibilité des deux modèles.
        return None  # Signale une comparaison impossible.
    return {nom: paired_bootstrap_difference(np.concatenate([pa[f"{nom}_i2t"], pa[f"{nom}_t2i"]]), np.concatenate([pb[f"{nom}_i2t"], pb[f"{nom}_t2i"]])) for nom in ("r1", "r5", "ap")}  # Compare les trois mesures.


def _cellule(entree: Dict[str, float], facteur: float = 100.0, decimales: int = 2) -> str:  # Formate une cellule.
    """Formate une valeur moyenne ± écart-type pour un tableau Markdown."""
    return f"{entree['moyenne'] * facteur:.{decimales}f} ± {entree['ecart_type'] * facteur:.{decimales}f}"  # Retourne le texte formaté.


def _table(entetes: List[str], lignes: List[List[str]]) -> str:  # Construit un tableau Markdown.
    """Construit un tableau Markdown à partir d'en-têtes et de lignes."""
    return "\n".join(["| " + " | ".join(entetes) + " |", "|" + "---|" * len(entetes)] + ["| " + " | ".join(ligne) + " |" for ligne in lignes])  # Retourne le tableau.


def write_report(resultats: Dict[str, Any], chemin: Path) -> None:  # Écrit le rapport.
    """Écrit le rapport des résultats en Markdown à partir des seules données mesurées."""
    v = resultats["variantes"]  # Lit le résumé des variantes.
    clip = resultats.get("clip")  # Lit le résumé de CLIP.
    ordre = [n for n in ("B0", "B1", "B2", "B3", "B4", "B4_sans_L_top", "B4_sans_tokens_topo", "B4_sans_attention_topo", "B4_sans_robustesse", "B4_sans_topologie") if n in v]  # Fixe l'ordre des variantes.
    texte: List[str] = ["# Résultats expérimentaux", "", "Ce fichier est généré par `scripts/aggregate_results.py`. Chaque valeur provient des fichiers de `outputs/runs/`.", ""]  # Démarre le rapport.
    meta = resultats["protocole"]  # Lit la description du protocole.
    texte += ["## Protocole", "", f"- Test : {meta['images_test']} images et {meta['legendes_test']} légendes, jamais utilisées pour l'entraînement ou la sélection.", f"- Entraînement : {meta['images_train']} images ; validation : {meta['images_val']} images.", f"- Niveau du hasard : R@1 image→texte {meta['hasard']['r1_i2t'] * 100:.2f} %, texte→image {meta['hasard']['r1_t2i'] * 100:.2f} % ; R@5 image→texte {meta['hasard']['r5_i2t'] * 100:.2f} %, texte→image {meta['hasard']['r5_t2i'] * 100:.2f} %.", "- Les valeurs sont données en %, sous la forme moyenne ± écart-type sur les graines.", ""]  # Décrit le protocole.
    texte += ["## Recherche inter-modale sur le test", ""]  # Ajoute le titre de la section.
    lignes = [[n, VARIANTS[n].description, str(v[n]["nb_graines"]), _cellule(v[n]["test"]["mean_recall_at_1"]), _cellule(v[n]["test"]["mean_recall_at_5"]), _cellule(v[n]["test"]["mean_map"]), _cellule(v[n]["test"]["i2t_recall_at_1"]), _cellule(v[n]["test"]["t2i_recall_at_1"])] for n in ordre]  # Construit les lignes.
    if clip:  # Vérifie la présence de CLIP.
        p = clip["propre"]  # Lit les métriques de CLIP.
        lignes.append(["CLIP", f"Référence pré-entraînée, sans entraînement : {clip['modele']}", "-", f"{p['mean_recall_at_1'] * 100:.2f}", f"{p['mean_recall_at_5'] * 100:.2f}", f"{p['mean_map'] * 100:.2f}", f"{p['i2t_recall_at_1'] * 100:.2f}", f"{p['t2i_recall_at_1'] * 100:.2f}"])  # Ajoute la ligne de CLIP.
    texte += [_table(["Modèle", "Description", "Graines", "R@1 moyen", "R@5 moyen", "mAP moyen", "R@1 i→t", "R@1 t→i"], lignes), ""]  # Ajoute le tableau.
    texte += ["## Comparaisons appariées", "", "Différence (A − B) en points de pourcentage, avec IC 95 % par bootstrap apparié sur les requêtes de test, sur les succès moyennés par graine. La mention « oui » indique un intervalle qui exclut 0. Cet intervalle reflète la variabilité des requêtes, pas celle des graines : à lire avec les écarts-types ci-dessus.", ""]  # Présente les comparaisons.
    lignes = []  # Réinitialise les lignes.
    for nom, comparaison in resultats["comparaisons"].items():  # Parcourt les comparaisons.
        if comparaison is not None:  # Vérifie que la comparaison existe.
            lignes.append([nom] + [f"{comparaison[m]['difference'] * 100:+.2f} [{comparaison[m]['ic95_bas'] * 100:+.2f}, {comparaison[m]['ic95_haut'] * 100:+.2f}] {'oui' if comparaison[m]['significatif'] else 'non'}" for m in ("r1", "r5", "ap")])  # Ajoute la ligne.
    texte += [_table(["A − B", "ΔR@1", "ΔR@5", "ΔAP"], lignes), ""]  # Ajoute le tableau.
    if "B4" in v and "inference_sans_tokens_topo" in v["B4"]:  # Vérifie la présence de l'ablation à l'inférence.
        i = v["B4"]["inference_sans_tokens_topo"]  # Lit l'ablation à l'inférence.
        texte += ["## Contribution des tokens topologiques à l'inférence (B4)", "", f"Remplacer les tokens H0 et H1 par des zéros sur le modèle B4 déjà entraîné donne R@1 = {_cellule(i['mean_recall_at_1'])}, R@5 = {_cellule(i['mean_recall_at_5'])}, mAP = {_cellule(i['mean_map'])}. Porte structurelle apprise : {v['B4']['porte_structurelle']['moyenne']:.3f}.", ""]  # Décrit l'ablation.
    texte += ["## Robustesse aux corruptions (test)", "", "R@5 moyen absolu en %, puis dégradation relative en % entre parenthèses.", ""]  # Présente la robustesse.
    scenarios = sorted(v[ordre[0]]["scenarios"]) if ordre else []  # Liste les scénarios.
    colonnes = [n for n in ("B1", "B3", "B4", "B4_sans_robustesse", "B4_sans_topologie") if n in v]  # Choisit les variantes affichées.
    lignes = []  # Réinitialise les lignes.
    for s in scenarios:  # Parcourt les scénarios.
        ligne = [s]  # Démarre la ligne.
        for n in colonnes:  # Parcourt les variantes.
            e = v[n]["scenarios"][s]  # Lit le résumé du scénario.
            ligne.append(f"{e['mean_recall_at_5']['moyenne'] * 100:.2f} ({e['degradation_mean_recall_at_5']['moyenne'] * 100:.1f})")  # Ajoute la cellule.
        if clip:  # Vérifie la présence de CLIP.
            e = clip["scenarios"][s]  # Lit le résumé de CLIP.
            ligne.append(f"{e['mean_recall_at_5'] * 100:.2f} ({e['degradation_mean_recall_at_5'] * 100:.1f})")  # Ajoute la cellule.
        lignes.append(ligne)  # Ajoute la ligne.
    texte += [_table(["Scénario"] + colonnes + (["CLIP"] if clip else []), lignes), ""]  # Ajoute le tableau.
    if "B4" in v and "modalite_manquante" in v["B4"]:  # Vérifie la présence des modalités manquantes.
        m = v["B4"]["modalite_manquante"]  # Lit les résultats.
        texte += ["## Modalités manquantes (B4)", "", f"Avec l'image seule, retrouver l'embedding fusionné complet donne R@1 = {_cellule(m['recall_at_1_image_seule_vers_complet'])} (cosinus {m['cosinus_complet_vs_image_seule']['moyenne']:.3f}). Avec le texte seul : R@1 = {_cellule(m['recall_at_1_texte_seul_vers_complet'])} (cosinus {m['cosinus_complet_vs_texte_seul']['moyenne']:.3f}). Aucune donnée n'est simulée : la modalité absente est exclue de la fusion.", ""]  # Décrit le résultat.
    texte += ["## Structure géométrique et topologique (exacte, 256 images de test)", "", "Plus la corrélation de Mantel est élevée, et plus le stress et les distances entre diagrammes sont faibles, mieux la structure est préservée.", ""]  # Présente la structure.
    cles = [("mantel_entree_latent_image", "Mantel entrée-latent image"), ("mantel_entree_latent_texte", "Mantel entrée-latent texte"), ("mantel_image_texte", "Mantel image-texte"), ("stress_entree_latent_image", "Stress image"), ("stress_entree_latent_texte", "Stress texte"), ("voisinage10_entree_latent_image", "Voisins@10 image"), ("voisinage10_entree_latent_texte", "Voisins@10 texte"), ("h0_wasserstein1_image_texte_latent", "W1 H0 image-texte"), ("h1_wasserstein1_image_texte_latent", "W1 H1 image-texte"), ("h1_bottleneck_image_texte_latent", "Bottleneck H1 image-texte")]  # Liste les mesures affichées.
    lignes = [[etiquette] + [_cellule(v[n]["structure"][cle], 1.0, 3) for n in ordre] for cle, etiquette in cles]  # Construit les lignes.
    texte += [_table(["Mesure"] + ordre, lignes), ""]  # Ajoute le tableau.
    texte += ["## Diagnostics des représentations", ""]  # Ajoute le titre de la section.
    lignes = [[n, f"{v[n]['rang_effectif_image']['moyenne']:.1f}", f"{v[n]['rang_effectif_texte']['moyenne']:.1f}", f"{v[n]['marge_alignement']['moyenne']:.3f}", f"{v[n]['porte_structurelle']['moyenne']:.3f}", "oui" if v[n]["effondrement"] else "non", str(v[n]["etapes_invalides"])] for n in ordre]  # Construit les lignes.
    texte += [_table(["Modèle", "Rang effectif image", "Rang effectif texte", "Marge cosinus vraies-fausses paires", "Porte structurelle", "Effondrement détecté", "Étapes NaN ignorées"], lignes), ""]  # Ajoute le tableau.
    if "B4" in v and "separation" in v["B4"]:  # Vérifie la présence de la séparation.
        s = v["B4"]["separation"]  # Lit la séparation.
        texte += [f"Séparation partagé/privé de B4 : alignement inter-modal des tokens partagés {s['alignement_partage_inter_modal']['moyenne']:.3f}, des tokens privés {s['alignement_prive_inter_modal']['moyenne']:.3f} ; recouvrement |cos| partagé-privé {s['cosinus_absolu_partage_prive_image']['moyenne']:.3f} (image), {s['cosinus_absolu_partage_prive_text']['moyenne']:.3f} (texte). Poids de fusion moyens : {', '.join(f'{c} {e['moyenne']:.3f}' for c, e in v['B4']['poids_fusion'].items())}.", ""]  # Décrit la séparation.
    if "B4" in v and "stabilite_tokens" in v["B4"]:  # Vérifie la présence de la stabilité des tokens.
        texte += ["## Vérification de la borne de stabilité (B4)", "", f"Violations de la borne d_B ≤ (L_phi + L_w) · max ||u_i − u'_i||, sur tous les scénarios et toutes les graines : **{v['B4']['violations_borne_stabilite']}**. Cette borne porte sur les points d'entrée de la filtration.", ""]  # Présente la vérification.
        lignes = [[s, f"{e['deplacement_moyen']['moyenne']:.3f}", f"{e['bottleneck_h0_moyen']['moyenne']:.4f}", f"{e['bottleneck_h1_moyen']['moyenne']:.4f}", f"{e['rapport_maximal_observe']['moyenne']:.3f}", f"{e['borne_theorique']['moyenne']:.3f}"] for s, e in v["B4"]["stabilite_tokens"].items()]  # Construit les lignes.
        texte += [_table(["Scénario", "Déplacement moyen", "d_B H0", "d_B H1", "Rapport maximal observé", "Borne théorique"], lignes), ""]  # Ajoute le tableau.
    texte += ["## Coûts mesurés sur CPU", ""]  # Ajoute le titre de la section.
    lignes = []  # Réinitialise les lignes.
    for n in ordre:  # Parcourt les variantes.
        c = v[n]["couts"]  # Lit les coûts.
        lignes.append([n, f"{v[n]['parametres'] / 1e6:.2f}", f"{v[n]['duree_entrainement_s']['moyenne'] / 60:.1f}", f"{c['inference_s_par_paire_image_texte']['moyenne'] * 1000:.2f}", f"{c['persistance_s_par_nuage']['moyenne'] * 1000:.3f}" if "persistance_s_par_nuage" in c else "-", f"{c['ultrametrique_s_par_nuage']['moyenne'] * 1000:.3f}" if "ultrametrique_s_par_nuage" in c else "-", f"{v[n]['pic_memoire_mo']['moyenne']:.0f}"])  # Ajoute la ligne.
    texte += [_table(["Modèle", "Paramètres (M)", "Entraînement (min)", "Inférence (ms/paire)", "Persistance (ms/nuage)", "Ultramétrique (ms/nuage)", "Pic mémoire (Mo)"], lignes), "", "L'inférence porte sur les têtes entraînables à partir des caractéristiques en cache, sans le coût des encodeurs gelés. Les durées dépendent de la charge de la machine pendant la mesure, car plusieurs expériences tournaient en parallèle.", ""]  # Ajoute le tableau.
    chemin.write_text("\n".join(texte), encoding="utf-8")  # Écrit le rapport.


def main() -> None:  # Définit le point d'entrée du script.
    """Agrège les résultats et écrit le résumé JSON et le rapport Markdown."""
    analyseur = argparse.ArgumentParser(description="Agrège les résultats des expériences.")  # Crée l'analyseur d'arguments.
    analyseur.add_argument("--config", default="config.yaml", help="Chemin du fichier de configuration.")  # Déclare le fichier de configuration.
    analyseur.add_argument("--runs", default="outputs/runs", help="Dossier des expériences.")  # Déclare le dossier des expériences.
    analyseur.add_argument("--sortie", default="outputs/resultats", help="Dossier de sortie.")  # Déclare le dossier de sortie.
    analyseur.add_argument("--rapport", default="docs/RESULTATS.md", help="Chemin du rapport Markdown.")  # Déclare le rapport.
    arguments = analyseur.parse_args()  # Lit les arguments.
    racine = Path(arguments.runs)  # Convertit le dossier des expériences.
    sortie = Path(arguments.sortie)  # Convertit le dossier de sortie.
    sortie.mkdir(parents=True, exist_ok=True)  # Crée le dossier de sortie.
    cfg = load_config(arguments.config)  # Charge la configuration.
    manifeste = prepare_manifest(cfg)  # Relit le manifeste.
    comptes = manifeste["meta"]["comptes"]  # Lit les comptes réels.
    from data.feature_cache import FeatureExtractor, FeatureSplit  # Importe l'accès au cache.

    test = FeatureSplit(FeatureExtractor(cfg, manifeste).racine, manifeste["splits"]["test"], "test")  # Ouvre le test.
    pertinence = test.relevance_matrix()  # Construit la matrice de pertinence.
    h1 = chance_level(pertinence, 1)  # Calcule le hasard au rang 1.
    h5 = chance_level(pertinence, 5)  # Calcule le hasard au rang 5.
    variantes = {n: _resume_variante(runs) for n in VARIANTS if (runs := _lire_runs(racine, n))}  # Résume les variantes disponibles.
    fichier_clip = racine / "CLIP" / "resultats.json"  # Calcule le fichier de CLIP.
    resultats = {  # Construit le résumé.
        "protocole": {"images_train": comptes["train"]["images"], "images_val": comptes["val"]["images"], "images_test": comptes["test"]["images"], "legendes_test": comptes["test"]["legendes"], "empreintes_decoupage": manifeste["meta"]["empreintes"], "hasard": {"r1_i2t": h1["i2t"], "r1_t2i": h1["t2i"], "r5_i2t": h5["i2t"], "r5_t2i": h5["t2i"]}},  # Décrit le protocole.
        "variantes": variantes,  # Ajoute le résumé des variantes.
        "clip": json.loads(fichier_clip.read_text(encoding="utf-8")) if fichier_clip.is_file() else None,  # Ajoute CLIP.
        "comparaisons": {f"{a} − {b}": _comparer(racine, a, b) for a, b in COMPARAISONS},  # Ajoute les comparaisons appariées.
    }  # Termine le résumé.
    (sortie / "resume.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=1), encoding="utf-8")  # Écrit le résumé.
    write_report(resultats, Path(arguments.rapport))  # Écrit le rapport.
    print(f"Résumé écrit dans {sortie / 'resume.json'} et rapport écrit dans {arguments.rapport}.", flush=True)  # Annonce la fin.


if __name__ == "__main__":  # Vérifie que le fichier est exécuté directement.
    main()  # Lance le script.
