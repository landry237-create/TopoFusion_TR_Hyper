# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Évaluation sur le jeu de test : données propres, corruptions contrôlées et modalités manquantes.

Le mot « certifié » n'est jamais employé : ces mesures sont empiriques. Seule la borne sur les
diagrammes de tokens est démontrée, et elle ne garantit pas la performance de recherche.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Any, Dict, List, Optional, Sequence, Tuple  # Importe les types des signatures.

import numpy as np  # Effectue les calculs matriciels.
import torch  # Fournit les tenseurs.

from data.feature_cache import FeatureExtractor, FeatureSplit, scenario_name  # Importe l'accès au cache.
from evaluation.embedding import encode_split  # Importe le calcul des embeddings.
from evaluation.topology_eval import diagram_distances, rips_diagrams, token_cloud_filtrations, token_stability  # Importe les mesures topologiques.
from models.multimodal_framework import modality_inputs  # Importe la conversion des lots.
from utils.metrics import pairwise_euclidean, retrieval_metrics, retrieval_per_query, summarize_retrieval  # Importe les métriques.


def relative_degradation(propre: float, corrompu: float) -> float:  # Calcule la dégradation relative.
    """Retourne (propre - corrompu) / propre ; une valeur positive indique une perte de performance."""
    return float((propre - corrompu) / propre) if propre > 1e-12 else 0.0  # Évite une division par zéro.


def report_scenario(z_i: np.ndarray, z_t: np.ndarray, z_i_propre: np.ndarray, z_t_propre: np.ndarray, pertinence: np.ndarray, reference: Dict[str, float], modalite: str, ks: Sequence[int], points: int) -> Dict[str, Any]:  # Évalue un scénario.
    """Calcule les performances absolues, la dégradation, la cohérence et la stabilité topologique."""
    metriques = retrieval_metrics(z_i, z_t, pertinence, ks)  # Calcule les métriques de recherche.
    rapport: Dict[str, Any] = {cle: metriques[cle] for cle in ("mean_recall_at_1", "mean_recall_at_5", "mean_map", "i2t_recall_at_1", "i2t_recall_at_5", "t2i_recall_at_1", "t2i_recall_at_5", "i2t_map", "t2i_map")}  # Conserve les métriques utiles.
    for cle in ("mean_recall_at_1", "mean_recall_at_5", "mean_map"):  # Parcourt les métriques principales.
        rapport[f"degradation_{cle}"] = relative_degradation(reference[cle], metriques[cle])  # Calcule la dégradation relative.
    touche = z_i if modalite == "image" else z_t  # Sélectionne la modalité corrompue.
    propre = z_i_propre if modalite == "image" else z_t_propre  # Sélectionne sa version propre.
    rapport["coherence_cosinus"] = float((touche * propre).sum(axis=1).mean())  # Mesure la cohérence des représentations.
    n = min(points, touche.shape[0])  # Limite la taille du nuage.
    distances_propres = rips_diagrams(pairwise_euclidean(propre[:n]))  # Calcule les diagrammes propres.
    distances_touchees = rips_diagrams(pairwise_euclidean(touche[:n]))  # Calcule les diagrammes corrompus.
    for dimension in (0, 1):  # Parcourt H0 et H1.
        ecarts = diagram_distances(distances_propres[dimension], distances_touchees[dimension])  # Compare les diagrammes.
        rapport[f"stabilite_h{dimension}_bottleneck"] = ecarts["bottleneck"]  # Enregistre la distance de bottleneck.
        rapport[f"stabilite_h{dimension}_wasserstein1"] = ecarts["wasserstein_1"]  # Enregistre la distance de Wasserstein.
    return rapport  # Retourne le rapport.


class RobustnessEvaluator:  # Pilote l'évaluation sur le test.
    """Évalue un modèle entraîné sur le test, propre et corrompu, avec des données identiques pour tous."""

    def __init__(self, cfg: Dict[str, Any], manifeste: Dict[str, Any], scenarios: Sequence[Tuple[str, str, float]], appareil: torch.device) -> None:  # Prépare l'évaluation.
        """Ouvre les caractéristiques de test mises en cache."""
        self.cfg = cfg  # Conserve la configuration.
        self.scenarios = list(scenarios)  # Conserve les scénarios.
        self.appareil = appareil  # Conserve le device.
        self.extracteur = FeatureExtractor(cfg, manifeste)  # Calcule les chemins du cache.
        self.entrees = manifeste["splits"]["test"]  # Conserve les entrées de test.
        self.propre = FeatureSplit(self.extracteur.racine, self.entrees, "test")  # Ouvre les caractéristiques propres.
        self.pertinence = self.propre.relevance_matrix()  # Construit la matrice de pertinence.
        e = cfg["evaluation"]  # Récupère la section d'évaluation.
        self.taille_lot = int(e["batch_size"])  # Conserve la taille des lots.
        self.ks = list(e["recall_ks"])  # Conserve les rangs de coupure.
        self.points = int(e["topology_points"])  # Conserve la taille des nuages topologiques.
        self.echantillons_tokens = int(e["token_stability_samples"])  # Conserve le nombre d'échantillons de tokens.

    def _split_corrompu(self, modalite: str, nom: str, severite: float) -> FeatureSplit:  # Ouvre un scénario.
        """Ouvre le sous-ensemble où seule la modalité indiquée est corrompue."""
        prefixe = f"test__{scenario_name(nom, severite)}"  # Calcule le préfixe du scénario.
        if modalite == "image":  # Traite une corruption d'image.
            return FeatureSplit(self.extracteur.racine, self.entrees, "test", prefixe_image=prefixe)  # Remplace uniquement les images.
        return FeatureSplit(self.extracteur.racine, self.entrees, "test", prefixe_texte=prefixe)  # Remplace uniquement les textes.

    def evaluate_model(self, modele: torch.nn.Module) -> Dict[str, Any]:  # Évalue un modèle.
        """Évalue la recherche propre, les corruptions, les modalités manquantes et la stabilité des tokens."""
        z_i, z_t = encode_split(modele, self.propre, self.taille_lot, self.appareil)  # Encode les données propres.
        par_requete = retrieval_per_query(np.matmul(z_i, z_t.T), self.pertinence)  # Calcule les mesures par requête.
        propre = summarize_retrieval(par_requete, self.ks)  # Résume les mesures.
        resultats: Dict[str, Any] = {"propre": propre, "par_requete": par_requete, "z_image": z_i, "z_texte": z_t, "scenarios": {}, "stabilite_tokens": {}}  # Prépare les résultats.
        for modalite, nom, severite in self.scenarios:  # Parcourt les scénarios.
            split = self._split_corrompu(modalite, nom, severite)  # Ouvre le scénario.
            c_i, c_t = encode_split(modele, split, self.taille_lot, self.appareil)  # Encode le scénario.
            resultats["scenarios"][scenario_name(nom, severite)] = report_scenario(c_i, c_t, z_i, z_t, self.pertinence, propre, modalite, self.ks, self.points)  # Évalue le scénario.
            stabilite = self._stabilite_tokens(modele, split, modalite)  # Mesure la stabilité des tokens.
            if stabilite is not None:  # Vérifie que la variante produit des tokens.
                resultats["stabilite_tokens"][scenario_name(nom, severite)] = stabilite  # Enregistre la stabilité.
        resultats["modalite_manquante"] = self.missing_modality(modele)  # Évalue les modalités manquantes.
        return resultats  # Retourne les résultats.

    def _stabilite_tokens(self, modele: torch.nn.Module, split: FeatureSplit, modalite: str) -> Optional[Dict[str, float]]:  # Mesure la stabilité.
        """Compare les diagrammes de tokens propres et corrompus, et vérifie la borne démontrée."""
        propres = token_cloud_filtrations(modele, self.propre, modalite, self.echantillons_tokens, self.appareil)  # Lit les tokens propres.
        touches = token_cloud_filtrations(modele, split, modalite, self.echantillons_tokens, self.appareil)  # Lit les tokens corrompus.
        if propres is None or touches is None:  # Vérifie que la variante produit des tokens.
            return None  # Signale l'absence de mesure.
        return token_stability(propres, touches, modele.filtration.lipschitz_bound()["borne_stabilite"])  # Mesure la stabilité.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def missing_modality(self, modele: torch.nn.Module) -> Optional[Dict[str, Any]]:  # Évalue les modalités manquantes.
        """Compare l'embedding fusionné complet à celui obtenu avec une seule modalité réelle ; rien n'est simulé."""
        if not modele.variante.banque:  # Vérifie que la variante possède une fusion.
            return None  # Signale l'absence de mesure.
        modele.eval()  # Désactive le dropout.
        complets: List[np.ndarray] = []  # Prépare les embeddings complets.
        image_seule: List[np.ndarray] = []  # Prépare les embeddings issus de l'image seule.
        texte_seul: List[np.ndarray] = []  # Prépare les embeddings issus du texte seul.
        for lot in self.propre.iter_eval_pairs(self.taille_lot, self.appareil):  # Parcourt les paires de test.
            entrees = modality_inputs(lot)  # Convertit le lot.
            complets.append(modele(entrees)["fused_embedding"].cpu().numpy())  # Encode les deux modalités.
            image_seule.append(modele({"image": entrees["image"]})["fused_embedding"].cpu().numpy())  # Encode l'image seule.
            texte_seul.append(modele({"text": entrees["text"]})["fused_embedding"].cpu().numpy())  # Encode le texte seul.
        f = np.concatenate(complets)  # Assemble les embeddings complets.
        i = np.concatenate(image_seule)  # Assemble les embeddings de l'image seule.
        t = np.concatenate(texte_seul)  # Assemble les embeddings du texte seul.
        identite = np.eye(f.shape[0], dtype=bool)  # Chaque requête doit retrouver sa propre version complète.
        return {  # Retourne le rapport.
            "cosinus_complet_vs_image_seule": float((f * i).sum(axis=1).mean()),  # Mesure la cohérence avec l'image seule.
            "cosinus_complet_vs_texte_seul": float((f * t).sum(axis=1).mean()),  # Mesure la cohérence avec le texte seul.
            "recall_at_1_image_seule_vers_complet": retrieval_metrics(i, f, identite, self.ks)["i2t_recall_at_1"],  # Mesure le Recall@1 depuis l'image seule.
            "recall_at_5_image_seule_vers_complet": retrieval_metrics(i, f, identite, self.ks)["i2t_recall_at_5"],  # Mesure le Recall@5 depuis l'image seule.
            "recall_at_1_texte_seul_vers_complet": retrieval_metrics(t, f, identite, self.ks)["i2t_recall_at_1"],  # Mesure le Recall@1 depuis le texte seul.
            "recall_at_5_texte_seul_vers_complet": retrieval_metrics(t, f, identite, self.ks)["i2t_recall_at_5"],  # Mesure le Recall@5 depuis le texte seul.
        }  # Termine le rapport.

    def evaluate_clip(self) -> Optional[Dict[str, Any]]:  # Évalue la référence CLIP.
        """Évalue CLIP pré-entraîné, sans aucun entraînement, sur le même test et les mêmes corruptions."""
        racine = self.extracteur.racine_clip  # Lit le dossier des embeddings CLIP.
        if racine is None or not (Path(racine) / "test.image.npy").is_file():  # Vérifie la présence des embeddings.
            return None  # Signale l'absence de CLIP.
        z_i = np.asarray(np.load(Path(racine) / "test.image.npy"), dtype=np.float32)  # Lit les embeddings d'images.
        z_t = np.asarray(np.load(Path(racine) / "test.text.npy"), dtype=np.float32)  # Lit les embeddings de textes.
        z_i = z_i / np.linalg.norm(z_i, axis=1, keepdims=True).clip(min=1e-9)  # Renormalise après la demi-précision.
        z_t = z_t / np.linalg.norm(z_t, axis=1, keepdims=True).clip(min=1e-9)  # Renormalise après la demi-précision.
        par_requete = retrieval_per_query(np.matmul(z_i, z_t.T), self.pertinence)  # Calcule les mesures par requête.
        propre = summarize_retrieval(par_requete, self.ks)  # Résume les mesures.
        resultats: Dict[str, Any] = {"propre": propre, "par_requete": par_requete, "z_image": z_i, "z_texte": z_t, "scenarios": {}}  # Prépare les résultats.
        for modalite, nom, severite in self.scenarios:  # Parcourt les scénarios.
            suffixe = "image" if modalite == "image" else "text"  # Choisit le fichier à relire.
            corrompu = np.asarray(np.load(Path(racine) / f"test__{scenario_name(nom, severite)}.{suffixe}.npy"), dtype=np.float32)  # Lit les embeddings corrompus.
            corrompu = corrompu / np.linalg.norm(corrompu, axis=1, keepdims=True).clip(min=1e-9)  # Renormalise après la demi-précision.
            c_i, c_t = (corrompu, z_t) if modalite == "image" else (z_i, corrompu)  # Remplace seulement la modalité corrompue.
            resultats["scenarios"][scenario_name(nom, severite)] = report_scenario(c_i, c_t, z_i, z_t, self.pertinence, propre, modalite, self.ks, self.points)  # Évalue le scénario.
        return resultats  # Retourne les résultats.
