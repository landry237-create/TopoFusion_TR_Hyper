# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Vérifie le modèle, les pertes, les métriques, l'entraînement et les points de contrôle."""

import numpy as np  # Fournit les tableaux de test.
import pytest  # Fournit le cadre de test.
import torch  # Fournit les tenseurs.

from evaluation.diagnostics import alignment_report, collapse_flags, effective_rank, representation_diagnostics  # Importe les diagnostics.
from evaluation.embedding import evaluate_retrieval  # Importe l'évaluation.
from evaluation.robustness_eval import relative_degradation  # Importe la dégradation relative.
from evaluation.topology_eval import diagram_distances, structure_report, token_stability  # Importe les mesures topologiques.
from losses.losses import CompositeLoss, geometric_loss, info_nce, mantel_correlation, pairwise_distances, signature_loss, variance_hinge  # Importe les pertes.
from models.attention import TopologicalAttention  # Importe l'attention.
from models.encoders import ExperimentalAudioEncoder, ExperimentalVideoEncoder, ModalityProjector, ModalityUnavailableError, pool_tokens  # Importe les encodeurs.
from models.latent_bank import AdaptiveFusion, SemanticLatentBank, SharedPrivateFactorization, orthogonality_penalty  # Importe la banque.
from models.multimodal_framework import VARIANTS, TopoFusionModel, get_variant, modality_inputs  # Importe le modèle.
from models.robustness import FeatureRobustnessModule  # Importe les perturbations.
from training.trainer import NumericalInstabilityError, Trainer, learning_rate_factor, phase_of_epoch  # Importe l'entraînement.
from utils.config import ConfigError, deep_update, load_config  # Importe la configuration.
from utils.metrics import chance_level, kruskal_stress, mantel_r, neighborhood_preservation, paired_bootstrap_difference, retrieval_metrics, retrieval_per_query  # Importe les métriques.


def _lot(b=8, t_img=5, t_txt=6, d=16):  # Fabrique un lot de test.
    """Retourne un lot aléatoire réservé aux tests de forme et de gradient."""
    masque = torch.ones(b, t_txt, dtype=torch.bool)  # Crée le masque.
    masque[:, 4:] = False  # Simule du padding.
    return {"image_tokens": torch.randn(b, t_img, d), "text_tokens": torch.randn(b, t_txt, d), "text_mask": masque, "text_ids": torch.arange(b)}  # Retourne le lot.


def _modele(cfg, nom="B4"):  # Fabrique un modèle de test.
    """Retourne un modèle réduit de la variante demandée."""
    torch.manual_seed(0)  # Fixe la graine.
    return TopoFusionModel(cfg, get_variant(nom), {"image": 16, "text": 16})  # Construit le modèle.


def _apres_une_etape(cfg, modele, lot):  # Exécute une étape d'optimisation.
    """Exécute une vraie étape d'optimisation, car la tête structurelle part de zéro."""
    critere = CompositeLoss(cfg, modele.variante)  # Construit l'objectif.
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=1e-3)  # Construit l'optimiseur.
    perte, _ = critere(modele(modality_inputs(lot)), None, modele.temperature, lot["text_ids"], 2)  # Calcule l'objectif.
    perte.backward()  # Calcule les gradients.
    optimiseur.step()  # Met à jour les poids.
    optimiseur.zero_grad(set_to_none=True)  # Efface les gradients.


@pytest.mark.parametrize("nom", sorted(VARIANTS))  # Répète le test pour chaque variante.
def test_passe_avant_de_chaque_variante(cfg, nom):  # Vérifie la passe avant.
    """Chaque variante doit produire des embeddings normalisés aux bonnes dimensions."""
    modele = _modele(cfg, nom)  # Construit le modèle.
    sorties = modele(modality_inputs(_lot()))  # Exécute le modèle.
    for modalite in ("image", "text"):  # Parcourt les modalités.
        z = sorties["embeddings"][modalite]  # Lit l'embedding.
        assert z.shape == (8, 32)  # Vérifie la forme.
        assert torch.allclose(z.norm(dim=-1), torch.ones(8), atol=1e-5)  # Vérifie la normalisation.


def test_tokens_partages_et_prives_dimensions(cfg):  # Vérifie les espaces partagé et privé.
    """Les tokens partagés et privés doivent respecter K_c et K - K_c."""
    sorties = _modele(cfg)(modality_inputs(_lot()))  # Exécute le modèle.
    assert sorties["shared"]["image"].shape == (8, 6, 32)  # Vérifie les tokens partagés.
    assert sorties["private"]["text"].shape == (8, 4, 32)  # Vérifie les tokens privés.
    assert sorties["reconstruction"]["text"].shape == (8, 16)  # Vérifie la reconstruction.


def test_tous_les_modules_recoivent_un_gradient(cfg):  # Corrige le défaut critique C5.
    """Dans B4, l'attention et tous les modules doivent recevoir un gradient réel."""
    modele = _modele(cfg)  # Construit le modèle.
    critere = CompositeLoss(cfg, get_variant("B4"))  # Construit l'objectif.
    lot = _lot()  # Crée un lot.
    _apres_une_etape(cfg, modele, lot)  # Sort la tête structurelle de son point de départ nul.
    perturbees = modele({"image": (lot["image_tokens"] + 0.01, torch.ones(8, 5, dtype=torch.bool)), "text": (lot["text_tokens"] + 0.01, lot["text_mask"])}, fusionner=False)  # Exécute la vue perturbée.
    perte, _ = critere(modele(modality_inputs(lot)), perturbees, modele.temperature, lot["text_ids"], 3)  # Calcule l'objectif.
    perte.backward()  # Calcule les gradients.
    morts = [n for n, p in modele.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]  # Liste les paramètres sans gradient.
    assert [n for n in morts if not n.endswith("fusion.confiance.3.bias")] == []  # Le biais final du softmax est invariant par translation et peut avoir un gradient nul.
    assert modele.log_temperature.grad.abs().sum() > 0  # Vérifie que la température est réellement apprise.


def test_embedding_depend_de_la_voie_topologique(cfg):  # Vérifie la contribution de la topologie.
    """Retirer les tokens topologiques à l'inférence doit modifier l'embedding de recherche."""
    modele = _modele(cfg)  # Construit le modèle.
    lot = _lot()  # Crée un lot.
    _apres_une_etape(cfg, modele, lot)  # Sort la tête structurelle de son point de départ nul.
    modele.eval()  # Désactive le dropout.
    with torch.no_grad():  # Désactive le gradient.
        a = modele(modality_inputs(lot))["embeddings"]["image"]  # Encode avec la topologie.
        b = modele(modality_inputs(lot), desactiver_topo=True)["embeddings"]["image"]  # Encode sans tokens topologiques.
    assert (a - b).abs().max() > 1e-4  # Vérifie la différence.


@pytest.mark.parametrize("nom", ["B3", "B4"])  # Répète le test pour les deux variantes structurelles.
def test_voie_structurelle_demarre_sans_perturber_l_embedding(cfg, nom):  # Corrige le défaut trouvé sur la validation.
    """À l'initialisation, la voie structurelle ne doit ajouter aucun décalage constant à l'embedding."""
    modele = _modele(cfg, nom)  # Construit le modèle.
    modele.eval()  # Désactive le dropout.
    lot = _lot()  # Crée un lot.
    with torch.no_grad():  # Désactive le gradient.
        sorties = modele(modality_inputs(lot))  # Exécute le modèle.
    for modalite in ("image", "text"):  # Parcourt les modalités.
        attendu = torch.nn.functional.normalize(sorties["semantic"][modalite], dim=-1)  # Calcule l'embedding sémantique seul.
        assert torch.allclose(sorties["embeddings"][modalite], attendu, atol=1e-6)  # Vérifie l'absence de décalage initial.


def test_voie_structurelle_recoit_un_gradient_malgre_le_depart_a_zero(cfg):  # Vérifie que la voie reste apprenable.
    """Partir de zéro ne doit pas bloquer l'apprentissage de la voie structurelle."""
    modele = _modele(cfg, "B3")  # Construit le modèle.
    lot = _lot()  # Crée un lot.
    critere = CompositeLoss(cfg, get_variant("B3"))  # Construit l'objectif.
    perte, _ = critere(modele(modality_inputs(lot)), None, modele.temperature, lot["text_ids"], 3)  # Calcule l'objectif.
    perte.backward()  # Calcule les gradients.
    assert modele.tete_structurelle[-1].weight.grad.abs().sum() > 0.0  # Vérifie le gradient de la dernière couche.


def test_tokens_topologiques_varient_entre_echantillons(cfg):  # Vérifie la standardisation.
    """Les tokens topologiques doivent varier nettement entre échantillons, après standardisation."""
    modele = _modele(cfg, "B3")  # Construit le modèle.
    modele.train()  # Utilise les statistiques du lot.
    lot = _lot(b=32)  # Crée un lot.
    vue = modele(modality_inputs(lot))["views"]["image"]  # Exécute la voie structurelle.
    assert vue.pooled.std(dim=0).mean().item() > 0.01  # Vérifie la variation entre échantillons.


def test_gradient_de_la_perte_top_traverse_la_persistance(cfg):  # Vérifie le gradient de la perte topologique.
    """L_top doit produire un gradient sur les embeddings à partir des seules arêtes critiques."""
    z = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1).requires_grad_(True)  # Crée un nuage.
    reference = pairwise_distances(torch.nn.functional.normalize(torch.randn(16, 8), dim=-1))  # Crée une référence.
    perte = signature_loss(reference, pairwise_distances(z), (0, 1))  # Calcule la perte.
    assert perte.item() > 0.0  # Vérifie que la perte n'est pas nulle.
    perte.backward()  # Calcule les gradients.
    assert z.grad.abs().sum() > 0.0  # Vérifie le gradient.


def test_mode_modalite_absente_exclut_la_modalite(cfg):  # Vérifie le masquage de modalité.
    """Une modalité absente ne doit recevoir aucun poids et ne doit pas modifier la fusion."""
    fusion = AdaptiveFusion(8)  # Crée la fusion.
    a = torch.randn(2, 3, 8)  # Crée les tokens de la première modalité.
    b = torch.randn(2, 3, 8)  # Crée les tokens de la seconde modalité.
    presence = {"image": torch.tensor([True, True]), "text": torch.tensor([True, False])}  # L'échantillon 1 n'a pas de texte.
    fusionne, alphas, noms = fusion({"image": a, "text": b}, presence)  # Fusionne.
    rang_texte = noms.index("text")  # Trouve la position du texte.
    assert alphas[1, rang_texte].abs().sum() == 0.0  # Vérifie le poids nul.
    assert torch.allclose(fusionne[1], a[1])  # Vérifie que seule l'image est utilisée.
    fusionne_2, _, _ = fusion({"image": a, "text": torch.randn(2, 3, 8)}, presence)  # Change le texte absent.
    assert torch.allclose(fusionne_2[1], fusionne[1])  # Vérifie l'absence d'influence.
    assert torch.allclose(alphas.sum(dim=1), torch.ones(2, 3, 1))  # Vérifie la combinaison convexe.


def test_fusion_refuse_un_echantillon_sans_modalite():  # Vérifie le cas sans modalité.
    """Un échantillon sans aucune modalité doit être refusé."""
    with pytest.raises(ValueError):  # Attend une erreur explicite.
        AdaptiveFusion(8)({"image": torch.randn(1, 2, 8)}, {"image": torch.tensor([False])})  # Essaie sans modalité.


def test_modele_fonctionne_avec_une_seule_modalite_reelle(cfg):  # Vérifie le scénario de modalité manquante.
    """Le modèle doit fonctionner avec une seule modalité, sans fabriquer l'autre."""
    modele = _modele(cfg)  # Construit le modèle.
    modele.eval()  # Désactive le dropout.
    lot = _lot()  # Crée un lot.
    sorties = modele({"image": (lot["image_tokens"], None)})  # Encode l'image seule.
    assert set(sorties["embeddings"]) == {"image"}  # Vérifie qu'aucune modalité n'est inventée.
    assert sorties["fused_embedding"].shape == (8, 32)  # Vérifie l'embedding fusionné.
    assert sorties["alpha_names"] == ["image"]  # Vérifie que la fusion ne voit que l'image.


def test_modalites_audio_video_absentes_du_modele(cfg):  # Vérifie l'absence de modalité artificielle.
    """Une modalité non construite doit être refusée, et aucune entrée audio ou vidéo n'existe."""
    modele = _modele(cfg)  # Construit le modèle.
    assert modele.modalites == ["image", "text"]  # Vérifie les modalités du modèle.
    with pytest.raises(KeyError):  # Attend une erreur explicite.
        modele({"audio": (torch.randn(1, 2, 16), None)})  # Essaie une modalité absente.


def test_config_refuse_audio_video_sans_donnees():  # Vérifie la validation de configuration.
    """Activer l'audio ou la vidéo doit être refusé, faute de données réelles."""
    with pytest.raises(ConfigError, match="expérimentales"):  # Attend une erreur explicite.
        load_config("config.yaml", {"modalities": {"audio": True}})  # Essaie d'activer l'audio.


def test_encodeurs_experimentaux_exigent_de_vraies_entrees():  # Vérifie les extensions expérimentales.
    """Les encodeurs expérimentaux refusent une entrée absente ou de forme invalide."""
    audio = ExperimentalAudioEncoder("modele-non-charge")  # Crée l'encodeur audio sans charger de poids.
    video = ExperimentalVideoEncoder("modele-non-charge")  # Crée l'encodeur vidéo sans charger de poids.
    assert audio.STATUT == "experimental" and video.STATUT == "experimental"  # Vérifie le statut.
    with pytest.raises(ModalityUnavailableError):  # Attend une erreur explicite.
        audio(None)  # Essaie sans audio.
    with pytest.raises(ModalityUnavailableError):  # Attend une erreur explicite.
        video(None)  # Essaie sans vidéo.
    with pytest.raises(ValueError):  # Attend une erreur de forme.
        audio(torch.zeros(2, 64, 10))  # Essaie une forme invalide.
    with pytest.raises(ValueError):  # Attend une erreur de forme.
        video(torch.zeros(2, 3, 224, 224))  # Essaie une forme invalide.


def test_banque_tokens_orthogonaux_et_masque_padding():  # Corrige les défauts M1 et H2.
    """Les tokens de la banque doivent être orthogonaux, et le padding sans effet sur le résultat."""
    banque = SemanticLatentBank(6, 32, 2, 0.0)  # Crée la banque.
    banque.eval()  # Désactive le dropout.
    produit = banque.tokens @ banque.tokens.T  # Calcule les produits scalaires.
    assert torch.allclose(produit, torch.eye(6), atol=1e-5)  # Vérifie l'orthogonalité.
    tokens = torch.randn(2, 8, 32)  # Crée des tokens.
    masque = torch.ones(2, 8, dtype=torch.bool)  # Crée le masque.
    masque[:, 5:] = False  # Marque le padding.
    autres = tokens.clone()  # Copie les tokens.
    autres[:, 5:] = 99.0  # Modifie uniquement le padding.
    assert torch.allclose(banque(tokens, masque), banque(autres, masque), atol=1e-5)  # Vérifie l'absence d'influence du padding.


def test_factorisation_et_penalite_orthogonalite():  # Vérifie la factorisation.
    """Des espaces orthogonaux doivent avoir une pénalité nulle, et des espaces identiques une pénalité élevée."""
    facto = SharedPrivateFactorization(3, 8, ["image", "text"])  # Crée la factorisation.
    partage, prive = facto(torch.randn(2, 5, 8), "image")  # Sépare les tokens.
    assert partage.shape == (2, 3, 8) and prive.shape == (2, 2, 8)  # Vérifie les formes.
    a = torch.eye(8)[:3].unsqueeze(0)  # Crée trois vecteurs de base.
    b = torch.eye(8)[3:6].unsqueeze(0)  # Crée trois vecteurs orthogonaux aux précédents.
    assert orthogonality_penalty(a, b).item() == pytest.approx(0.0, abs=1e-7)  # Vérifie la nullité.
    assert orthogonality_penalty(a, a).item() > 0.3  # Vérifie la pénalité d'espaces identiques.


def test_biais_de_l_attention_modifie_les_poids():  # Vérifie l'attention topologique.
    """Le biais topologique doit réellement modifier les poids d'attention."""
    attention = TopologicalAttention(16, 2, 0.0)  # Crée l'attention.
    attention.eval()  # Désactive le dropout.
    x = torch.randn(1, 4, 16)  # Crée des tokens.
    biais = torch.zeros(1, 4, 4)  # Crée un biais nul.
    biais[0, 0, 1] = 20.0  # Favorise fortement le lien entre les tokens 0 et 1.
    _, sans = attention(x)  # Calcule l'attention sans biais.
    _, avec = attention(x, biais)  # Calcule l'attention avec biais.
    assert avec.shape == (1, 2, 4, 4)  # Vérifie la forme.
    assert avec[0, :, 0, 1].min() > 0.99 and sans[0, :, 0, 1].max() < 0.99  # Vérifie l'effet du biais.
    assert torch.allclose(avec.sum(dim=-1), torch.ones(1, 2, 4))  # Vérifie la normalisation.


def test_projecteur_et_pooling():  # Vérifie le projecteur et le pooling.
    """Le projecteur doit changer la dimension, et le pooling ignorer le padding."""
    assert ModalityProjector(16, 32).eval()(torch.randn(2, 3, 16)).shape == (2, 3, 32)  # Vérifie le projecteur.
    tokens = torch.tensor([[[1.0], [3.0], [100.0]]])  # Crée des tokens avec une valeur de padding extrême.
    assert pool_tokens(tokens, torch.tensor([[True, True, False]])).item() == 2.0  # Vérifie la moyenne sans padding.


def test_robustesse_conserve_au_moins_une_modalite_et_un_token():  # Vérifie les perturbations.
    """Les perturbations doivent conserver au moins une modalité et un token."""
    module = FeatureRobustnessModule(0.05, 1.0, 1.0)  # Crée un module qui retire tout ce qu'il peut.
    tokens, masque = module.perturb(torch.randn(4, 6, 8), torch.ones(4, 6, dtype=torch.bool))  # Perturbe les tokens.
    assert masque[:, 0].all() and not masque[:, 1:].any() and tokens.shape == (4, 6, 8)  # Vérifie la conservation du premier token.
    presence = module.availability(32, ["image", "text"], torch.device("cpu"))  # Tire la présence des modalités.
    assert (presence["image"] | presence["text"]).all()  # Vérifie qu'au moins une modalité reste présente.


def test_info_nce_exclut_les_faux_negatifs():  # Vérifie InfoNCE.
    """Deux légendes identiques ne doivent pas être traitées comme des négatifs."""
    z = torch.nn.functional.normalize(torch.eye(4), dim=-1)  # Crée des embeddings orthogonaux.
    bon = info_nce(z, z, torch.tensor(0.1)).item()  # Calcule la perte de paires parfaites.
    melange = info_nce(z, z[[1, 0, 2, 3]], torch.tensor(0.1)).item()  # Calcule la perte avec deux paires échangées.
    assert bon < 0.01 and melange > bon  # Vérifie le comportement attendu.
    x = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]]), dim=-1)  # Crée deux embeddings proches.
    ids = torch.tensor([7, 7, 3])  # Marque les deux premiers textes comme identiques.
    assert info_nce(x, x, torch.tensor(0.1), ids).item() < info_nce(x, x, torch.tensor(0.1)).item()  # Vérifie la réduction de la perte.


def test_perte_geometrique_penalise_l_effondrement():  # Corrige le défaut H3.
    """Une représentation effondrée doit recevoir une perte élevée, et non nulle."""
    x = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)  # Crée des entrées.
    fidele = geometric_loss({"image": x}, {"image": x})  # Calcule la perte d'une représentation fidèle.
    effondree = geometric_loss({"image": torch.ones(16, 8) / (8 ** 0.5)}, {"image": x})  # Calcule la perte d'une représentation effondrée.
    assert fidele.item() < 1e-5  # Vérifie la perte nulle d'une représentation fidèle.
    assert effondree.item() == pytest.approx(1.0, abs=1e-3)  # Vérifie la pénalité maximale de l'effondrement.


def test_correlation_mantel_invariante_a_l_echelle():  # Vérifie la corrélation de Mantel.
    """La corrélation de Mantel doit être invariante au changement d'échelle."""
    d = pairwise_distances(torch.randn(12, 4))  # Calcule une matrice de distances.
    assert mantel_correlation(d, 7.0 * d).item() == pytest.approx(1.0, abs=1e-4)  # Vérifie l'invariance.


def test_distances_sans_gradient_infini():  # Vérifie la stabilité numérique.
    """Des points confondus ne doivent pas produire de gradient infini ou NaN."""
    x = torch.zeros(4, 3, requires_grad=True)  # Crée des points confondus.
    pairwise_distances(x).sum().backward()  # Calcule les gradients.
    assert torch.isfinite(x.grad).all()  # Vérifie la finitude.


def test_variance_hinge_detecte_l_effondrement():  # Vérifie la pénalité de variance.
    """La pénalité de variance doit être positive pour un lot effondré et nulle pour un lot varié."""
    assert variance_hinge(torch.ones(8, 4)).item() > 0.05  # Vérifie un lot effondré.
    assert variance_hinge(torch.randn(256, 4)).item() == pytest.approx(0.0, abs=1e-6)  # Vérifie un lot varié.


def test_termes_de_perte_selon_phase_et_variante(cfg):  # Vérifie l'activation des termes.
    """Les termes doivent s'activer selon la phase et la variante."""
    critere = CompositeLoss(cfg, get_variant("B4"))  # Crée l'objectif complet.
    assert "top" not in critere.active_terms(1) and "rob" not in critere.active_terms(2)  # Vérifie l'activation progressive.
    assert {"top", "rob", "orth", "rec", "var", "geom", "fusion"} == set(critere.active_terms(3))  # Vérifie la phase finale.
    assert "top" not in CompositeLoss(cfg, get_variant("B4_sans_L_top")).active_terms(3)  # Vérifie l'ablation.
    assert CompositeLoss(cfg, get_variant("B1")).active_terms(3) == ()  # Vérifie la variante contrastive.


def test_retrieval_matrice_analytique():  # Vérifie les métriques sur un cas calculé à la main.
    """Le rappel et le mAP doivent correspondre à un classement connu."""
    sims = np.array([[0.9, 0.8, 0.1], [0.1, 0.2, 0.9]])  # Fixe les similarités.
    pertinence = np.array([[True, False, False], [False, True, False]])  # Fixe les éléments pertinents.
    mesures = retrieval_per_query(sims, pertinence)  # Calcule les mesures.
    assert mesures["rangs_i2t"].tolist() == [0, 1]  # Vérifie les rangs image vers texte.
    assert mesures["ap_i2t"].tolist() == pytest.approx([1.0, 0.5])  # Vérifie la précision moyenne.


def test_map_avec_plusieurs_pertinents():  # Vérifie que le mAP n'est pas un simple MRR.
    """Le mAP doit prendre en compte plusieurs éléments pertinents, contrairement à l'ancien calcul."""
    sims = np.array([[0.9, 0.8, 0.7, 0.6]])  # Fixe les similarités.
    pertinence = np.array([[True, False, True, False]])  # Fixe deux éléments pertinents, aux rangs 1 et 3.
    assert retrieval_per_query(sims, pertinence)["ap_i2t"][0] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)  # Vérifie la précision moyenne.


def test_scores_egaux_ne_donnent_pas_un_score_parfait():  # Vérifie la gestion des égalités.
    """Un modèle effondré doit obtenir un score proche du hasard, jamais un score parfait."""
    z = np.ones((200, 8)) / np.sqrt(8.0)  # Crée des embeddings identiques.
    resultats = retrieval_metrics(z, z, np.eye(200, dtype=bool), (1, 5))  # Calcule les métriques.
    assert resultats["mean_recall_at_1"] < 0.05  # Vérifie un score proche du hasard.


def test_niveau_du_hasard():  # Vérifie le calcul du niveau du hasard.
    """Le niveau du hasard doit correspondre à une valeur simple dans un cas élémentaire."""
    assert chance_level(np.eye(100, dtype=bool), 1)["t2i"] == pytest.approx(0.01)  # Vérifie le rappel à 1.
    assert chance_level(np.eye(100, dtype=bool), 5)["i2t"] == pytest.approx(0.05)  # Vérifie le rappel à 5.


def test_metriques_de_structure():  # Vérifie les métriques de structure.
    """Une isométrie doit donner un stress nul, une corrélation de 1 et des voisinages conservés."""
    x = np.random.default_rng(0).normal(size=(40, 6))  # Tire un nuage.
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)  # Calcule les distances.
    assert kruskal_stress(d, 3.0 * d) == pytest.approx(0.0, abs=1e-6)  # Vérifie le stress.
    assert mantel_r(d, 3.0 * d) == pytest.approx(1.0, abs=1e-9)  # Vérifie la corrélation.
    assert neighborhood_preservation(d, 3.0 * d, 5) == pytest.approx(1.0)  # Vérifie les voisinages.
    assert mantel_r(d, np.ones_like(d)) == 0.0  # Vérifie le cas d'une matrice constante.


def test_ancien_faux_score_d_isometrie_n_existe_plus():  # Corrige le défaut H8.
    """Deux nuages indépendants doivent donner une corrélation proche de 0, et non 0,89."""
    generateur = np.random.default_rng(1)  # Fixe le générateur.
    a = generateur.normal(size=(80, 16))  # Tire un premier nuage.
    b = generateur.normal(size=(80, 16))  # Tire un second nuage indépendant.
    d_a = np.linalg.norm(a[:, None] - a[None], axis=-1)  # Calcule ses distances.
    d_b = np.linalg.norm(b[:, None] - b[None], axis=-1)  # Calcule ses distances.
    assert abs(mantel_r(d_a, d_b)) < 0.2  # Vérifie l'absence de corrélation artificielle.


def test_bootstrap_apparie():  # Vérifie la comparaison statistique.
    """Une différence nette doit être significative, et des résultats identiques ne pas l'être."""
    generateur = np.random.default_rng(0)  # Fixe le générateur.
    a = generateur.random(1000) < 0.7  # Simule un meilleur modèle.
    b = generateur.random(1000) < 0.3  # Simule un moins bon modèle.
    assert paired_bootstrap_difference(a, b)["significatif"] is True  # Vérifie une différence nette.
    assert paired_bootstrap_difference(a, a)["significatif"] is False  # Vérifie des résultats identiques.


def test_diagnostics_effondrement():  # Vérifie les diagnostics.
    """Les diagnostics doivent détecter un effondrement et mesurer le rang."""
    assert collapse_flags(np.ones((64, 8)))["effondre"] is True  # Vérifie un nuage effondré.
    assert collapse_flags(np.random.default_rng(0).normal(size=(64, 8)))["effondre"] is False  # Vérifie un nuage varié.
    assert effective_rank(np.random.default_rng(0).normal(size=(500, 4))) > 3.0  # Vérifie le rang effectif.
    assert alignment_report(np.eye(4), np.eye(4))["marge_alignement"] == pytest.approx(1.0)  # Vérifie l'alignement.


def test_degradation_relative():  # Vérifie la dégradation relative.
    """La dégradation relative doit être positive lors d'une perte de performance."""
    assert relative_degradation(0.5, 0.25) == pytest.approx(0.5)  # Vérifie une perte de moitié.
    assert relative_degradation(0.0, 0.25) == 0.0  # Vérifie le cas d'une référence nulle.


def test_distances_entre_diagrammes():  # Vérifie les distances entre diagrammes.
    """Le bottleneck doit valoir une valeur connue, et les classes essentielles être écartées."""
    a = np.array([[0.0, 1.0], [0.0, np.inf]])  # Crée un diagramme.
    b = np.array([[0.0, 1.5], [0.0, np.inf]])  # Crée un second diagramme.
    assert diagram_distances(a, b)["bottleneck"] == pytest.approx(0.5)  # Vérifie la distance.


def test_structure_et_diagrammes_d_un_modele_effondre():  # Vérifie le cas limite de B0.
    """Un modèle effondré a des diagrammes presque vides : les mesures doivent rester finies sans planter."""
    assert diagram_distances(np.empty((0, 2)), np.empty((0, 2)))["bottleneck"] == 0.0  # Vérifie deux diagrammes vides.
    assert diagram_distances(np.empty((0, 2)), np.array([[0.0, 1.0]]))["wasserstein_1"] > 0.0  # Vérifie un diagramme vide contre un point.
    effondre = np.ones((24, 8)) / np.sqrt(8.0)  # Crée un nuage entièrement effondré.
    x = np.random.default_rng(3).normal(size=(24, 8))  # Tire un nuage d'entrée.
    x = x / np.linalg.norm(x, axis=1, keepdims=True)  # Normalise le nuage d'entrée.
    rapport = structure_report(effondre, effondre, x, x)  # Calcule le rapport.
    assert all(np.isfinite(valeur) for valeur in rapport.values())  # Vérifie que toutes les mesures sont finies.
    assert rapport["mantel_entree_latent_image"] == 0.0  # Vérifie l'absence de corrélation.


def test_rapport_de_structure_complet():  # Vérifie le rapport de structure.
    """Le rapport de structure doit contenir les mesures exactes de H0 et H1."""
    generateur = np.random.default_rng(2)  # Fixe le générateur.
    normaliser = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # Définit la normalisation.
    z_i, z_t, x_i, x_t = [normaliser(generateur.normal(size=(24, 8))) for _ in range(4)]  # Tire quatre nuages.
    rapport = structure_report(z_i, z_t, x_i, x_t)  # Calcule le rapport.
    for cle in ("mantel_image_texte", "stress_entree_latent_image", "h0_bottleneck_image_texte_latent", "h1_wasserstein1_entree_latent_texte", "fiabilite10_entree_latent_texte"):  # Parcourt quelques clés.
        assert cle in rapport and np.isfinite(rapport[cle])  # Vérifie leur présence.


def test_stabilite_des_tokens_respecte_la_borne(cfg):  # Vérifie la borne sur un modèle.
    """Sur la filtration d'un modèle, la borne démontrée ne doit jamais être violée."""
    modele = _modele(cfg)  # Construit le modèle.
    modele.eval()  # Désactive le dropout.
    u = torch.nn.functional.normalize(torch.randn(6, 6, 32), dim=-1)  # Crée des points.
    v = torch.nn.functional.normalize(u + 0.05 * torch.randn_like(u), dim=-1)  # Crée des points perturbés.
    with torch.no_grad():  # Désactive le gradient.
        propres = {"points": u.numpy(), "filtrations": modele.filtration(u).numpy().astype(np.float64)}  # Calcule la filtration propre.
        touches = {"points": v.numpy(), "filtrations": modele.filtration(v).numpy().astype(np.float64)}  # Calcule la filtration perturbée.
    rapport = token_stability(propres, touches, modele.filtration.lipschitz_bound()["borne_stabilite"])  # Mesure la stabilité.
    assert rapport["violations_de_la_borne"] == 0  # Vérifie l'absence de violation.
    assert rapport["rapport_maximal_observe"] <= rapport["borne_theorique"] + 1e-3  # Vérifie le rapport observé.


def test_planning_du_taux_d_apprentissage():  # Vérifie le planning.
    """Le planning doit monter pendant l'échauffement, puis descendre jusqu'au minimum."""
    assert learning_rate_factor(0, 100, 0.1, 0.01) == pytest.approx(0.1)  # Vérifie le début.
    assert learning_rate_factor(9, 100, 0.1, 0.01) == pytest.approx(1.0)  # Vérifie la fin de l'échauffement.
    assert learning_rate_factor(99, 100, 0.1, 0.01) < 0.02  # Vérifie la fin du planning.


def test_phases():  # Vérifie les phases.
    """Les phases doivent couvrir les époques selon les fractions demandées."""
    phases = [phase_of_epoch(e, 10, [0.2, 0.3, 0.5]) for e in range(10)]  # Calcule les phases.
    assert phases == [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]  # Vérifie le découpage.


def test_etape_entrainement_et_validation_complete(cfg, splits, tmp_path):  # Vérifie l'entraînement.
    """L'entraînement complet doit se dérouler, améliorer l'alignement et produire un point de contrôle."""
    train, val = splits  # Récupère les sous-ensembles.
    entraineur = Trainer(cfg, get_variant("B4"), train, val, tmp_path / "run", 42, torch.device("cpu"))  # Prépare l'entraînement.
    resume = entraineur.fit()  # Entraîne le modèle.
    assert resume["meilleure_epoque"] >= 0 and (tmp_path / "run" / "best.pt").is_file()  # Vérifie le point de contrôle.
    assert (tmp_path / "run" / "historique.json").is_file()  # Vérifie l'historique.
    assert len(entraineur.historique["train"]) == 3 and len(entraineur.historique["val"]) == 4  # Vérifie l'historique, avec la mesure initiale.
    assert "L_top" in entraineur.historique["train"][-1] and "L_rob" in entraineur.historique["train"][-1]  # Vérifie l'objectif complet.
    assert entraineur.historique["train"][0]["phase"] == 1 and entraineur.historique["train"][-1]["phase"] == 3  # Vérifie les phases.
    assert resume["epoque_debut_selection"] == 2  # Vérifie que la sélection n'a lieu qu'après l'objectif complet.
    assert "bloc" in entraineur.historique["normes_gradient"][-1]  # Vérifie le suivi du gradient de l'attention.
    diagnostics = representation_diagnostics(entraineur.modele, val, 16, torch.device("cpu"))  # Exécute les diagnostics.
    assert "partage_prive" in diagnostics and "lipschitz" in diagnostics  # Vérifie les diagnostics.
    assert diagnostics["lipschitz"]["borne_stabilite"] <= 2.0 + 1e-4  # Vérifie la borne après entraînement.


def test_contrastif_apprend_sur_un_signal_commun(cfg, splits, tmp_path):  # Vérifie la capacité d'apprentissage.
    """Une variante contrastive doit dépasser largement son niveau initial sur un signal partagé."""
    train, val = splits  # Récupère les sous-ensembles.
    cfg = deep_update(cfg, {"training": {"epochs": 12, "lr": 0.002}})  # Allonge l'entraînement de ce test.
    entraineur = Trainer(cfg, get_variant("B1"), train, val, tmp_path / "b1", 0, torch.device("cpu"))  # Prépare l'entraînement.
    entraineur.fit()  # Entraîne le modèle.
    avant = entraineur.historique["val"][0]["mean_recall_at_1"]  # Lit le score initial.
    apres = evaluate_retrieval(entraineur.modele, val, 8, torch.device("cpu"))["mean_recall_at_1"]  # Lit le score final.
    assert apres > avant + 0.2  # Vérifie l'apprentissage.


def test_point_de_controle_restitue_le_modele(cfg, splits, tmp_path):  # Vérifie la sauvegarde et le chargement.
    """Sauvegarder puis recharger doit redonner exactement les mêmes sorties."""
    train, val = splits  # Récupère les sous-ensembles.
    premier = Trainer(cfg, get_variant("B3"), train, val, tmp_path / "ck", 1, torch.device("cpu"))  # Prépare l'entraînement.
    premier.fit()  # Entraîne le modèle.
    lot = _lot()  # Crée un lot.
    premier.modele.eval()  # Désactive le dropout.
    with torch.no_grad():  # Désactive le gradient.
        attendu = premier.modele(modality_inputs(lot))["embeddings"]["image"]  # Encode avec le modèle d'origine.
    second = TopoFusionModel(cfg, get_variant("B3"), {"image": 16, "text": 16})  # Construit un nouveau modèle.
    second.load_state_dict(torch.load(tmp_path / "ck" / "best.pt", weights_only=False)["modele"])  # Charge le point de contrôle.
    second.eval()  # Désactive le dropout.
    with torch.no_grad():  # Désactive le gradient.
        obtenu = second(modality_inputs(lot))["embeddings"]["image"]  # Encode avec le modèle rechargé.
    assert torch.allclose(attendu, obtenu)  # Vérifie l'égalité.


def test_reprise_refuse_une_autre_configuration(cfg, splits, tmp_path):  # Vérifie la reprise.
    """La reprise doit refuser un point de contrôle produit par une autre configuration."""
    train, val = splits  # Récupère les sous-ensembles.
    Trainer(cfg, get_variant("B1"), train, val, tmp_path / "r", 1, torch.device("cpu")).fit()  # Produit un point de contrôle.
    autre = deep_update(cfg, {"training": {"lr": 0.5}})  # Modifie la configuration.
    with pytest.raises(ValueError, match="Reprise refusée"):  # Attend une erreur explicite.
        Trainer(autre, get_variant("B1"), train, val, tmp_path / "r", 1, torch.device("cpu")).resume()  # Essaie de reprendre.


def test_entrainement_reproductible(cfg, splits, tmp_path):  # Vérifie la reproductibilité.
    """Deux entraînements de même graine doivent donner les mêmes poids, et deux graines des poids différents."""
    train, val = splits  # Récupère les sous-ensembles.
    a = Trainer(cfg, get_variant("B4"), train, val, tmp_path / "a", 5, torch.device("cpu"))  # Prépare le premier entraînement.
    a.fit()  # Entraîne le premier modèle.
    b = Trainer(cfg, get_variant("B4"), train, val, tmp_path / "b", 5, torch.device("cpu"))  # Prépare le deuxième entraînement.
    b.fit()  # Entraîne le deuxième modèle.
    c = Trainer(cfg, get_variant("B4"), train, val, tmp_path / "c", 6, torch.device("cpu"))  # Prépare le troisième entraînement.
    c.fit()  # Entraîne le troisième modèle.
    poids_a = dict(a.modele.named_parameters())  # Lit les poids du premier modèle.
    poids_b = dict(b.modele.named_parameters())  # Lit les poids du deuxième modèle.
    poids_c = dict(c.modele.named_parameters())  # Lit les poids du troisième modèle.
    assert all(torch.equal(poids_a[n], poids_b[n]) for n in poids_a)  # Vérifie l'égalité à même graine.
    assert any(not torch.equal(poids_a[n], poids_c[n]) for n in poids_a)  # Vérifie la différence entre graines.


def test_instabilite_numerique_detectee(cfg, splits, tmp_path):  # Vérifie la détection de NaN.
    """Des valeurs NaN répétées doivent interrompre l'entraînement sans modifier les poids."""
    train, val = splits  # Récupère les sous-ensembles.
    cfg = deep_update(cfg, {"training": {"max_bad_steps": 1}})  # Réduit la tolérance.
    entraineur = Trainer(cfg, get_variant("B1"), train, val, tmp_path / "n", 1, torch.device("cpu"))  # Prépare l'entraînement.
    avant = {n: p.clone() for n, p in entraineur.modele.named_parameters()}  # Copie les poids.
    lot = next(train.iter_train_batches(8, np.random.default_rng(0), torch.device("cpu")))  # Récupère un lot.
    lot["image_tokens"] = lot["image_tokens"] * float("nan")  # Introduit des NaN.
    assert entraineur._etape(lot, 1) is None  # Vérifie que l'étape est ignorée.
    assert all(torch.equal(avant[n], p) for n, p in entraineur.modele.named_parameters())  # Vérifie que les poids sont intacts.
    with pytest.raises(NumericalInstabilityError):  # Attend l'interruption.
        entraineur._etape(lot, 1)  # Répète l'étape invalide.


def test_config_invalide_message_utile():  # Vérifie les messages d'erreur.
    """Une configuration invalide doit produire un message qui nomme le problème."""
    with pytest.raises(ConfigError, match="num_shared_tokens"):  # Attend une erreur explicite.
        load_config("config.yaml", {"model": {"num_shared_tokens": 24}})  # Essaie une valeur invalide.
    with pytest.raises(ConfigError, match="lr"):  # Attend une erreur explicite.
        load_config("config.yaml", {"training": {"lr": "beaucoup"}})  # Essaie une valeur non numérique.
    with pytest.raises(ConfigError, match="introuvable"):  # Attend une erreur explicite.
        load_config("absent.yaml")  # Essaie un fichier absent.
    assert isinstance(load_config("config.yaml", {"training": {"lr": "1e-4"}})["training"]["lr"], float)  # Vérifie la conversion d'une chaîne numérique.


def test_variante_inconnue():  # Vérifie le message d'erreur des variantes.
    """Une variante inconnue doit produire une erreur qui liste les noms valides."""
    with pytest.raises(KeyError, match="B4"):  # Attend une erreur explicite.
        get_variant("B9")  # Essaie une variante inconnue.
