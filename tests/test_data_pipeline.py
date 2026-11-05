# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Vérifie le chargement, la validation, le découpage, la correspondance et les corruptions."""

import numpy as np  # Crée les images de test.
import pytest  # Fournit le cadre de test.
from PIL import Image  # Crée les images de test.

from data.corruptions import char_corruption, gaussian_blur, gaussian_noise, list_scenarios, random_mask, token_deletion  # Importe les corruptions.
from data.dataset_loader import CaptionTable, DatasetError, ImageStore, assign_split, build_manifest, image_id_from_file, load_caption_table, select_candidates, verify_no_leakage  # Importe le chargeur.


def _source_factice(lignes, colonnes=("image", "etiquettes", "legende")):  # Fabrique une source de lignes.
    """Retourne une source de lignes pour tester la validation sans réseau."""
    return lambda _: (iter(lignes), tuple(colonnes))  # Retourne la fonction attendue par le chargeur.


def _fichier(i):  # Fabrique un nom de fichier valide.
    """Retourne un nom de fichier COCO valide."""
    return f"COCO_train2014_{i:012d}.jpg"  # Formate le nom de fichier.


def test_chargeur_recupere_la_ligne_prise_pour_en_tete(cfg):  # Vérifie la récupération de la première ligne.
    """La première ligne du CSV sert d'en-tête au chargeur CSV et ne doit pas être perdue."""
    colonnes = (_fichier(9), "plate rack", "colorful dishes")  # Simule l'en-tête mal interprété.
    table = load_caption_table(cfg, _source_factice([(_fichier(25), "gazelle", "two giraffes")], colonnes))  # Charge la table.
    assert table.captions_by_image[_fichier(9)] == ["colorful dishes"]  # Vérifie la récupération.
    assert table.captions_by_image[_fichier(25)] == ["two giraffes"]  # Vérifie la ligne suivante.
    assert table.report["en_tete_recupere"] is True  # Vérifie le bilan.


def test_chargeur_rejette_et_compte_les_donnees_invalides(cfg):  # Vérifie la validation.
    """Une donnée invalide est exclue et comptée, jamais remplacée."""
    longue = " ".join(["mot"] * 500)  # Crée une légende anormalement longue.
    lignes = [("mauvais.jpg", "x", "ok"), (_fichier(1), "x", None), (_fichier(2), "x", "   "), (_fichier(3), "x", longue), (_fichier(4), "x", "bonne legende"), (_fichier(4), "x", "bonne legende")]  # Crée des lignes défectueuses.
    table = load_caption_table(cfg, _source_factice(lignes))  # Charge la table.
    assert table.report["rejets"] == {"nom_fichier_invalide": 1, "legende_vide": 2, "legende_trop_longue": 1, "doublon_image_legende": 1}  # Vérifie le comptage.
    assert list(table.captions_by_image) == [_fichier(4)]  # Vérifie que seule la bonne image reste.


def test_chargeur_refuse_de_continuer_sans_donnees_reelles(cfg):  # Vérifie l'absence de repli synthétique.
    """Sans donnée valide, le chargeur doit échouer au lieu de générer un repli synthétique."""
    with pytest.raises(DatasetError):  # Attend une erreur explicite.
        load_caption_table(cfg, _source_factice([("mauvais.jpg", "x", "y")]))  # Charge une source sans ligne valide.


def test_colonne_etiquettes_jamais_utilisee_comme_texte(cfg):  # Vérifie l'absence de fuite des étiquettes.
    """Les étiquettes visuelles viennent de l'image : elles ne doivent jamais entrer dans le texte."""
    table = load_caption_table(cfg, _source_factice([(_fichier(5), "etiquette_derivee_image", "vraie legende")]))  # Charge une ligne.
    assert table.captions_by_image[_fichier(5)] == ["vraie legende"]  # Vérifie le texte retenu.
    assert table.report["colonne_etiquettes_utilisee"] is False  # Vérifie le bilan.


def test_identifiant_image():  # Vérifie l'extraction de l'identifiant.
    """L'identifiant COCO doit être extrait, et un nom invalide refusé."""
    assert image_id_from_file(_fichier(36)) == 36  # Vérifie un nom valide.
    with pytest.raises(DatasetError):  # Attend une erreur explicite.
        image_id_from_file("COCO_val2014_000000000036.jpg")  # Essaie un autre sous-ensemble COCO.


def test_decoupage_deterministe_et_par_image():  # Vérifie le découpage.
    """Le découpage doit être déterministe, et ses proportions proches des fractions demandées."""
    fichiers = [_fichier(i) for i in range(5000)]  # Crée une liste de fichiers.
    a = [assign_split(1, f, (0.8, 0.1, 0.1)) for f in fichiers]  # Calcule un premier découpage.
    b = [assign_split(1, f, (0.8, 0.1, 0.1)) for f in fichiers]  # Calcule un second découpage.
    assert a == b  # Vérifie le déterminisme.
    assert a != [assign_split(2, f, (0.8, 0.1, 0.1)) for f in fichiers]  # Vérifie l'effet de la graine.
    assert abs(a.count("train") / 5000 - 0.8) < 0.03  # Vérifie la proportion d'entraînement.
    assert abs(a.count("test") / 5000 - 0.1) < 0.03  # Vérifie la proportion de test.


def _table_et_store(cfg, tmp_path, nombre):  # Prépare une table et un cache d'images.
    """Prépare une table de légendes et un cache local d'images de test."""
    cfg["dataset"].update({"cache_dir": str(tmp_path), "num_train_images": 8, "num_val_images": 2, "num_test_images": 2, "image_cache_size": 64, "train_captions_per_image": 2, "eval_captions_per_image": 5})  # Réduit les quotas.
    table = CaptionTable({_fichier(i): [f"legende {i} {j}" for j in range(5)] for i in range(nombre)})  # Crée la table.
    store = ImageStore(cfg)  # Crée le cache d'images.
    for fichier in table.captions_by_image:  # Parcourt les fichiers.
        Image.new("RGB", (80, 80), (10, 20, 30)).save(store.chemin(fichier), format="JPEG")  # Écrit une image de test.
    return table, store  # Retourne la table et le cache.


def test_manifeste_sans_fuite_et_correspondance_preservee(cfg, tmp_path):  # Vérifie le manifeste.
    """Les sous-ensembles doivent être disjoints, et chaque légende rester associée à son image."""
    table, store = _table_et_store(cfg, tmp_path, 200)  # Prépare les données de test.
    manifeste = build_manifest(table, store, cfg)  # Construit le manifeste.
    assert [len(manifeste["splits"][n]) for n in ("train", "val", "test")] == [8, 2, 2]  # Vérifie les quotas.
    for nom in ("train", "val", "test"):  # Parcourt les sous-ensembles.
        for entree in manifeste["splits"][nom]:  # Parcourt les entrées.
            numero = image_id_from_file(entree["file"])  # Lit l'identifiant de l'image.
            assert all(legende.startswith(f"legende {numero} ") for legende in entree["captions"])  # Vérifie la correspondance image-texte.
    assert all(len(e["captions"]) == 2 for e in manifeste["splits"]["train"])  # Vérifie le nombre de légendes d'entraînement.
    assert all(len(e["captions"]) == 5 for e in manifeste["splits"]["test"])  # Vérifie le nombre de légendes de test.


def test_image_corrompue_exclue_et_remplacee_par_une_vraie_image(cfg, tmp_path):  # Vérifie la gestion des images corrompues.
    """Une image illisible et non téléchargeable est exclue, et remplacée par une image réelle de la réserve."""
    table, store = _table_et_store(cfg, tmp_path, 400)  # Prépare les données de test.
    store.gabarit = "http://127.0.0.1:9/{file}"  # Rend le téléchargement impossible.
    store._lire_octets = lambda nom: (_ for _ in ()).throw(DatasetError(f"indisponible : {nom}"))  # Évite l'attente réseau.
    victime = select_candidates(table, cfg)["test"][0]  # Choisit une image de test.
    store.chemin(victime).write_bytes(b"pas une image")  # Corrompt le fichier.
    manifeste = build_manifest(table, store, cfg)  # Construit le manifeste.
    fichiers = [e["file"] for e in manifeste["splits"]["test"]]  # Lit les fichiers retenus.
    assert victime not in fichiers and len(fichiers) == 2  # Vérifie l'exclusion et le remplacement.
    assert manifeste["meta"]["images_indisponibles"] == 1  # Vérifie le comptage.


def test_detecteur_de_fuite():  # Vérifie le détecteur de fuite.
    """Une image présente dans deux sous-ensembles doit interrompre le protocole."""
    entree = {"file": _fichier(1)}  # Crée une entrée.
    with pytest.raises(DatasetError):  # Attend une erreur explicite.
        verify_no_leakage({"splits": {"train": [entree], "val": [], "test": [entree]}})  # Simule une fuite.


def test_quota_impossible_explique(cfg, tmp_path):  # Vérifie le message d'erreur des quotas.
    """Un quota impossible doit produire une erreur explicite."""
    table, _ = _table_et_store(cfg, tmp_path, 5)  # Prépare trop peu d'images.
    with pytest.raises(DatasetError, match="Quota impossible"):  # Attend une erreur explicite.
        select_candidates(table, cfg)  # Sélectionne les candidats.


def test_corruptions_image_deterministes_et_effectives():  # Vérifie les corruptions d'image.
    """Chaque corruption doit modifier l'image et rester reproductible."""
    pixels = np.random.default_rng(0).integers(0, 255, size=(96, 96, 3), dtype=np.uint8)  # Crée une image de test.
    image = Image.fromarray(pixels)  # Convertit en image PIL.
    for fonction, parametre in ((gaussian_noise, 0.2), (gaussian_blur, 2.0), (random_mask, 0.5)):  # Parcourt les corruptions.
        a = np.asarray(fonction(image, parametre, 7, "cle"))  # Applique la corruption une première fois.
        b = np.asarray(fonction(image, parametre, 7, "cle"))  # Applique la corruption une seconde fois.
        assert (a == b).all()  # Vérifie la reproductibilité.
        assert (a != pixels).any()  # Vérifie que l'image est modifiée.
        assert a.shape == pixels.shape  # Vérifie la conservation de la forme.


def test_masque_proportion_exacte():  # Vérifie la proportion de masquage.
    """Le masquage doit toucher la fraction demandée de blocs."""
    image = Image.new("RGB", (96, 96), (255, 255, 255))  # Crée une image blanche.
    resultat = np.asarray(random_mask(image, 0.25, 1, "x", cote_bloc=32))  # Applique le masque.
    assert abs((resultat[..., 0] == 127).mean() - 2.0 / 9.0) < 1e-6  # Vérifie la fraction masquée, arrondie à deux blocs sur neuf.


def test_corruptions_texte():  # Vérifie les corruptions de texte.
    """Les corruptions de texte doivent être reproductibles, et conserver au moins un mot."""
    texte = "a man riding a wave on a surfboard"  # Définit une légende.
    assert token_deletion(texte, 0.4, 1, "k") == token_deletion(texte, 0.4, 1, "k")  # Vérifie la reproductibilité.
    assert len(token_deletion(texte, 1.0, 1, "k").split()) == 1  # Vérifie la conservation d'un mot.
    assert char_corruption(texte, 0.0, 1, "k") == texte  # Vérifie le cas sans corruption.
    assert char_corruption(texte, 0.5, 1, "k") != texte  # Vérifie que le texte change.
    assert len(char_corruption(texte, 0.5, 1, "k")) == len(texte)  # Vérifie la conservation de la longueur.


def test_scenarios_configuration(cfg):  # Vérifie la liste des scénarios.
    """Tous les scénarios demandés doivent exister, et un nom inconnu être refusé."""
    scenarios = list_scenarios(cfg)  # Énumère les scénarios.
    noms = {nom for _, nom, _ in scenarios}  # Lit les noms des scénarios.
    assert noms == {"image_gaussian_noise", "image_blur_radius", "image_random_mask", "text_token_deletion", "text_char_corruption"}  # Vérifie la couverture.
    cfg["corruptions"]["inconnue"] = [1.0]  # Ajoute un scénario inconnu.
    with pytest.raises(ValueError):  # Attend une erreur explicite.
        list_scenarios(cfg)  # Énumère les scénarios.


def test_split_relevance_multi_legendes(splits):  # Vérifie la matrice de pertinence.
    """La pertinence doit relier chaque image à toutes ses légendes."""
    train, _ = splits  # Récupère l'entraînement.
    matrice = train.relevance_matrix()  # Construit la matrice.
    assert matrice.shape == (32, 64)  # Vérifie la forme.
    assert matrice.sum() == 64 and (matrice.sum(axis=1) == 2).all()  # Vérifie le nombre de légendes pertinentes.
    assert matrice[3, train.image_de_legende == 3].all()  # Vérifie la correspondance.


def test_texte_partage_pertinent_pour_chaque_image(tmp_path):  # Vérifie la gestion des légendes partagées.
    """Une légende identique doit être pertinente pour toutes les images qu'elle décrit."""
    from data.feature_cache import FeatureSplit  # Importe l'accès aux caractéristiques.

    np.save(tmp_path / "s.image.npy", np.zeros((2, 3, 4), dtype=np.float16))  # Écrit les images de test.
    np.save(tmp_path / "s.text.npy", np.zeros((2, 3, 4), dtype=np.float16))  # Écrit les textes de test.
    np.save(tmp_path / "s.text_mask.npy", np.ones((2, 3), dtype=bool))  # Écrit les masques de test.
    entrees = [{"file": "a", "image_id": 1, "captions": ["même texte"]}, {"file": "b", "image_id": 2, "captions": ["même texte"]}]  # Crée deux images partageant un texte.
    split = FeatureSplit(tmp_path, entrees, "s")  # Ouvre le sous-ensemble.
    assert split.relevance_matrix().all()  # Vérifie que le texte reste pertinent pour les deux images.
    assert split.id_texte[0] == split.id_texte[1]  # Vérifie l'identifiant commun utilisé pour exclure les faux négatifs.


def test_lots_entrainement_une_image_une_fois_par_epoque(splits):  # Vérifie les lots d'entraînement.
    """Une image apparaît une fois par époque, avec une de ses propres légendes."""
    import torch  # Importe PyTorch pour le device.

    train, _ = splits  # Récupère l'entraînement.
    generateur = np.random.default_rng(0)  # Crée le générateur.
    lots = list(train.iter_train_batches(8, generateur, torch.device("cpu")))  # Parcourt une époque.
    assert len(lots) == 4  # Vérifie le nombre de lots.
    assert lots[0]["image_tokens"].shape == (8, 5, 16)  # Vérifie la forme des images.
    assert lots[0]["text_tokens"].shape == (8, 6, 16)  # Vérifie la forme des textes.
    assert lots[0]["text_mask"].dtype == torch.bool  # Vérifie le type du masque.
    correlation = float(np.corrcoef(lots[0]["image_tokens"].mean(1).flatten().numpy(), lots[0]["text_tokens"][:, :4].mean(1).flatten().numpy())[0, 1])  # Mesure le lien entre image et texte du lot.
    assert correlation > 0.9  # Vérifie que l'appariement est préservé par le lot.
