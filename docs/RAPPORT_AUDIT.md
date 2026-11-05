# Rapport d'audit technique et scientifique — TopoFusion-TR Hyper

**Date de l'audit initial :** 5 novembre 2026
**Périmètre :** code tel que reçu (`main.py`, `config.yaml`, `data/`, `models/`, `losses/`, `training/`, `utils/`, `tests/`, `README.md`, `requirements.txt`).
**Méthode :** lecture intégrale du code, puis sondes numériques exécutées sur le code d'origine, sans le modifier. Les valeurs indiquées ci-dessous sont mesurées, pas estimées.

Ce rapport décrit l'état **avant correction**. Le suivi des corrections se trouve à la fin du document, section 7.

---

## 1. Synthèse

Le dépôt d'origine exécute un entraînement et produit des courbes, mais ces courbes ne mesurent pas ce qu'elles prétendent mesurer :

| Constat mesuré | Valeur | Conséquence |
|---|---|---|
| Gradients reçus par `topo_attention_block` (phases 1, 2 et 3) | **0 paramètre sur 17** | L'attention topologique n'apprend jamais |
| Gradients reçus par `structural_fusion` (phases 1, 2 et 3) | **0 paramètre sur 4** | L'espace structurel commun n'apprend jamais |
| Chemin de l'embedding de recherche | `global_embed_head(s_img.mean(1))` | La topologie n'entre pas dans le Recall rapporté |
| `requires_grad` des paysages de persistance | **False** | Aucun gradient ne provient de la persistance |
| Jeu de validation | **le même objet `DataLoader` que l'entraînement** | Le Recall affiché est un Recall d'entraînement |
| Poids de fusion moyens (image, texte, audio, vidéo) | 0,301 / 0,228 / **0,170 / 0,301** | Près de la moitié de la fusion provient de bruit `torch.randn` |
| `L_geom` sur une représentation totalement effondrée | **0,0** | La perte géométrique récompense l'effondrement |
| `topological_isometry_score` sur deux nuages **indépendants** | **0,8915** | La métrique est gonflée par les zéros de la diagonale |
| Norme spectrale exacte d'une couche « 1-Lipschitz » | **1,0033** | La garantie annoncée est fausse à itération finie |
| Paramètres gelés de l'encodeur AST inutilisé | 86 187 264 | Mémoire gaspillée pour un encodeur jamais appelé |
| Écart entre deux passes de BERT « gelé » | 0,276 | Le dropout reste actif dans les encodeurs gelés |

**Conclusion :** les figures fournies dans `outputs/` ne constituent pas une preuve expérimentale et ne doivent pas être présentées comme telles.

### Trois corrections à apporter au cahier des charges

L'audit contredit trois hypothèses de départ. Elles sont signalées ici parce qu'elles changent le travail à réaliser.

1. **`AhmedSSabir/Textual-Image-Caption-Dataset` ne contient aucune image.** Le dépôt fournit des CSV à trois colonnes : nom de fichier COCO, étiquettes visuelles et légende. Pour obtenir de vraies paires image-texte, il faut résoudre chaque nom de fichier vers l'image COCO train2014 correspondante.
2. **L'appel exigé échoue avec `datasets` 5.0.1.** `load_dataset("AhmedSSabir/Textual-Image-Caption-Dataset", split="train")` lève `DatasetGenerationCastError`, car les fichiers du dépôt n'ont pas les mêmes colonnes. Le chargeur doit tenter cet appel, puis se rabattre sur `data_files="All_Caption_ID.csv"`. La première ligne de données sert en outre d'en-tête et doit être récupérée.
3. **Le dépôt ne contient aucune intégration Meta.** Une recherche de `muse`, `spark`, `llama` ou `api.meta` ne renvoie aucun résultat. L'API officielle existe, mais sa documentation n'expose ni embeddings ni états cachés (voir `docs/INTEGRATION_META.md`).

### Statistiques du jeu de données cible (mesurées)

| Mesure | Valeur |
|---|---|
| Lignes de `All_Caption_ID.csv` (hors première ligne prise pour un en-tête) | 413 387 |
| Images COCO train2014 distinctes | 82 783 |
| Légendes par image | 5 (82 255 images), 4 (528 images) |
| Noms de fichiers non conformes | 0 |
| Légendes vides | 1 |
| Longueur des légendes en mots | moyenne 10,47 ; p99 = 19 ; **max = 6 535** |
| Couples (image, légende) dupliqués | 133 |
| Légendes identiques sur plusieurs lignes | 6 332 |

---

## 2. Priorité CRITICAL

| ID | Cat. | Fichier et fonction | Problème technique exact | Conséquence scientifique | Correction proposée |
|---|---|---|---|---|---|
| C1 | J | `data/dataset_loader.py`, `__getitem__` | `audio_features = torch.randn(128, 512, ...)` et `video_features = torch.randn(64, 1024, ...)` pour chaque échantillon | Du bruit traverse la banque latente et la fusion (17 % et 30 % du poids), alors que le README affirme le contraire | Supprimer ces modalités du flux, ajouter un dictionnaire de modalités disponibles et des masques |
| C2 | J | `DocumentVQADataset.__init__` et `__getitem__` | Repli silencieux vers `torch.randn(3,224,224)` et des identifiants aléatoires si le chargement échoue | Une expérience peut tourner entièrement sur du bruit sans erreur | Supprimer le repli et lever une erreur explicite |
| C3 | A, J | `dataset_loader.py` et `config.yaml` | Le code utilise `HuggingFaceM4/DocumentVQA` avec des questions comme texte, et `item.get('question', "What is shown...")` invente un texte par défaut | Mauvais jeu de données ; une question n'est pas une description, et plusieurs questions visent le même document | Réécrire le chargeur pour le jeu cible et résoudre les images COCO |
| C4 | F, G, H | `training/trainer.py`, `validate` ; `main.py` | `validate` parcourt `self.dataloader`, identique à l'entraînement ; aucun jeu de test n'existe | Toutes les métriques sont des métriques d'entraînement ; les courbes fournies sont invalides | Découpage train/val/test par image, test utilisé une seule fois |
| C5 | D, E | `models/multimodal_framework.py`, `forward` | `image_embed` et `text_embed` contournent la topologie ; `multimodal_embed` et `structural_tokens` n'entrent dans aucune perte | L'attention topologique reçoit 0 gradient sur 17 ; la contribution topologique affichée est nulle par construction | Faire passer la voie structurelle dans l'embedding, avec une porte résiduelle mesurable |
| C6 | L, D | `models/topology.py`, `TopologicalEncoder` ; `losses.py`, `TopologicalLoss` | Paysages calculés depuis NumPy, donc détachés. Le « pont différentiable » est une moyenne de tokens pondérée par softmax, sans lien avec la persistance. `L_top` est une MSE entre tokens | Le gradient ne vient jamais de l'homologie. La docstring affirme à tort qu'une distance L2 de paysages majore Wasserstein, alors que la stabilité va dans l'autre sens | Couche de persistance différentiable : appariement exact par GUDHI, relecture des valeurs dans le tenseur PyTorch |
| C7 | L | `topology.py`, `LearnedFiltration` et `compute_diagrams` | La filtration d'arête ajoute `1 - A_ij`, issu d'un réseau non contraint ; le seuil `weight > 0.05` crée une discontinuité ; la norme spectrale mesurée vaut 1,0033 | Le théorème de stabilité est invoqué hors de ses hypothèses | Filtration de Rips pondérée sur métrique apprise, norme spectrale exacte par SVD, borne démontrée et testée |
| C8 | K | `README.md` | Scripts inexistants (`scripts/visualiser_dataset.py`, `scripts/entrainer_baseline.py`), baselines B0-B3 inexistantes, métriques non implémentées, affirmation « n'invente pas de faux audio/vidéo » contredite par le code | Les affirmations ne sont pas vérifiables, et plusieurs sont fausses | Réécrire le README à partir de ce qui est réellement exécuté |
| C9 | K | Ensemble du dépôt | Aucune ligne de code ne concerne Meta, alors que la présentation mentionne Muse Spark | Risque de revendiquer une intégration inexistante | Adaptateur documenté, explicitement inactif sans clé, sans prétention d'intégration |

---

## 3. Priorité HIGH

| ID | Cat. | Fichier et fonction | Problème technique exact | Conséquence scientifique | Correction proposée |
|---|---|---|---|---|---|
| H1 | I, D | `encoders.py` ; `Trainer.train_epoch` | `model.train()` réactive le dropout de ViT et BERT gelés (écart mesuré : 0,276) | Les encodeurs « gelés » sont stochastiques et l'expérience n'est pas reproductible | Encodeurs toujours en `eval`, `no_grad`, caractéristiques mises en cache |
| H2 | E, B | `multimodal_framework.py`, `forward` | `self.latent_bank(modality_tokens['text'])` n'envoie jamais `attention_mask` | La banque d'attention lit les jetons `[PAD]` | Transmettre un masque de padding à chaque appel |
| H3 | L | `losses.py`, `GeometricLoss` | Aucune distance d'entrée n'est utilisée ; la perte compare deux espaces latents ; l'effondrement donne 0,0 ; la diagonale nulle biaise la normalisation | L'« isométrie » n'est pas définie et le minimum global est trivial | Corrélation de Mantel entrée-latent et image-texte sur le triangle supérieur ; l'effondrement est pénalisé |
| H4 | J, L | `TemporalLoss` ; `forward` | `L_temp` traite les tokens partagés d'une vidéo aléatoire comme une suite temporelle | La perte n'a aucun sens : l'indice d'un token n'est pas un temps, et l'entrée est du bruit | Retirer du protocole, conserver en extension expérimentale |
| H5 | D | `Trainer.__init__` ; `SemanticLoss` | La température appartient au critère, pas à l'optimiseur | La « température apprenable » reste figée à 0,07 | Déplacer le paramètre dans le modèle |
| H6 | D | `SharedPrivateFactorization` ; `CompositeMultimodalLoss` | `orthogonality_loss` n'est jamais appelée ; `shared_private` reçoit 4 gradients sur 8 | L'espace privé reste aléatoire ; la séparation partagé/privé n'existe pas | Ajouter `L_orth` et une reconstruction `L_rec` |
| H7 | I, N | `trainer.py`, `main.py`, `config.yaml` | Aucune graine, ni planificateur (`scheduler` et `warmup_ratio` ignorés), ni point de contrôle, ni arrêt précoce, ni détection de NaN | Expériences non reproductibles | Boucle d'entraînement reconstruite |
| H8 | H | `utils/metrics.py`, `topological_isometry_score` | Corrélation calculée en incluant la diagonale nulle, sur les deux premiers lots mélangés ; rien de topologique | 0,8915 sur deux nuages indépendants : la métrique ne mesure rien | La remplacer par Mantel, stress de Kruskal, bottleneck et Wasserstein exacts |
| H9 | H | `utils/metrics.py` | Le `mAP` est en réalité un MRR ; une seule direction ; `drop_last=True` et mélange pendant la validation ; suppose une correspondance 1-1 | Métriques incomplètes et incompatibles avec 5 légendes par image | Matrice de pertinence, deux directions, AP réelle, intervalles par bootstrap |
| H10 | L | `compute_diagrams` | Mort essentielle remplacée par `birth + 2.0` ; point fictif `(0,0)` pour un diagramme vide ; grille `[0,2]` tronquée | Diagrammes faux, y compris pour les diagnostics | Conserver `inf`, séparer les classes essentielles, grille adaptée à la borne des filtrations |
| H11 | N | Racine du dépôt | `pytest -q` échoue à la collecte avec `ModuleNotFoundError: No module named 'training'` | La commande demandée ne fonctionne pas | Ajouter `pytest.ini` avec `pythonpath = .` |
| H12 | N | `tests/` | `assert shared.grad is not None` passe via un chemin sans topologie ; aucune vérification de H0 ou H1 ; aucune passe avant complète ; aucun test des données | 11 tests réussis qui ne vérifient pas les propriétés annoncées | Nouvelle suite avec cas analytiques et vérification par différences finies |
| H13 | F, G | Jeu cible | Chaque image a 4 ou 5 légendes : un découpage par ligne placerait la même image en train et en test | Fuite de données massive | Découpage par hachage de l'identifiant d'image |
| H14 | L | `LearnedFiltration` ; `compute_diagrams` | La docstring annonce « lower-star », ce qui n'est pas implémenté ; `scale_deltas` a un gradient `None` (paramètre mort) | Description fausse et échelles multi-résolution fictives | Rips pondéré explicite ; échelles utilisées pour les courbes de Betti douces |
| H15 | F | Jeu cible | La colonne d'étiquettes vient de classifieurs appliqués à l'image | L'utiliser comme texte reviendrait à donner l'image au texte | Exclure la colonne du texte d'entrée et le documenter |

---

## 4. Priorité MEDIUM

| ID | Cat. | Fichier et fonction | Problème technique exact | Conséquence | Correction proposée |
|---|---|---|---|---|---|
| M1 | L | `SemanticLatentBank.__init__` | `orthogonal_` sur un tenseur `(1,K,d)` aplatit en une seule ligne : \|cos\| hors diagonale jusqu'à 0,2456 | Les tokens ne sont pas orthogonaux, contrairement au README | Initialiser sur `(K,d)` |
| M2 | L | `compute_landscapes_from_diagrams` | Si N < k, les tentes sont concaténées sans tri point par point | Ce ne sont pas des paysages au sens de Bubenik | Toujours trier après remplissage |
| M3 | M | `compute_diagrams` ; `AdaptiveLatentGraph` ; `AudioEncoder` | Persistance calculée deux fois ; boucles Python ; paires en dimension 4d (d=768) ; AST chargé sans être appelé | Coût inutile (297 M de paramètres, 14,8 M entraînables) | Insertion par lots, projection préalable, encodeurs expérimentaux chargés à la demande |
| M4 | C | `VideoEncoder._get_videomae` | Modèle attaché paresseusement, jamais déplacé vers le device | Plantage sur GPU et `state_dict` instable | Chargement explicite avec placement sur le device |
| M5 | B | `AudioEncoder.forward` | Heuristiques de forme ; la branche 512 dimensions ne sert qu'aux faux descripteurs | Traitement ambigu d'un vrai audio | Contrat de forme strict avec erreur explicite |
| M6 | N | `config.yaml` ; `_as_float` | PyYAML lit `1e-4` comme une chaîne ; plusieurs clés sont ignorées sans message | Configuration trompeuse | Validation avec erreurs utiles |
| M7 | H | `main.py` ; `plot_training_curves` | Courbes concaténées entre phases aux objectifs différents | Une baisse apparente peut venir du changement d'objectif | Tracer aussi les termes séparément et marquer les phases |
| M8 | N | `get_dataloader` | `collate_fn` défini dans une fonction : non sérialisable sous Windows si `num_workers > 0` | Plantage sous Windows | Fonction au niveau du module |
| M9 | H | `visualize_latent_space` ; `outputs/` | ACP ajustées séparément : axes non comparables ; figures héritées invalides | Lecture visuelle trompeuse | ACP commune ; archiver les anciennes figures comme obsolètes |
| M10 | L | `AdaptiveLatentGraph` | « Incertitude bayésienne » sans vraisemblance ni a priori | Terme usurpé ; σ n'a aucune sémantique | Terme hétéroscédastique explicite, renommé honnêtement |
| M11 | J, H | Jeu cible | Légende vide, légende de 6 535 mots, 133 doublons, 6 332 légendes partagées | Faux négatifs dans InfoNCE et les métriques | Validation, filtrage, exclusion des faux négatifs |
| M12 | E | `RobustnessPerturbationModule` | Perturbe l'image uniquement ; `apply_modality_dropout` jamais appelée ; modalité absente remplacée par des zéros qui restent dans le softmax | Robustesse partielle | Perturbation par masque, modalité exclue de la fusion |

---

## 5. Priorité LOW

| ID | Cat. | Fichier | Problème | Correction |
|---|---|---|---|---|
| L1 | N | `requirements.txt` | Doublon `PyYAML` et `pyyaml` ; `networkx`, `pandas` et `ripser` non utilisés ; versions non fixées | Fixer des versions testées |
| L2 | M | `SemanticLatentBank` | Projections Q, K, V redondantes avec celles de `MultiheadAttention` | Les retirer |
| L3 | N | Plusieurs modules | Importations inutilisées | Nettoyage |
| L4 | L | `LearnedFiltration` | `softplus(log(diff))` ne redonne pas les échelles annoncées | Utiliser l'inverse exact de softplus |
| L5 | N | `Trainer.__init__` | Repli silencieux sur CPU | Journaliser le device réel |
| L6 | N | Racine et `docs/` | `archi.png` dupliqué ; `docs/architecture.png` cité mais absent | Référence corrigée |

---

## 6. Couverture des catégories demandées

| Catégorie | Identifiants |
|---|---|
| A. Erreurs d'exécution | C3, H11, M4 |
| B. Dimensions de tenseurs | H2, M5 |
| C. Device et dtype | M4 |
| D. Gradient interrompu | C5, C6, H1, H5, H6 |
| E. Fusion incorrecte | C5, H2, M12 |
| F. Fuites de données | C4, H13, H15 |
| G. Découpage incorrect | C4, H13 |
| H. Métriques non valides | C4, H8, H9, M7, M9, M11 |
| I. Non-reproductibilité | H1, H7 |
| J. Modalités artificielles | C1, C2, C3, H4, M11 |
| K. Affirmations non vérifiées | C8, C9 |
| L. Incohérences mathématiques | C6, C7, H3, H4, H10, H14, M1, M2, M10, L4 |
| M. Goulots d'étranglement | M3, L2 |
| N. Tests et erreurs | H7, H11, H12, M6, M8, L1, L3, L5, L6 |

**Note sur le device :** la température de `SemanticLoss` reste sur CPU, mais c'est un scalaire de dimension 0, ce que PyTorch autorise. Ce point ne provoque donc pas de plantage sur GPU ; le vrai défaut est H5.

---

## 7. État après correction

Voir [`docs/SUIVI_CORRECTIONS.md`](SUIVI_CORRECTIONS.md), qui indique pour chaque identifiant la correction apportée et le test associé.
