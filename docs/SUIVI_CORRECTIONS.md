# Suivi des corrections

Chaque défaut de `docs/RAPPORT_AUDIT.md` est relié à sa correction et, lorsqu'il existe, au test qui l'empêche de revenir. Les fichiers de test sont dans `tests/`.

## CRITICAL

| ID | Correction | Test associé |
|---|---|---|
| C1 | Audio et vidéo retirés du flux. Seules les modalités réelles sont reçues. Activer l'audio ou la vidéo dans la configuration est refusé | `test_modalites_audio_video_absentes_du_modele`, `test_config_refuse_audio_video_sans_donnees`, `test_aucune_generation_aleatoire_de_modalite_dans_le_pipeline` |
| C2 | Repli synthétique supprimé ; absence de données = erreur explicite | `test_chargeur_refuse_de_continuer_sans_donnees_reelles` |
| C3 | Chargeur réécrit pour le jeu cible, images résolues depuis COCO train2014 | `test_chargeur_recupere_la_ligne_prise_pour_en_tete`, `test_manifeste_sans_fuite_et_correspondance_preservee` |
| C4 | Découpage train/val/test par hachage de l'image ; test évalué une seule fois, après sélection sur la validation | `test_decoupage_deterministe_et_par_image`, `test_detecteur_de_fuite` |
| C5 | La voie structurelle entre dans l'embedding par une porte apprise ; tous les modules reçoivent un gradient | `test_tous_les_modules_recoivent_un_gradient`, `test_embedding_depend_de_la_voie_topologique` |
| C6 | Couche de persistance différentiable : appariement exact, valeurs relues dans le tenseur. Perte de signature à la place de la MSE entre tokens | `test_gradient_autograd_coincide_avec_differences_finies`, `test_encodeur_topologique_transmet_le_gradient_des_diagrammes`, `test_gradient_de_la_perte_top_traverse_la_persistance` |
| C7 | Rips pondéré sur métrique 1-lipschitzienne, SVD exacte, seuil supprimé, borne démontrée | `test_borne_de_stabilite_verifiee_numeriquement`, `test_norme_spectrale_exacte_bornee_par_un`, `test_stabilite_des_tokens_respecte_la_borne` |
| C8 | README réécrit à partir de ce qui est réellement exécuté | Relecture manuelle |
| C9 | Adaptateur documenté et inactif sans clé ; Muse Spark n'est pas revendiqué dans les résultats. DINOv2 de Meta est utilisé localement | `test_sans_cle_l_adaptateur_refuse_explicitement`, `test_embeddings_internes_refuses_explicitement`, `test_integration_meta_inactive_par_defaut` |

## HIGH

| ID | Correction | Test associé |
|---|---|---|
| H1 | Encodeurs gelés maintenus en mode évaluation, caractéristiques en cache | Vérifié par sonde : mode `eval` conservé après `train()` |
| H2 | Masque de padding transmis à la banque | `test_banque_tokens_orthogonaux_et_masque_padding` |
| H3 | Corrélation de Mantel entrée-latent et image-texte, sans la diagonale | `test_perte_geometrique_penalise_l_effondrement`, `test_correlation_mantel_invariante_a_l_echelle` |
| H4 | Perte temporelle retirée, vidéo laissée en extension expérimentale | `test_encodeurs_experimentaux_exigent_de_vraies_entrees` |
| H5 | Température déplacée dans le modèle, donc dans l'optimiseur | `test_tous_les_modules_recoivent_un_gradient` |
| H6 | `L_orth` et `L_rec` actifs, projecteur privé par modalité | `test_factorisation_et_penalite_orthogonalite`, `test_tous_les_modules_recoivent_un_gradient` |
| H7 | Graines, planning, points de contrôle, arrêt précoce, détection de NaN | `test_entrainement_reproductible`, `test_planning_du_taux_d_apprentissage`, `test_point_de_controle_restitue_le_modele`, `test_instabilite_numerique_detectee` |
| H8 | Score d'isométrie remplacé par Mantel hors diagonale, stress, bottleneck et Wasserstein | `test_ancien_faux_score_d_isometrie_n_existe_plus`, `test_metriques_de_structure` |
| H9 | Matrice de pertinence, deux directions, AP réelle, départage aléatoire des égalités, bootstrap | `test_map_avec_plusieurs_pertinents`, `test_retrieval_matrice_analytique`, `test_scores_egaux_ne_donnent_pas_un_score_parfait`, `test_bootstrap_apparie` |
| H10 | Mort infinie conservée, aucun point fictif, grille adaptée à chaque échantillon | `test_classes_essentielles_restent_infinies`, `test_h0_deux_amas_analytique` |
| H11 | `pytest.ini` avec `pythonpath = .` | La commande `pytest -q` elle-même |
| H12 | Suite réécrite : cas analytiques, comparaison à Ripser, différences finies | `tests/test_topology_math.py` |
| H13 | Découpage par image, toutes les légendes d'une image dans le même sous-ensemble | `test_manifeste_sans_fuite_et_correspondance_preservee` |
| H14 | Rips pondéré nommé correctement ; échelles utilisées par les courbes de Betti | `test_encodeur_topologique_transmet_le_gradient_des_diagrammes`, `test_filtration_monotone_et_positive` |
| H15 | Colonne d'étiquettes ignorée | `test_colonne_etiquettes_jamais_utilisee_comme_texte` |

## MEDIUM

| ID | Correction | Test associé |
|---|---|---|
| M1 | `orthogonal_` appliqué à la matrice `(K, d)` | `test_banque_tokens_orthogonaux_et_masque_padding` |
| M2 | Complément avant le tri par `topk` | `test_paysage_trie_meme_avec_peu_de_points` |
| M3 | Insertion par lots, calcul unique, projection du graphe, encodeurs expérimentaux non chargés | Coûts mesurés dans `docs/RESULTATS.md` |
| M4 | Encodeur vidéo placé sur le device de l'entrée | Non exécuté : aucune vraie vidéo disponible |
| M5 | Contrat de forme strict pour l'audio et la vidéo | `test_encodeurs_experimentaux_exigent_de_vraies_entrees` |
| M6 | Validation de configuration avec messages explicites | `test_config_invalide_message_utile` |
| M7 | Phases marquées sur les courbes ; terme `L_sem` tracé séparément | Figure `02_courbes_perte.png` |
| M8 | Lots assemblés sans `DataLoader` ni fonction locale | Non testé sous Windows : voir les limitations |
| M9 | Projection commune aux deux modalités ; anciennes figures archivées dans `docs/legacy_outputs_documentvqa/` | Figures `04` et `05` |
| M10 | Incertitude renommée et dotée d'une vraisemblance hétéroscédastique | `test_tous_les_modules_recoivent_un_gradient` |
| M11 | Validation et comptage des rejets ; faux négatifs exclus ; pertinence partagée | `test_chargeur_rejette_et_compte_les_donnees_invalides`, `test_texte_partage_pertinent_pour_chaque_image`, `test_info_nce_exclut_les_faux_negatifs` |
| M12 | Perturbations sur les deux modalités ; modalité absente exclue de la fusion | `test_mode_modalite_absente_exclut_la_modalite`, `test_robustesse_conserve_au_moins_une_modalite_et_un_token` |

## LOW

| ID | Correction |
|---|---|
| L1 | Versions fixées d'après l'environnement réel ; `networkx`, `pandas` et `seaborn` retirés, `ripser` désormais réellement utilisé |
| L2 | Projections redondantes retirées |
| L3 | Importations nettoyées lors de la réécriture |
| L4 | Inverse exact de softplus |
| L5 | Device réel journalisé au démarrage de chaque expérience |
| L6 | Référence `docs/archi.png` corrigée dans le README |

## Défauts découverts après l'audit initial

Ces défauts sont apparus en exécutant le pipeline réécrit, et non en lisant le code d'origine. Aucun n'a été détecté à partir des métriques de test.

| Défaut observé | Cause | Correction | Vérification |
|---|---|---|---|
| B3 dégradé en début d'entraînement : R@5 de validation à 8,8 % à l'époque 1 | La voie structurelle ajoutait un même vecteur à tous les embeddings | Dernière couche de la tête structurelle initialisée à zéro et sans biais, descripteurs standardisés par vue | `test_embedding_depend_de_la_voie_topologique` et `test_tous_les_modules_recoivent_un_gradient`, après une vraie étape d'optimisation. L'absence de dégradation initiale a été contrôlée sur la validation |
| Norme totale du gradient à 525, contre 6 attendu | Division par la plage du paysage et BatchNorm sur une variance presque nulle | Division retirée, plancher de variance `eps=2,5e-3` ; gradient mesuré à 6,4 ensuite, `L_sem` identique à B2 | `test_amplification_du_gradient_bornee_par_l_encodeur` |
| Paysage aveugle aux barres H1 courtes | Grille fixe trop large pour la persistance réelle des nuages latents | Grille adaptée à chaque échantillon | `test_grille_adaptative_voit_une_barre_courte`, `test_grille_adaptative_independante_du_lot` |
| GUDHI ne renvoie aucune paire pour un point unique | Cas limite de `flag_persistence_generators` | Retour explicite si le nuage compte au plus un point | `test_paires_de_petits_nuages` |
| Perte de robustesse trop coûteuse à l'entraînement | Calcul de la vue perturbée sur le lot entier | Sous-lot de `robust_points` points, sans changer la nature de la perte | `test_robustesse_sur_sous_lot_transmet_le_gradient`, coûts dans `docs/RESULTATS.md` |
| Mesure du coût topologique sur un nuage aléatoire | Reliquat de prototypage | Mesure sur de vrais tokens | Relecture de `measure_costs` |
| Réutilisation possible de résultats obsolètes | Présence d'un fichier de résultat jugée suffisante | Contrôle de `empreinte_config` avant réutilisation | `test_resultats_reutilises_seulement_avec_la_meme_configuration` |
| Reprise après interruption non vérifiée de bout en bout | État complet sauvegardé, mais continuité jamais comparée à une exécution ininterrompue | Aucune correction de code nécessaire : la continuité est désormais testée | `test_reprise_identique_a_une_execution_continue` |
| Durée d'entraînement fausse après une reprise ou une pause de la machine | La durée de `fit` ne couvrait que le segment suivant la reprise, et une époque suspendue gonflait la somme | Durée calculée depuis l'historique complet : durée médiane d'une époque multipliée par le nombre d'époques, avec le compte des époques écartées | Relecture de `docs/RESULTATS.md`, 7 époques écartées |
| `resource` absent sous Windows | Module réservé à Unix | Repli par `psutil` | Non exécuté sous Windows |
