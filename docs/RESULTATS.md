# Résultats expérimentaux

Ce fichier est généré par `scripts/aggregate_results.py`. Chaque valeur provient des fichiers de `outputs/runs/`.

## Protocole

- Test : 1000 images et 4992 légendes, jamais utilisées pour l'entraînement ou la sélection.
- Entraînement : 8000 images ; validation : 1000 images.
- Niveau du hasard : R@1 image→texte 0.10 %, texte→image 0.10 % ; R@5 image→texte 0.50 %, texte→image 0.50 %.
- Les valeurs sont données en %, sous la forme moyenne ± écart-type sur les graines.

## Recherche inter-modale sur le test

| Modèle | Description | Graines | R@1 moyen | R@5 moyen | mAP moyen | R@1 i→t | R@1 t→i |
|---|---|---|---|---|---|---|---|
| B0 | Projection naïve : alignement cosinus sans négatifs | 3 | 0.66 ± 0.22 | 2.13 ± 0.40 | 1.16 ± 0.11 | 1.00 ± 0.44 | 0.33 ± 0.07 |
| B1 | Alignement contrastif InfoNCE | 3 | 30.60 ± 0.72 | 64.75 ± 0.74 | 36.57 ± 0.19 | 34.27 ± 1.44 | 26.94 ± 0.43 |
| B2 | InfoNCE + isométrie géométrique approchée | 3 | 30.80 ± 0.84 | 64.77 ± 0.54 | 36.85 ± 0.14 | 33.87 ± 1.80 | 27.74 ± 0.40 |
| B3 | B2 + représentation topologique | 3 | 30.99 ± 0.90 | 64.88 ± 0.39 | 36.87 ± 0.15 | 34.17 ± 1.79 | 27.81 ± 0.03 |
| B4 | TopoFusion-TR Hyper complet | 3 | 33.20 ± 1.19 | 66.68 ± 0.57 | 38.80 ± 0.36 | 36.87 ± 1.90 | 29.54 ± 0.49 |
| B4_sans_L_top | B4 sans perte topologique | 2 | 33.39 ± 0.46 | 66.89 ± 0.23 | 38.88 ± 0.22 | 37.10 ± 0.85 | 29.68 ± 0.07 |
| B4_sans_tokens_topo | B4 sans tokens H0 et H1 | 2 | 32.67 ± 0.11 | 67.92 ± 0.10 | 39.44 ± 0.22 | 35.55 ± 0.64 | 29.79 ± 0.42 |
| B4_sans_attention_topo | B4 sans biais topologique de l'attention | 2 | 32.69 ± 0.57 | 67.12 ± 0.22 | 38.63 ± 0.18 | 36.00 ± 1.27 | 29.38 ± 0.13 |
| B4_sans_robustesse | B4 sans perturbation ni perte de robustesse | 2 | 33.28 ± 0.61 | 66.65 ± 0.80 | 38.85 ± 0.22 | 36.85 ± 0.92 | 29.72 ± 0.30 |
| B4_sans_topologie | Banque et fusion seules, sans aucune topologie | 3 | 32.78 ± 0.46 | 67.23 ± 0.29 | 39.28 ± 0.34 | 35.63 ± 0.64 | 29.93 ± 0.40 |
| CLIP | Référence pré-entraînée, sans entraînement : laion/CLIP-ViT-B-32-laion2B-s34B-b79K | - | 64.75 | 88.06 | 65.03 | 73.20 | 56.29 |

## Comparaisons appariées

Différence (A − B) en points de pourcentage, avec IC 95 % par bootstrap apparié sur les requêtes de test, sur les succès moyennés par graine. La mention « oui » indique un intervalle qui exclut 0. Cet intervalle reflète la variabilité des requêtes, pas celle des graines : à lire avec les écarts-types ci-dessus.

| A − B | ΔR@1 | ΔR@5 | ΔAP |
|---|---|---|---|
| B1 − B0 | +27.72 [+26.85, +28.67] oui | +60.73 [+59.70, +61.78] oui | +39.26 [+38.49, +40.07] oui |
| B2 − B1 | +0.60 [+0.15, +1.06] oui | +0.61 [+0.17, +1.03] oui | +0.62 [+0.37, +0.88] oui |
| B3 − B2 | +0.11 [-0.31, +0.53] non | +0.01 [-0.43, +0.42] non | +0.03 [-0.20, +0.27] non |
| B4 − B3 | +1.89 [+1.36, +2.40] oui | +1.79 [+1.29, +2.30] oui | +1.76 [+1.43, +2.07] oui |
| B4 − B1 | +2.60 [+2.05, +3.15] oui | +2.41 [+1.86, +2.93] oui | +2.41 [+2.08, +2.73] oui |
| B4 − B4_sans_topologie | -0.12 [-0.62, +0.33] non | -0.64 [-1.11, -0.19] oui | -0.58 [-0.86, -0.33] oui |
| B4 − B4_sans_L_top | -0.15 [-0.61, +0.29] non | -0.21 [-0.64, +0.21] non | -0.16 [-0.42, +0.11] non |
| B4 − B4_sans_tokens_topo | +0.01 [-0.60, +0.58] non | -1.17 [-1.73, -0.65] oui | -0.73 [-1.11, -0.37] oui |
| B4 − B4_sans_attention_topo | +0.28 [-0.24, +0.80] non | -0.23 [-0.70, +0.24] non | +0.04 [-0.27, +0.34] non |
| B4 − B4_sans_robustesse | -0.14 [-0.61, +0.30] non | +0.05 [-0.38, +0.50] non | -0.20 [-0.47, +0.07] non |
| B4 − CLIP | -28.35 [-29.64, -27.06] oui | -20.17 [-21.30, -19.07] oui | -24.16 [-25.17, -23.20] oui |
| B1 − CLIP | -30.95 [-32.23, -29.63] oui | -22.57 [-23.75, -21.45] oui | -26.57 [-27.56, -25.62] oui |

## Contribution des tokens topologiques à l'inférence (B4)

Remplacer les tokens H0 et H1 par des zéros sur le modèle B4 déjà entraîné donne R@1 = 32.77 ± 1.33, R@5 = 66.67 ± 0.83, mAP = 38.58 ± 0.41. Porte structurelle apprise : 0.253.

## Robustesse aux corruptions (test)

R@5 moyen absolu en %, puis dégradation relative en % entre parenthèses.

| Scénario | B1 | B3 | B4 | B4_sans_robustesse | B4_sans_topologie | CLIP |
|---|---|---|---|---|---|---|
| image_blur_radius_2.0 | 60.11 (7.2) | 60.26 (7.1) | 62.94 (5.6) | 63.40 (4.9) | 63.59 (5.4) | 84.06 (4.5) |
| image_blur_radius_4.0 | 40.47 (37.5) | 41.16 (36.6) | 42.97 (35.6) | 43.16 (35.2) | 43.26 (35.7) | 73.97 (16.0) |
| image_gaussian_noise_0.1 | 59.38 (8.3) | 59.49 (8.3) | 61.55 (7.7) | 61.77 (7.3) | 61.78 (8.1) | 82.74 (6.0) |
| image_gaussian_noise_0.2 | 45.19 (30.2) | 45.54 (29.8) | 48.36 (27.5) | 48.65 (27.0) | 48.07 (28.5) | 70.12 (20.4) |
| image_random_mask_0.25 | 55.82 (13.8) | 56.23 (13.3) | 58.98 (11.5) | 59.31 (11.0) | 59.66 (11.3) | 78.30 (11.1) |
| image_random_mask_0.5 | 36.41 (43.8) | 37.43 (42.3) | 39.85 (40.2) | 40.33 (39.5) | 41.74 (37.9) | 55.94 (36.5) |
| text_char_corruption_0.1 | 35.89 (44.6) | 36.76 (43.3) | 40.18 (39.7) | 39.49 (40.7) | 41.70 (38.0) | 69.27 (21.3) |
| text_char_corruption_0.2 | 17.34 (73.2) | 17.19 (73.5) | 19.37 (71.0) | 19.12 (71.3) | 20.78 (69.1) | 44.36 (49.6) |
| text_token_deletion_0.2 | 55.34 (14.5) | 55.62 (14.3) | 57.63 (13.6) | 57.84 (13.2) | 58.59 (12.8) | 79.90 (9.3) |
| text_token_deletion_0.4 | 42.30 (34.7) | 42.44 (34.6) | 43.76 (34.4) | 43.62 (34.6) | 45.05 (33.0) | 68.69 (22.0) |

## Modalités manquantes (B4)

Avec l'image seule, retrouver l'embedding fusionné complet donne R@1 = 99.20 ± 0.46 (cosinus 0.906). Avec le texte seul : R@1 = 97.10 ± 0.61 (cosinus 0.879). Aucune donnée n'est simulée : la modalité absente est exclue de la fusion.

## Structure géométrique et topologique (exacte, 256 images de test)

Plus la corrélation de Mantel est élevée, et plus le stress et les distances entre diagrammes sont faibles, mieux la structure est préservée.

| Mesure | B0 | B1 | B2 | B3 | B4 | B4_sans_L_top | B4_sans_tokens_topo | B4_sans_attention_topo | B4_sans_robustesse | B4_sans_topologie |
|---|---|---|---|---|---|---|---|---|---|---|
| Mantel entrée-latent image | 0.365 ± 0.007 | 0.680 ± 0.007 | 0.827 ± 0.002 | 0.819 ± 0.002 | 0.818 ± 0.002 | 0.818 ± 0.002 | 0.816 ± 0.003 | 0.819 ± 0.002 | 0.821 ± 0.003 | 0.816 ± 0.003 |
| Mantel entrée-latent texte | 0.675 ± 0.007 | 0.367 ± 0.006 | 0.677 ± 0.004 | 0.683 ± 0.004 | 0.693 ± 0.010 | 0.680 ± 0.000 | 0.697 ± 0.003 | 0.687 ± 0.002 | 0.697 ± 0.002 | 0.688 ± 0.003 |
| Mantel image-texte | 0.166 ± 0.070 | 0.597 ± 0.012 | 0.689 ± 0.005 | 0.673 ± 0.003 | 0.666 ± 0.005 | 0.674 ± 0.006 | 0.671 ± 0.005 | 0.664 ± 0.001 | 0.663 ± 0.009 | 0.681 ± 0.004 |
| Stress image | 0.176 ± 0.007 | 0.077 ± 0.001 | 0.075 ± 0.001 | 0.071 ± 0.001 | 0.066 ± 0.000 | 0.068 ± 0.000 | 0.065 ± 0.000 | 0.066 ± 0.001 | 0.065 ± 0.000 | 0.068 ± 0.001 |
| Stress texte | 0.145 ± 0.005 | 0.132 ± 0.001 | 0.100 ± 0.000 | 0.098 ± 0.000 | 0.094 ± 0.002 | 0.097 ± 0.000 | 0.093 ± 0.001 | 0.095 ± 0.000 | 0.093 ± 0.000 | 0.096 ± 0.000 |
| Voisins@10 image | 0.265 ± 0.012 | 0.594 ± 0.002 | 0.653 ± 0.003 | 0.651 ± 0.001 | 0.645 ± 0.005 | 0.643 ± 0.002 | 0.646 ± 0.007 | 0.646 ± 0.001 | 0.645 ± 0.001 | 0.640 ± 0.003 |
| Voisins@10 texte | 0.284 ± 0.006 | 0.347 ± 0.002 | 0.394 ± 0.004 | 0.403 ± 0.003 | 0.401 ± 0.004 | 0.390 ± 0.001 | 0.391 ± 0.002 | 0.396 ± 0.003 | 0.400 ± 0.002 | 0.385 ± 0.006 |
| W1 H0 image-texte | 7.415 ± 0.672 | 4.096 ± 0.417 | 8.518 ± 0.069 | 8.140 ± 0.689 | 6.793 ± 0.274 | 6.975 ± 0.040 | 6.687 ± 0.231 | 7.074 ± 0.198 | 7.083 ± 0.325 | 6.538 ± 0.333 |
| W1 H1 image-texte | 1.288 ± 0.345 | 3.265 ± 0.195 | 2.411 ± 0.125 | 2.563 ± 0.102 | 2.476 ± 0.105 | 2.561 ± 0.118 | 2.472 ± 0.231 | 2.506 ± 0.033 | 2.582 ± 0.074 | 2.462 ± 0.065 |
| Bottleneck H1 image-texte | 0.028 ± 0.008 | 0.037 ± 0.008 | 0.034 ± 0.003 | 0.039 ± 0.004 | 0.036 ± 0.005 | 0.035 ± 0.005 | 0.038 ± 0.016 | 0.033 ± 0.006 | 0.038 ± 0.001 | 0.036 ± 0.003 |

## Diagnostics des représentations

| Modèle | Rang effectif image | Rang effectif texte | Marge cosinus vraies-fausses paires | Porte structurelle | Effondrement détecté | Étapes NaN ignorées |
|---|---|---|---|---|---|---|
| B0 | 48.3 | 41.7 | 0.000 | 0.000 | oui | 0 |
| B1 | 25.5 | 22.7 | 0.613 | 0.000 | non | 0 |
| B2 | 21.3 | 21.1 | 0.595 | 0.000 | non | 0 |
| B3 | 22.8 | 22.1 | 0.586 | 0.248 | non | 0 |
| B4 | 25.5 | 24.6 | 0.591 | 0.253 | non | 0 |
| B4_sans_L_top | 24.1 | 23.6 | 0.601 | 0.253 | non | 0 |
| B4_sans_tokens_topo | 25.9 | 25.3 | 0.597 | 0.257 | non | 0 |
| B4_sans_attention_topo | 25.3 | 24.5 | 0.591 | 0.253 | non | 0 |
| B4_sans_robustesse | 25.5 | 24.9 | 0.579 | 0.253 | non | 0 |
| B4_sans_topologie | 24.5 | 23.9 | 0.609 | 0.257 | non | 0 |

Séparation partagé/privé de B4 : alignement inter-modal des tokens partagés 0.508, des tokens privés 0.000 ; recouvrement |cos| partagé-privé 0.026 (image), 0.022 (texte). Poids de fusion moyens : image 0.463, text 0.537.

## Vérification de la borne de stabilité (B4)

Violations de la borne d_B ≤ (L_phi + L_w) · max ||u_i − u'_i||, sur tous les scénarios et toutes les graines : **0**. Cette borne porte sur les points d'entrée de la filtration.

| Scénario | Déplacement moyen | d_B H0 | d_B H1 | Rapport maximal observé | Borne théorique |
|---|---|---|---|---|---|
| image_gaussian_noise_0.1 | 0.303 | 0.0113 | 0.0013 | 0.111 | 2.000 |
| image_gaussian_noise_0.2 | 0.487 | 0.0182 | 0.0012 | 0.105 | 2.000 |
| image_blur_radius_2.0 | 0.257 | 0.0097 | 0.0012 | 0.096 | 2.000 |
| image_blur_radius_4.0 | 0.561 | 0.0194 | 0.0015 | 0.094 | 2.000 |
| image_random_mask_0.25 | 0.314 | 0.0114 | 0.0013 | 0.088 | 2.000 |
| image_random_mask_0.5 | 0.535 | 0.0181 | 0.0013 | 0.084 | 2.000 |
| text_token_deletion_0.2 | 0.451 | 0.0169 | 0.0012 | 0.118 | 2.000 |
| text_token_deletion_0.4 | 0.709 | 0.0260 | 0.0015 | 0.099 | 2.000 |
| text_char_corruption_0.1 | 0.716 | 0.0247 | 0.0013 | 0.078 | 2.000 |
| text_char_corruption_0.2 | 1.011 | 0.0385 | 0.0018 | 0.083 | 2.000 |

## Coûts mesurés sur CPU

| Modèle | Paramètres (M) | Entraînement (min) | Époque médiane (s) | Inférence (ms/paire) | Persistance (ms/nuage) | Ultramétrique (ms/nuage) | Pic mémoire (Mo) |
|---|---|---|---|---|---|---|---|
| B0 | 0.69 | 2.2 | 11 | 0.83 | - | - | 2391 |
| B1 | 0.69 | 3.1 | 12 | 0.77 | - | - | 2377 |
| B2 | 0.69 | 6.6 | 27 | 0.64 | - | - | 2393 |
| B3 | 1.16 | 15.1 | 60 | 2.99 | 0.946 | 0.125 | 2440 |
| B4 | 3.02 | 46.8 | 187 | 4.87 | 0.454 | 0.044 | 2888 |
| B4_sans_L_top | 3.02 | 40.8 | 163 | 5.70 | 0.613 | 0.050 | 2942 |
| B4_sans_tokens_topo | 2.73 | 37.5 | 150 | 5.34 | 0.944 | 0.087 | 2860 |
| B4_sans_attention_topo | 3.02 | 48.9 | 196 | 4.33 | 0.518 | 0.041 | 2914 |
| B4_sans_robustesse | 3.02 | 35.8 | 143 | 3.45 | 0.296 | 0.028 | 2736 |
| B4_sans_topologie | 2.68 | 30.1 | 121 | 3.38 | - | - | 2700 |

L'inférence porte sur les têtes entraînables à partir des caractéristiques en cache, sans le coût des encodeurs gelés. La durée d'entraînement vaut le nombre d'époques multiplié par la durée médiane d'une époque, sans la validation. La médiane écarte 7 époques plus de trois fois plus longues que la médiane de leur expérience, allongées par une pause de la machine pendant le calcul. Les durées dépendent de la charge de la machine, car plusieurs expériences tournaient en parallèle. Voir `docs/LIMITATIONS.md`, section Logiciel, pour les variantes B4 touchées par une contention plus forte.
