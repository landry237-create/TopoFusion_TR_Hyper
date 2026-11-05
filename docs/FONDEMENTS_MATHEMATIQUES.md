# Fondements mathématiques

Ce document énonce ce qui est démontré, ce qui est seulement mesuré, et ce qui n'est pas garanti. Chaque garantie précise ses hypothèses.

## 1. Trois niveaux à ne pas confondre

| Niveau | Rôle | Implémentation | Différentiable ? |
|---|---|---|---|
| **A. Persistance exacte** | Diagnostic et évaluation | GUDHI (`SimplexTree`) et Ripser | Non : les tableaux sont détachés de PyTorch |
| **B. Persistance différentiable** | Optimisation | Appariement exact par GUDHI, valeurs relues dans le tenseur PyTorch | Oui, presque partout |
| **C. Descripteurs auxiliaires** | Encodage en tokens | Paysages (exacts sur la grille) ; courbes de Betti douces (approximation) | Oui |

**Point essentiel :** le niveau B n'est pas une approximation. Une valeur de naissance ou de mort est toujours la valeur de filtration d'un simplexe critique. GUDHI détermine lequel, sans gradient, et cette valeur est ensuite lue dans le tenseur de filtration, qui porte le gradient. Les valeurs coïncident donc avec celles du niveau A (test `test_valeurs_differentiables_identiques_aux_valeurs_exactes`).

L'application « filtration → diagramme » est continue, et dérivable tant que l'ordre des simplexes ne change pas. Ce gradient est donc un sous-gradient valide presque partout, mais il n'est pas défini aux points où l'ordre change (Carrière et al., 2021 ; Leygonie et al., 2021). Le test `test_gradient_autograd_coincide_avec_differences_finies` compare ce gradient aux différences finies de la persistance exacte.

## 2. Filtration de Rips pondérée sur une métrique apprise

Soit un nuage de points $u_1, \dots, u_N$ de la sphère unité, sur lequel on place :

- une métrique apprise $D_{ij} = \|\phi(u_i) - \phi(u_j)\|_2$, où $\phi$ est $L_\phi$-lipschitzienne ;
- un poids de sommet $w_i = \mathrm{softplus}(g(u_i) - c) \ge 0$, où $g$ est lipschitzienne.

La filtration (Rips pondéré, Anai et al., 2020) est définie par :

$$F(i) = w_i, \qquad F(ij) = \max\left(w_i,\ w_j,\ \frac{D_{ij} + w_i + w_j}{2}\right), \qquad F(\sigma) = \max_{e \subset \sigma} F(e) \text{ pour } \dim \sigma \ge 2.$$

**Validité.** Une arête apparaît après ses sommets, car $F(ij) \ge \max(w_i, w_j)$, et un triangle apparaît après ses arêtes par construction. $F$ est donc une filtration monotone du complexe de drapeaux complet.

**Choix du complexe.** On calcule le squelette de dimension 2 du complexe de drapeaux complet : c'est le minimum nécessaire pour obtenir $H_0$ et $H_1$ exacts. Ce complexe coïncide avec Vietoris-Rips lorsque les poids sont nuls, ce que vérifie le test `test_gudhi_coincide_avec_ripser` sur des nuages de distances.

**Ce qui a été retiré.** La version d'origine ajoutait `1 - A_ij` à la filtration, issu d'un réseau non contraint, et supprimait les arêtes sous un seuil de 0,05. Le seuil rendait le diagramme discontinu, et le réseau n'avait aucune constante de Lipschitz. Le graphe alimente désormais les points d'entrée par passage de messages : son effet entre dans $u_i$, et la borne suivante reste valide par rapport à ces points.

## 3. Contrainte de Lipschitz

$\phi$ et $g$ sont composées de couches affines $x \mapsto W'x + b$, avec $W' = W / \max(1, \|W\|_2)$, et d'activations ReLU. La norme $\|W\|_2$ est la plus grande valeur singulière, calculée exactement par SVD.

- $\|W'\|_2 \le 1$ et la translation $b$ ne change rien ; ReLU et softplus sont 1-lipschitziennes.
- Par composition, $L_\phi \le 1$ et $L_g \le 1$, à l'erreur d'arrondi flottant près.
- Le décalage $c$ est une translation et ne change pas la constante : $L_w \le L_g \le 1$.

**Pourquoi la SVD.** La normalisation spectrale usuelle (Miyato et al., 2018) estime $\|W\|_2$ par itération de la puissance. Une sous-estimation donne une norme réelle supérieure à 1 : **1,0033** a été mesuré dans la version d'origine. Cette estimation est acceptable pour régulariser, mais pas pour énoncer une borne.

## 4. Proposition de stabilité démontrée

**Proposition.** Soient $U = (u_i)$ et $U' = (u'_i)$ deux nuages de $N$ points indexés par les mêmes sommets, et $\delta = \max_i \|u_i - u'_i\|_2$. Pour $k \in \{0, 1\}$ :

$$d_B\big(\mathrm{Dgm}_k(F_U),\ \mathrm{Dgm}_k(F_{U'})\big) \le (L_\phi + L_w)\,\delta \le 2\delta.$$

**Démonstration.**

1. Pour tout sommet, $|w_i - w'_i| \le L_w \delta$.
2. Par l'inégalité triangulaire, $|D_{ij} - D'_{ij}| \le \|\phi(u_i) - \phi(u'_i)\| + \|\phi(u_j) - \phi(u'_j)\| \le 2 L_\phi \delta$.
3. Donc $\left|\frac{D_{ij} + w_i + w_j}{2} - \frac{D'_{ij} + w'_i + w'_j}{2}\right| \le L_\phi\delta + L_w\delta$.
4. Le maximum est 1-lipschitzien en norme infinie, donc $|F(ij) - F'(ij)| \le \max(L_w\delta,\ (L_\phi + L_w)\delta) = (L_\phi + L_w)\delta$.
5. Le même argument s'applique à chaque triangle, puisque sa valeur est le maximum de celles de ses arêtes. Donc $\|F - F'\|_\infty \le (L_\phi + L_w)\delta$ sur tous les simplexes.
6. Le théorème de stabilité appliqué à deux fonctions monotones sur un même complexe simplicial fini donne $d_B \le \|F - F'\|_\infty$ (Cohen-Steiner, Edelsbrunner et Harer, 2007 ; Chazal et al., 2009). $\square$

**Hypothèses à respecter :** mêmes sommets, ce qui exclut un masquage qui changerait les tokens ; même complexe ; mêmes poids $\theta$ ; $\delta$ mesuré sur les points d'entrée de la filtration.

**Vérification numérique.** Le test `test_borne_de_stabilite_verifiee_numeriquement` contrôle la borne sur des nuages aléatoires. `scripts/run_experiments.py` la contrôle aussi sur de vraies corruptions d'images et de textes du jeu de test.

### Ce que cette borne ne dit pas

Ces trois propriétés sont différentes, et seule la première est démontrée ici :

1. **Stabilité des diagrammes** par rapport aux points d'entrée de la filtration : **démontrée** ci-dessus.
2. **Stabilité des représentations latentes** par rapport aux pixels ou aux mots : **non garantie**. Encodeurs gelés, projecteurs, attention et `LayerNorm` ne sont pas contraints. Une petite perturbation de l'image peut produire un grand $\delta$. La chaîne complète n'est donc pas 1-lipschitzienne.
3. **Robustesse de la recherche** : **seulement mesurée**. Une borne sur un diagramme ne borne ni le rang ni le Recall.

Aucun résultat de ce projet n'est présenté comme « certifié ».

## 5. Perte géométrique : isométrie approchée, jamais garantie

Pour des matrices de distances $D^{(a)}, D^{(b)}$ d'un lot, on calcule la corrélation de Mantel $r$ : la corrélation de Pearson sur le triangle supérieur strict, sans la diagonale.

$$\mathcal{L}_{geom} = \frac{1}{3}\Big[(1 - r(D^{x_I}, D^{z_I})) + (1 - r(D^{x_T}, D^{z_T})) + (1 - r(D^{z_I}, D^{z_T}))\Big]$$

Propriétés et limites :

- **Invariance d'échelle :** $r$ ne dépend pas d'un facteur multiplicatif.
- **Pas de solution triviale :** une représentation effondrée a une variance de distance nulle, donc $r = 0$ et une perte maximale. La version d'origine donnait $0$ dans ce cas (test `test_perte_geometrique_penalise_l_effondrement`).
- **Préservation affine seulement :** $r = 1$ équivaut à $D^{(b)} = \alpha D^{(a)} + \beta$ avec $\alpha > 0$. Une perte nulle n'établit donc pas une isométrie, même à l'échelle près.
- **Tension assumée avec l'alignement :** si $z_I = z_T$, les deux géométries latentes coïncident, alors que les géométries d'entrée diffèrent. Les trois termes ne peuvent donc pas s'annuler en même temps, sauf cas particulier. Le poids $\lambda_{geom}$ règle ce compromis.

L'évaluation rapporte le stress-1 de Kruskal à échelle optimale, qui mesure l'écart à une véritable isométrie à l'échelle près, ainsi que la préservation des voisinages et la fiabilité.

## 6. Perte topologique

Pour des distances normalisées $\tilde D^{(a)}, \tilde D^{(b)}$ d'un sous-lot et l'ensemble $E^{(a)}$ des arêtes critiques exactes de $\tilde D^{(a)}$ (morts de $H_0$, naissances et morts de $H_1$) :

$$\mathcal{L}_{sig}(a, b) = \frac{1}{2}\left[\frac{1}{|E^{(a)}|}\sum_{e \in E^{(a)}} (\tilde D^{(a)}_e - \tilde D^{(b)}_e)^2 + \frac{1}{|E^{(b)}|}\sum_{e \in E^{(b)}} (\tilde D^{(a)}_e - \tilde D^{(b)}_e)^2\right]$$

C'est la perte de signature des auto-encodeurs topologiques (Moor et al., 2020). Elle est appliquée entre l'entrée et le latent de chaque modalité, et entre les latents image et texte.

**Honnêteté.** Elle compare des valeurs aux arêtes critiques ; elle n'optimise pas directement une distance de Wasserstein. L'évaluation mesure donc séparément les distances exactes de bottleneck et de Wasserstein-1. L'énoncé de la version d'origine, selon lequel une distance L2 de paysages « majore » Wasserstein, est faux : la stabilité de Bubenik (2015) va dans l'autre sens, la norme infinie des paysages étant majorée par la distance de bottleneck.

## 7. Paysages et courbes de Betti

**Paysage (Bubenik, 2015).** $\lambda_k(t)$ est la $k$-ième plus grande valeur de $\max(0, \min(t - b, d - t))$. Il est calculé exactement sur une grille propre à chaque échantillon, car une grille globale fixe peut laisser une barre courte entre deux points, ce qui rend le paysage aveugle à cette barre. Les bornes de la grille figurent aussi dans les statistiques des diagrammes.

Comparer deux paysages exige une grille commune : la distance L2 est donc calculée sur une grille construite pour chaque paire.

**Amplification du gradient : un écueil constaté.** Les diagrammes sont lipschitziens par rapport à la filtration, mais un encodage peut détruire cette propriété. Une première version divisait le paysage par la plage de la grille (environ 0,02 pour $H_1$), puis standardisait les descripteurs, dont la variance est très faible. Sur les données d'entraînement, le gradient total a atteint **525**, contre 6 sans topologie, et l'écrêtage a alors écrasé tout l'apprentissage. La division a été retirée, et la standardisation utilise un plancher de variance de $2{,}5 \times 10^{-3}$, qui borne son facteur d'amplification à 20. Ce correctif ramène le gradient à 6,4 au même stade.

**Courbe de Betti douce (approximation).** $\tilde\beta(\varepsilon) = \sum_p \sigma((\varepsilon - b_p)/\tau)\,\sigma((d_p - \varepsilon)/\tau)$. Elle tend vers la courbe exacte quand $\tau \to 0$ (test `test_betti_douce_converge_vers_exacte`). La figure `13_betti_douce_vs_exacte.png` compare les deux courbes sur un vrai échantillon de test. Les échelles $\varepsilon_m$ sont apprises et utilisées ici : dans la version d'origine, ce paramètre existait mais ne recevait aucun gradient.

## 8. Biais de l'attention topologique

$$A_{ij} = \mathrm{softmax}_j\left(\frac{q_i k_j^\top}{\sqrt{d_h}} + \lambda T_{ij}\right), \qquad T_{ij} = \tfrac{1}{2}\left(1 - \frac{u_{ij}}{\max u} + W_{ij}\right)$$

- $u_{ij}$ est l'ultramétrique de liaison simple de $F$ : la valeur de filtration à laquelle $i$ et $j$ deviennent connectés. Elle code exactement la hiérarchie de fusion de $H_0$ et reste stable en norme infinie (Carlsson et Mémoli, 2010).
- $W_{ij}$ est le poids d'arête du graphe adaptatif, et $\lambda$ est appris pour chaque tête.
- Le biais n'encode que $H_0$. $H_1$ atteint l'attention par ses tokens.

**Incertitude des arêtes.** La variance apprise n'est pas bayésienne. Elle prend son sens grâce à une vraisemblance hétéroscédastique (Kendall et Gal, 2017) dans la perte de robustesse : $\frac{1}{2}e^{-s_{ij}}(W_{ij} - \tilde W_{ij})^2 + \frac{1}{2}s_{ij}$, où $\tilde W$ est le graphe perturbé.

## 9. Objectif composite

$$\mathcal{L} = \mathcal{L}_{sem} + \lambda_{fusion}\mathcal{L}_{fusion} + \lambda_{geom}\mathcal{L}_{geom} + \lambda_{top}\mathcal{L}_{top} + \lambda_{rob}\mathcal{L}_{rob} + \lambda_{orth}\mathcal{L}_{orth} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{var}\mathcal{L}_{var}$$

| Terme | Justification | Phase |
|---|---|---|
| $\mathcal{L}_{sem}$ | InfoNCE symétrique, faux négatifs exclus, température apprise | 1, 2, 3 |
| $\mathcal{L}_{fusion}$ | Relie l'embedding fusionné à chaque modalité réelle | 1, 2, 3 |
| $\mathcal{L}_{geom}$ | Préservation affine des distances (section 5) | 1, 2, 3 |
| $\mathcal{L}_{orth}$ | Séparation partagé/privé : $\|\hat S_c \hat S_p^\top\|_F^2$ (Bousmalis et al., 2016) | 1, 2, 3 |
| $\mathcal{L}_{rec}$ | Donne un rôle à l'espace privé : sans lui, l'orthogonalité seule serait satisfaite par des tokens privés dépourvus d'information | 1, 2, 3 |
| $\mathcal{L}_{var}$ | Anti-effondrement par seuil de variance (Bardes et al., 2022) | 1, 2, 3 |
| $\mathcal{L}_{top}$ | Préservation de $H_0$ et $H_1$ (section 6) | 2, 3 |
| $\mathcal{L}_{rob}$ | Cohérence et recherche sous perturbation, cohérence topologique et du graphe | 3 |

**Retirée :** la perte temporelle d'origine traitait les tokens partagés d'une vidéo aléatoire comme une suite temporelle. Sans vraie vidéo, ce terme n'a aucun sens.

## 10. Complexité

Pour $N$ points, le squelette de dimension 2 contient $O(N^3)$ simplexes. C'est pourquoi les nuages restent petits : tokens partagés par échantillon ($K_c = 12$) et sous-lot pour la perte ($n = 64$). L'évaluation sur 256 images utilise Ripser. Les coûts mesurés figurent dans `docs/RESULTATS.md`.

Le squelette complet est préférable ici à une troncature de $\varepsilon$ : un seuil modifierait l'ensemble des simplexes selon les données, ce qui romprait l'hypothèse « même complexe » de la proposition.

## Références

- Anai, H. et al. (2020). *DTM-based filtrations*. Topological Data Analysis, Abel Symposia.
- Bardes, A., Ponce, J., LeCun, Y. (2022). *VICReg*. ICLR.
- Bousmalis, K. et al. (2016). *Domain Separation Networks*. NeurIPS.
- Bubenik, P. (2015). *Statistical Topological Data Analysis using Persistence Landscapes*. JMLR 16.
- Carlsson, G., Mémoli, F. (2010). *Characterization, Stability and Convergence of Hierarchical Clustering Methods*. JMLR 11.
- Carrière, M. et al. (2021). *Optimizing persistent homology based functions*. ICML.
- Chazal, F. et al. (2009). *Proximity of persistence modules and their diagrams*. SoCG.
- Cohen-Steiner, D., Edelsbrunner, H., Harer, J. (2007). *Stability of Persistence Diagrams*. Discrete & Computational Geometry 37.
- Kendall, A., Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS.
- Leygonie, J., Oudot, S., Tillmann, U. (2021). *A Framework for Differential Calculus on Persistence Barcodes*. Foundations of Computational Mathematics.
- Miyato, T. et al. (2018). *Spectral Normalization for Generative Adversarial Networks*. ICLR.
- Moor, M. et al. (2020). *Topological Autoencoders*. ICML.
