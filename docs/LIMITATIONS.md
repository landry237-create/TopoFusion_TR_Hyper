# Limitations

Ce document liste ce que le prototype ne démontre pas. Les chiffres mesurés et leur lecture figurent dans `docs/RESULTATS.md`.

## Portée des données

- Un seul jeu de données est évalué : les légendes de `AhmedSSabir/Textual-Image-Caption-Dataset`, associées aux images COCO train2014. Aucune conclusion ne se transfère à un autre corpus sans nouvelle expérience.
- Le protocole utilise un sous-ensemble de 10 000 images sur les 82 783 disponibles : 8 000 pour l'entraînement, 1 000 pour la validation et 1 000 pour le test. Un rappel mesuré sur 1 000 images n'est pas comparable à un résultat publié sur une galerie de taille différente.
- L'entraînement retient 3 légendes par image et l'évaluation au plus 5. Les 15 images d'origine indisponibles ont été remplacées à partir de la réserve.
- Les images sont téléchargées depuis le serveur public de COCO. Si des fichiers disparaissent, le manifeste reconstruit peut différer du manifeste utilisé ici.

## Modèle et encodeurs

- Les encodeurs DINOv2-small et BERT sont gelés. Seules les couches au-dessus de leurs caractéristiques sont entraînées, donc le résultat dépend fortement de la qualité de ces caractéristiques.
- DINOv2-small a remplacé DINOv2-base pour une raison de coût : l'extraction prenait environ 0,3 s par image sur le processeur disponible, sans GPU.
- Les tokens d'image sont réduits au CLS et à une grille 4×4. Les tokens de texte sont limités à 24.
- L'entraînement dure 15 époques au maximum. Les variantes n'ont pas été poussées jusqu'à une convergence établie sur une longue durée.

## Protocole statistique

- Le plan utilise 3 graines pour B0 à B4 et pour `B4_sans_topologie`, et 2 graines pour les quatre autres ablations. Avec si peu de graines, l'écart-type est lui-même imprécis, et une différence faible entre variantes ne doit pas être interprétée.
- Les intervalles bootstrap portent sur les requêtes de test, pour un modèle donné. Ils ne capturent pas la variabilité due à l'initialisation ni au découpage.
- Le découpage train/val/test est unique. Aucune validation croisée n'a été réalisée.
- Les comparaisons multiples entre variantes ne sont pas corrigées.
- Les réglages (`lr`, `geom`, `top`) ont été choisis sur la validation, avec la seule graine 42 et une petite grille. Le réglage `top=8` n'a pas abouti et a été abandonné. D'autres poids pourraient donner des résultats différents.
- Plusieurs corrections de conception de la voie structurelle (décalage constant, explosion du gradient) ont été motivées par des observations sur la validation. Cela n'introduit aucune fuite du test, mais la validation n'est plus tout à fait indépendante du choix de l'architecture.
- Après le gel des réglages, deux essais de bout en bout d'une époque (B1 et B4) ont vérifié le pipeline, y compris l'évaluation sur le test. Leurs métriques de test n'ont pas été lues ni utilisées, et ils ont été écrits dans un dossier temporaire.

## Portée des garanties mathématiques

- La borne de stabilité `d_B ≤ (L_phi + L_w) · max_i ||u_i − u'_i||` s'applique uniquement aux points d'entrée de la filtration. Elle ne s'étend pas aux embeddings finaux, ni à la qualité de recherche, ni aux images d'origine.
- Elle suppose que `phi` et `g` sont 1-lipschitziens, ce qui est assuré par la SVD exacte à chaque passe avant, à la précision numérique près. Le modèle ne fournit aucune garantie d'isométrie : `L_geom` et les corrélations de Mantel sont des objectifs et des mesures.
- La borne n'est jamais violée sur les corruptions testées, mais elle est très lâche : la figure `12_stabilite_borne.png` montre des distances de bottleneck bien inférieures à la borne. Elle garantit une limite, sans décrire le comportement réel des diagrammes.
- Aucune robustesse n'est certifiée. Les scénarios de robustesse sont des mesures empiriques sur un ensemble fini de perturbations.
- Le gradient de la persistance n'existe que presque partout : l'appariement est constant par morceaux et change aux égalités de filtration. La propriété est vérifiée numériquement par différences finies sur des cas génériques.
- Le paysage, les courbes de Betti douces et les tokens topologiques sont des représentations dérivées des diagrammes. Les courbes de Betti douces sont des approximations, comparées au calcul exact dans `test_betti_douce_converge_vers_exacte`. À la température utilisée (`soft_betti_temperature: 0.05`), elles restent très lissées par rapport aux courbes exactes, car les valeurs de filtration des tokens sont de l'ordre de 0,1 à 0,25 : la figure `13_betti_douce_vs_exacte.png` le montre sur un échantillon de test.
- Le nuage de points de la perte topologique compte au plus 64 points par lot (`loss_points`). L'évaluation topologique en utilise 256 (`topology_points`). Ces échantillons ne décrivent pas la topologie de l'espace latent entier.

## Robustesse

- Pendant l'entraînement, les perturbations s'appliquent aux caractéristiques mises en cache, et non aux images ou aux textes bruts. Le passage de l'une à l'autre n'est pas établi.
- Au test, les corruptions s'appliquent aux vraies images et aux vrais textes, puis aux encodeurs gelés. Elles se limitent à 5 familles et à 2 sévérités chacune, plus l'absence d'une modalité dans la fusion.
- Une seule graine de corruption est utilisée, ce qui ne mesure pas la variabilité due au tirage des perturbations.

## Comparaison avec CLIP

- CLIP est évalué tel quel, avec les poids LAION pré-entraînés sur un corpus massif de paires image-texte. Il n'est pas réentraîné sur le sous-ensemble de 8 000 images, alors que TopoFusion y est entraîné. Les deux ne sont donc pas comparés à entraînement égal.
- Les poids `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` ont été choisis parce qu'ils sont disponibles en safetensors, alors que torch 2.5.1 refuse les poids OpenAI au format `.bin` dans notre configuration.
- La comparaison sert de référence externe et ne mesure pas un remplacement possible. TopoFusion n'est pas présenté comme équivalent à CLIP.

## Modalités non exécutées

- L'audio et la vidéo ne sont jamais exécutés : aucune vraie donnée de ces modalités n'est disponible dans le protocole. Les encodeurs `ExperimentalAudioEncoder` et `ExperimentalVideoEncoder` sont des extensions expérimentales, et les activer dans la configuration est refusé.
- Aucun résultat de ce dépôt ne porte sur l'audio ou la vidéo.

## Intégration Meta

- Muse Spark n'est jamais appelé par ce prototype et ne contribue à aucun résultat. L'adaptateur `integrations/meta_model_api.py` n'a pas été exécuté contre la vraie API, faute de clé. Il est testé uniquement contre un serveur simulé : format des requêtes, nouvelles tentatives, cache et refus explicites.
- La documentation consultée ne décrit aucun accès aux embeddings de Muse Spark. Le modèle n'est donc pas utilisé comme encodeur.
- DINOv2 est un modèle ouvert de Meta exécuté localement. Il ne s'agit pas d'une intégration de Muse Spark.
- Aucune page officielle du hackathon n'a été trouvée lors de la vérification. Les règles et critères du concours n'ont donc pas pu être confrontés au projet.

## Logiciel

- Le code a été exécuté uniquement sous Linux, en CPU, avec Python 3.12.7 et les versions de `requirements.txt`. Windows n'a pas été testé : la gestion des chemins et la mesure de la mémoire sont prévues pour ce système, sans validation réelle. L'exécution sur GPU n'a pas été testée non plus.
- Les durées d'entraînement et d'inférence des expériences finales ont été mesurées avec 4 expériences simultanées à 1 fil chacune, sur un processeur à 4 cœurs. Ce sont des ordres de grandeur, et non un banc de performance isolé.
- Le plan final a changé d'environnement en cours de route. B0 à B3 ont été calculés avec l'environnement Anaconda. Le 9 novembre vers 03:50, un second lancement du plan depuis `.venv` a repris B4 (graines 42, 43, 44) et `B4_sans_topologie` (graine 42) à leurs points de contrôle ; le premier plan a été arrêté à 03:59, et tout le reste du plan a été calculé dans `.venv`. Les deux environnements ont les mêmes versions de PyTorch, NumPy, SciPy et GUDHI, et un entraînement B4 réduit y produit des poids identiques au bit près. Pendant 8 minutes, 8 processus ont partagé les 4 cœurs : les durées d'époque de ces quatre expériences sont donc surestimées sur cette période.
- La machine a été éteinte à 08:22 puis redémarrée, ce qui a interrompu `B4_sans_attention_topo` et `B4_sans_robustesse` (graines 42 et 43). Ces quatre expériences ont repris à leur dernier point de contrôle. La reprise exacte est vérifiée par `test_reprise_identique_a_une_execution_continue`, mais seulement sur un petit jeu.
- Sept époques, réparties sur quatre ablations, ont duré de 66 à 81 minutes au lieu de 2 à 3, car la machine a été suspendue ou figée pendant ces époques. Le tableau des coûts écarte ces pauses en multipliant la durée médiane d'une époque par le nombre d'époques.
- La reprise après interruption restaure les poids, l'optimiseur, le planning et les générateurs aléatoires. Un test CPU sur un petit jeu confirme l'égalité exacte avec une exécution continue. Cette égalité n'a pas été vérifiée à l'échelle complète, ni sur un autre processeur ou une autre version de PyTorch.
- Les points de contrôle sont lus avec `torch.load(..., weights_only=False)`, car ils contiennent des métadonnées Python. Ne chargez que des points de contrôle dont vous connaissez l'origine.
- Réutiliser un résultat existant suppose la même empreinte de configuration. Cette empreinte ne couvre pas le code source : après une modification du code, les résultats doivent être supprimés pour être recalculés.
