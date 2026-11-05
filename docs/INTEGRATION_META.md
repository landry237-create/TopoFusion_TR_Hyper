# Intégration de l'écosystème Meta

**Vérification effectuée le 5 novembre 2026 sur la documentation officielle.** Ces informations peuvent évoluer : à revérifier avant toute présentation.

## En bref

| Élément | Statut dans ce dépôt |
|---|---|
| **DINOv2** (`facebook/dinov2-small`, poids ouverts Meta) | **Réellement utilisé** dans tous les résultats, exécuté localement comme encodeur d'image gelé |
| **Muse Spark** (Meta Model API) | **Non utilisé dans les résultats.** Un adaptateur existe, mais il n'a jamais été exécuté contre la vraie API, faute de clé |
| **Muse Spark Open** | Non intégré : poids ouverts prévus pour des GPU, hors de portée de la machine de test |

**Formulation exacte à employer pour le hackathon :** « TopoFusion-TR Hyper utilise DINOv2 de Meta comme encodeur d'image et propose un adaptateur vers la Meta Model API, non exécuté à ce jour. » Il ne faut pas écrire que Muse Spark est intégré au modèle ni qu'il a contribué aux résultats.

## Ce que documente Meta

| Point | Information officielle | Source |
|---|---|---|
| URL de base | `https://api.meta.ai/v1` | [Quickstart](https://dev.meta.ai/docs/quickstart) |
| Authentification | En-tête `Authorization: Bearer $MODEL_API_KEY` | [Quickstart](https://dev.meta.ai/docs/quickstart) |
| Modèle le plus récent de la famille | `muse-spark-1.3` ; aussi `muse-spark-1.2`, `muse-spark-1.1` et des variantes `-contributor` | [Models](https://dev.meta.ai/docs/models) |
| Entrées et sortie | Texte, image, vidéo, audio, PDF en entrée ; **texte** en sortie. Pour 1.3, l'entrée audio n'est pas entièrement prise en charge | [Models](https://dev.meta.ai/docs/models) |
| Point de terminaison | `POST /chat/completions` et `POST /responses` | [Chat completions](https://dev.meta.ai/docs/protocols/chat-completions) |
| Images | Partie `image_url` : URL publique ou URL de données base64. Formats JPEG, PNG, GIF, WebP, ICO. 50 Mo par image en ligne, 50 images par requête | [Image understanding](https://dev.meta.ai/docs/image-understanding) |
| Logprobs | `logprobs: true` renvoie une erreur 400 | [Chat completions](https://dev.meta.ai/docs/protocols/chat-completions) |
| Erreurs temporaires | 429 : recul exponentiel avec gigue. 504 possible sur les longues requêtes sans streaming | [Chat completions](https://dev.meta.ai/docs/protocols/chat-completions) |
| Disponibilité | Peut varier selon le pays, voir la politique géographique | [Models](https://dev.meta.ai/docs/models) |

## Limitations déterminantes

**Aucun embedding n'est documenté.** Les pages consultées ne décrivent ni point de terminaison `/embeddings`, ni modèle d'embeddings, ni accès aux états cachés, et les logprobs sont refusés. Par conséquent :

- ce dépôt n'invente aucune dimension, aucune architecture interne et aucun point de terminaison ;
- Muse Spark **ne peut pas** remplacer un encodeur de TopoFusion, car l'architecture exige des tokens numériques ;
- `MetaModelApiClient.internal_embeddings()` lève explicitement `EmbeddingsNotExposed`.

**Points non vérifiés :** tarifs et limites de débit actuels, éligibilité géographique de l'utilisateur, conditions exactes du hackathon. Aucune page officielle du « Meta Global AI Developer Hackathon 2026 — PUSH THE FRONTIER » n'a été trouvée lors de la recherche : les règles doivent être vérifiées directement auprès des organisateurs.

## L'adaptateur légitime

La seule sortie documentée exploitable est un **texte**. `integrations/meta_model_api.py` demande donc une description de l'image. Ce texte peut ensuite passer par l'encodeur de texte local.

```text
Image réelle ──> Muse Spark /chat/completions ──> Texte ──> BERT local ──> Tokens
                       (sortie documentée)                     │
                                                   Provenance : dérivée d'un texte,
                                                   pas une représentation interne
```

Chaque description porte la provenance `meta_model_api_text_output`. Un cache local évite de payer deux fois la même image, et la clé n'est jamais écrite sur le disque.

### Exécuter l'adaptateur

Cela envoie de vraies images à un service externe et peut être facturé. Le script exige donc une confirmation explicite. Vérifiez d'abord que les conditions de COCO et de l'API autorisent cet envoi.

```bash
export MODEL_API_KEY="votre-cle"
python scripts/meta_describe_images.py --nombre 8 --confirmer-envoi
```

Sous Windows PowerShell : `$env:MODEL_API_KEY = "votre-cle"`.

Sans clé, le script s'arrête avec un message explicite. Il n'existe aucun repli ni résultat simulé.

## Séparation des caractéristiques

| Origine | Caractéristiques | Où |
|---|---|---|
| **Modèle Meta, poids ouverts, local** | Tokens d'image DINOv2 | `data/feature_cache.py`, `provenance.json` |
| **Encodeur local tiers** | Tokens de texte BERT (Google) | `data/feature_cache.py` |
| **Modules TopoFusion, entraînés ici** | Projecteurs, banque, graphe, filtration, tokens H0/H1, attention | `models/` |
| **Modèle Meta par API** | Descriptions textuelles, seulement si l'utilisateur exécute l'adaptateur | `integrations/meta_model_api.py` |
| **Référence de comparaison** | CLIP ViT-B/32 (LAION), sans lien avec Meta | `models/encoders.py` |

`models.encoders.describe_feature_sources` imprime cette séparation au début de chaque expérience.

## Tests : ce qu'ils prouvent ou non

`tests/test_meta_adapter_and_project.py` utilise un **serveur local de substitution**. Ces tests vérifient le format documenté de la requête, l'authentification, les nouvelles tentatives, le cache et le refus sans clé. Ils **ne prouvent pas** que l'adaptateur fonctionne contre la vraie API.

## Pour aller plus loin

Pour une vraie intégration avec des caractéristiques internes, il faudrait soit que Meta expose des embeddings, soit passer aux poids ouverts (Muse Spark Open, Muse Glimmer), à condition que leurs modèles exposent des états cachés, ce que la documentation consultée n'indique pas. Cela demanderait une vérification et du matériel GPU.
