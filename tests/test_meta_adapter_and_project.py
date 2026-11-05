# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Vérifie l'adaptateur Meta contre un serveur local de substitution, et les règles du projet.

Ce serveur est un simple substitut de test : il vérifie le format documenté de la requête,
mais ne prouve en aucun cas l'accès à la vraie API.
"""

import base64  # Décode l'image transmise dans la requête.
import io  # Lit l'image décodée en mémoire.
import json  # Lit et écrit le corps des requêtes.
import threading  # Exécute le serveur de substitution en arrière-plan.
from http.server import BaseHTTPRequestHandler, HTTPServer  # Fournit le serveur de substitution.
from pathlib import Path  # Manipule les chemins indépendamment du système.

import pytest  # Fournit le cadre de test.
from PIL import Image  # Crée et décode l'image de test.

from integrations.meta_model_api import PROVENANCE, EmbeddingsNotExposed, MetaApiError, MetaApiUnavailable, MetaModelApiClient  # Importe l'adaptateur.
from scripts.verifier_commentaires import analyser_projet, lignes_sans_commentaire  # Importe le vérificateur de commentaires.
from utils.config import load_config  # Charge la configuration.

RACINE = Path(__file__).resolve().parents[1]  # Calcule la racine du projet.


class _Substitut(BaseHTTPRequestHandler):  # Simule la forme des réponses documentées.
    """Serveur local de test : enregistre les requêtes reçues et renvoie une réponse de forme documentée."""

    requetes = []  # Conserve les requêtes reçues.
    codes = []  # Liste les codes d'erreur à renvoyer d'abord.

    def do_POST(self):  # Traite une requête POST.
        """Enregistre la requête, puis renvoie soit une erreur programmée, soit une réponse valide."""
        taille = int(self.headers["Content-Length"])  # Lit la taille du corps.
        corps = json.loads(self.rfile.read(taille).decode("utf-8"))  # Lit le corps de la requête.
        _Substitut.requetes.append({"chemin": self.path, "autorisation": self.headers.get("Authorization"), "corps": corps})  # Enregistre la requête.
        code = _Substitut.codes.pop(0) if _Substitut.codes else 200  # Choisit le code de réponse.
        reponse = {"model": corps["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "texte de substitution"}, "finish_reason": "stop"}]} if code == 200 else {"error": {"message": "erreur simulée"}}  # Construit la réponse.
        contenu = json.dumps(reponse).encode("utf-8")  # Sérialise la réponse.
        self.send_response(code)  # Envoie le code de réponse.
        self.send_header("Content-Type", "application/json")  # Envoie le type du contenu.
        self.send_header("Content-Length", str(len(contenu)))  # Envoie la taille du contenu.
        self.end_headers()  # Termine les en-têtes.
        self.wfile.write(contenu)  # Envoie le contenu.

    def log_message(self, *_):  # Réduit le bruit du serveur.
        """Désactive les messages du serveur de test."""


@pytest.fixture()  # Déclare un montage.
def serveur():  # Démarre le serveur de substitution.
    """Démarre le serveur de substitution sur un port libre local."""
    _Substitut.requetes = []  # Réinitialise les requêtes.
    _Substitut.codes = []  # Réinitialise les erreurs programmées.
    instance = HTTPServer(("127.0.0.1", 0), _Substitut)  # Crée le serveur sur un port libre.
    fil = threading.Thread(target=instance.serve_forever, daemon=True)  # Prépare le fil d'exécution.
    fil.start()  # Démarre le serveur.
    yield f"http://127.0.0.1:{instance.server_port}/v1"  # Fournit l'URL au test.
    instance.shutdown()  # Arrête le serveur.
    instance.server_close()  # Libère le port.


def _cfg_meta(url):  # Fabrique la section de configuration.
    """Retourne la section de configuration de l'adaptateur pour le test."""
    return {"base_url": url, "model": "muse-spark-1.3", "api_key_env": "CLE_TEST_TOPOFUSION", "max_completion_tokens": 32, "max_retries": 3, "timeout_seconds": 5}  # Construit la section.


def test_sans_cle_l_adaptateur_refuse_explicitement(monkeypatch):  # Vérifie l'absence de repli.
    """Sans clé, aucun repli silencieux ni résultat simulé ne doit être utilisé."""
    monkeypatch.delenv("CLE_TEST_TOPOFUSION", raising=False)  # Retire la clé de test.
    with pytest.raises(MetaApiUnavailable, match="Aucun repli"):  # Attend une erreur explicite.
        MetaModelApiClient(_cfg_meta("http://127.0.0.1:9/v1"))  # Essaie de créer le client.


def test_requete_conforme_au_format_documente(monkeypatch, serveur):  # Vérifie le format de la requête.
    """La requête doit suivre le format documenté, et la provenance être indiquée."""
    monkeypatch.setenv("CLE_TEST_TOPOFUSION", "cle-de-test")  # Définit une clé de test.
    client = MetaModelApiClient(_cfg_meta(serveur))  # Crée le client.
    resultat = client.describe_image(Image.new("RGB", (32, 32), (200, 10, 10)))  # Demande une description.
    assert resultat.provenance == PROVENANCE and resultat.modele == "muse-spark-1.3"  # Vérifie la provenance.
    requete = _Substitut.requetes[0]  # Lit la requête reçue.
    assert requete["chemin"] == "/v1/chat/completions"  # Vérifie le chemin documenté.
    assert requete["autorisation"] == "Bearer cle-de-test"  # Vérifie l'authentification documentée.
    assert requete["corps"]["model"] == "muse-spark-1.3"  # Vérifie l'identifiant du modèle.
    assert "max_completion_tokens" in requete["corps"] and "max_tokens" not in requete["corps"]  # Vérifie le paramètre recommandé.
    contenu = requete["corps"]["messages"][0]["content"]  # Lit le contenu du message.
    assert contenu[0]["type"] == "text" and contenu[1]["type"] == "image_url"  # Vérifie la structure documentée.
    url = contenu[1]["image_url"]["url"]  # Lit l'URL de données.
    assert url.startswith("data:image/jpeg;base64,")  # Vérifie le format documenté.
    assert Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size == (32, 32)  # Vérifie que l'image est décodable.


def test_nouvelle_tentative_sur_429_et_echec_sur_400(monkeypatch, serveur):  # Vérifie la gestion des erreurs.
    """L'erreur 429 doit provoquer une nouvelle tentative, et l'erreur 400 remonter immédiatement."""
    monkeypatch.setenv("CLE_TEST_TOPOFUSION", "cle-de-test")  # Définit une clé de test.
    monkeypatch.setattr("integrations.meta_model_api.time.sleep", lambda _: None)  # Supprime l'attente pendant le test.
    client = MetaModelApiClient(_cfg_meta(serveur))  # Crée le client.
    _Substitut.codes = [429]  # Programme une erreur temporaire.
    assert client.describe_image(Image.new("RGB", (8, 8))).texte == "texte de substitution"  # Vérifie la réussite après la nouvelle tentative.
    assert len(_Substitut.requetes) == 2  # Vérifie le nombre d'appels.
    _Substitut.codes = [400]  # Programme une erreur définitive.
    with pytest.raises(MetaApiError, match="400"):  # Attend une erreur explicite.
        client.describe_image(Image.new("RGB", (8, 8), (1, 2, 3)))  # Envoie une nouvelle image.


def test_cache_evite_un_second_appel(monkeypatch, serveur, tmp_path):  # Vérifie le cache local.
    """Une même image ne doit pas être facturée deux fois, et la clé ne doit pas être écrite sur le disque."""
    monkeypatch.setenv("CLE_TEST_TOPOFUSION", "cle-secrete-de-test")  # Définit une clé de test.
    client = MetaModelApiClient(_cfg_meta(serveur), tmp_path)  # Crée le client avec un cache.
    image = Image.new("RGB", (8, 8), (9, 9, 9))  # Crée une image.
    premier = client.describe_image(image)  # Effectue le premier appel.
    second = client.describe_image(image)  # Effectue le second appel.
    assert len(_Substitut.requetes) == 1 and second.depuis_cache and not premier.depuis_cache  # Vérifie l'usage du cache.
    assert all("cle-secrete" not in f.read_text(encoding="utf-8") for f in tmp_path.iterdir())  # Vérifie l'absence de la clé dans le cache.


def test_embeddings_internes_refuses_explicitement(monkeypatch, serveur):  # Vérifie l'absence de fonctionnalité inventée.
    """L'adaptateur ne doit pas prétendre fournir des embeddings que la documentation ne décrit pas."""
    monkeypatch.setenv("CLE_TEST_TOPOFUSION", "cle-de-test")  # Définit une clé de test.
    with pytest.raises(EmbeddingsNotExposed):  # Attend une erreur explicite.
        MetaModelApiClient(_cfg_meta(serveur)).internal_embeddings()  # Essaie de demander des embeddings.


def test_integration_meta_inactive_par_defaut():  # Vérifie la configuration par défaut.
    """L'adaptateur doit être inactif par défaut, sans clé dans la configuration."""
    meta = load_config(str(RACINE / "config.yaml"))["meta_model_api"]  # Charge la section.
    assert meta["enabled"] is False  # Vérifie l'inactivité.
    assert "key" not in " ".join(k for k in meta if k != "api_key_env").lower()  # Vérifie l'absence de clé stockée.


def test_aucune_generation_aleatoire_de_modalite_dans_le_pipeline():  # Vérifie l'absence de fausses modalités.
    """Le code du pipeline ne doit pas générer de caractéristiques aléatoires d'audio ou de vidéo."""
    for dossier in ("data", "models", "training", "evaluation", "scripts", "integrations"):  # Parcourt les dossiers du code.
        for fichier in (RACINE / dossier).rglob("*.py"):  # Parcourt les fichiers.
            texte = fichier.read_text(encoding="utf-8")  # Lit le fichier.
            for motif in ("audio_features", "video_features", "Synthetic query", "synthétique multimodal"):  # Parcourt les motifs interdits.
                assert motif not in texte, f"{motif} trouvé dans {fichier}"  # Vérifie l'absence du motif.


def test_le_verificateur_detecte_une_ligne_sans_commentaire():  # Vérifie le vérificateur.
    """Le vérificateur doit signaler une ligne de code sans commentaire, et accepter le reste."""
    source = 'x = 1  # commentaire\ny = 2\n"""docstring\nsur deux lignes"""\n\nz = (3,  # a\n     4)\n'  # Définit un code de test.
    assert lignes_sans_commentaire(source) == [2, 7]  # Vérifie les lignes signalées.


def test_chaque_ligne_de_code_du_projet_porte_un_commentaire():  # Vérifie l'exigence du projet.
    """Chaque ligne de code Python du projet doit porter un commentaire explicatif."""
    defauts = analyser_projet(RACINE)  # Analyse le projet.
    assert defauts == {}, f"Lignes sans commentaire : { {f: l[:8] for f, l in defauts.items()} }"  # Vérifie l'absence de défaut.
