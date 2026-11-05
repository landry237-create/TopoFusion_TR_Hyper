# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Vérifie que chaque ligne de code Python du projet porte un commentaire explicatif.

Les lignes vides et le texte interne des docstrings sont exemptés : une docstring est déjà
un commentaire explicatif, et aucun commentaire ne peut être placé à l'intérieur d'une chaîne.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import io  # Permet d'analyser un texte en mémoire.
import sys  # Fournit le code de sortie du script.
import tokenize  # Découpe le code source en unités lexicales.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Dict, List, Set  # Importe les types des signatures.

IGNORES = {"NEWLINE", "NL", "INDENT", "DEDENT", "ENDMARKER", "ENCODING", "COMMENT"}  # Liste les unités qui ne constituent pas du code.
EXCLUS = {"data_cache", "__pycache__", "outputs", ".git"}  # Liste les dossiers à ne pas analyser.


def lignes_sans_commentaire(source: str) -> List[int]:  # Analyse un fichier source.
    """Retourne les numéros des lignes de code dépourvues de commentaire."""
    lignes_code: Set[int] = set()  # Prépare les lignes contenant du code.
    lignes_commentees: Set[int] = set()  # Prépare les lignes contenant un commentaire.
    lignes_chaine_multiple: Set[int] = set()  # Prépare les lignes intérieures des chaînes multilignes.
    for jeton in tokenize.generate_tokens(io.StringIO(source).readline):  # Parcourt les unités lexicales.
        nom = tokenize.tok_name[jeton.type]  # Lit le type de l'unité.
        if nom == "COMMENT":  # Détecte un commentaire.
            lignes_commentees.add(jeton.start[0])  # Enregistre la ligne commentée.
        elif nom == "STRING" and jeton.start[0] != jeton.end[0]:  # Détecte une chaîne multiligne.
            lignes_chaine_multiple.update(range(jeton.start[0], jeton.end[0] + 1))  # Exempte toutes ses lignes.
        elif nom == "STRING" and jeton.line.strip().startswith(('"""', "'''")):  # Détecte une docstring sur une seule ligne.
            lignes_chaine_multiple.add(jeton.start[0])  # Exempte cette ligne.
        elif nom not in IGNORES:  # Détecte du code.
            lignes_code.update(range(jeton.start[0], jeton.end[0] + 1))  # Enregistre la ligne de code.
    return sorted(lignes_code - lignes_commentees - lignes_chaine_multiple)  # Retourne les lignes en défaut.


def analyser_projet(racine: Path) -> Dict[str, List[int]]:  # Analyse tous les fichiers du projet.
    """Retourne, pour chaque fichier Python en défaut, ses lignes sans commentaire."""
    defauts: Dict[str, List[int]] = {}  # Prépare le bilan.
    for chemin in sorted(racine.rglob("*.py")):  # Parcourt les fichiers Python.
        if EXCLUS & set(chemin.relative_to(racine).parts):  # Ignore les dossiers exclus.
            continue  # Passe au fichier suivant.
        lignes = lignes_sans_commentaire(chemin.read_text(encoding="utf-8"))  # Analyse le fichier.
        if lignes:  # Vérifie la présence de défauts.
            defauts[str(chemin.relative_to(racine))] = lignes  # Enregistre les défauts.
    return defauts  # Retourne le bilan.


def main() -> int:  # Définit le point d'entrée du script.
    """Affiche le bilan et retourne un code d'erreur s'il existe des lignes sans commentaire."""
    racine = Path(__file__).resolve().parents[1]  # Calcule la racine du projet.
    defauts = analyser_projet(racine)  # Analyse le projet.
    for fichier, lignes in defauts.items():  # Parcourt les fichiers en défaut.
        print(f"{fichier} : {len(lignes)} lignes sans commentaire, par exemple {lignes[:10]}")  # Affiche le détail.
    print("Tous les fichiers sont conformes." if not defauts else f"{len(defauts)} fichiers à corriger.")  # Affiche le bilan.
    return 1 if defauts else 0  # Retourne le code de sortie.


if __name__ == "__main__":  # Vérifie que le fichier est exécuté directement.
    sys.exit(main())  # Termine avec le code calculé.
