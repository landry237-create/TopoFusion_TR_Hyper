# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Boucle d'entraînement reproductible et contrôlée.

Le jeu de test n'apparaît jamais dans ce module : le choix du point de contrôle et l'arrêt
précoce utilisent uniquement la validation.
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

import json  # Écrit l'historique d'entraînement.
import math  # Calcule le planning cosinus.
import time  # Mesure les durées.
from pathlib import Path  # Manipule les chemins indépendamment du système.
from typing import Any, Dict, List, Optional  # Importe les types des signatures.

import numpy as np  # Fournit le générateur de lots.
import torch  # Fournit les tenseurs et l'optimiseur.

from data.feature_cache import FeatureSplit  # Importe l'accès aux caractéristiques.
from evaluation.embedding import evaluate_retrieval  # Importe l'évaluation de la recherche.
from losses.losses import CompositeLoss  # Importe l'objectif composite.
from models.multimodal_framework import TopoFusionModel, VariantSpec, count_parameters, modality_inputs  # Importe le modèle.
from models.robustness import FeatureRobustnessModule  # Importe les perturbations d'entraînement.
from utils.config import config_hash  # Importe l'empreinte de configuration.
from utils.seed import capture_rng_state, restore_rng_state, set_global_seed  # Importe la gestion des graines.


class NumericalInstabilityError(RuntimeError):  # Définit l'erreur d'instabilité numérique.
    """Signale des valeurs NaN ou Inf répétées pendant l'entraînement."""


def learning_rate_factor(etape: int, total: int, echauffement: float, minimum: float) -> float:  # Calcule le facteur du taux d'apprentissage.
    """Retourne le facteur du planning : montée linéaire, puis décroissance cosinus."""
    nb_echauffement = max(1, int(total * echauffement))  # Calcule la durée de l'échauffement.
    if etape < nb_echauffement:  # Vérifie si l'échauffement est en cours.
        return (etape + 1) / nb_echauffement  # Retourne la montée linéaire.
    progression = (etape - nb_echauffement) / max(1, total - nb_echauffement)  # Calcule l'avancement après l'échauffement.
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * min(progression, 1.0)))  # Retourne la décroissance cosinus.


def phase_of_epoch(epoque: int, total: int, fractions: List[float]) -> int:  # Détermine la phase courante.
    """Retourne la phase 1, 2 ou 3 correspondant à une époque numérotée à partir de 0."""
    fin_1 = int(round(total * fractions[0]))  # Calcule la fin de la phase 1.
    fin_2 = fin_1 + int(round(total * fractions[1]))  # Calcule la fin de la phase 2.
    return 1 if epoque < fin_1 else (2 if epoque < fin_2 else 3)  # Retourne la phase.


class Trainer:  # Pilote l'entraînement d'une variante.
    """Entraîne une variante sur le sous-ensemble d'entraînement et la sélectionne sur la validation."""

    def __init__(self, cfg: Dict[str, Any], variante: VariantSpec, train: FeatureSplit, val: FeatureSplit, dossier: Path, graine: int, appareil: Optional[torch.device] = None) -> None:  # Prépare l'entraînement.
        """Construit le modèle, l'optimiseur et les structures de suivi."""
        set_global_seed(graine)  # Fixe toutes les graines avant de créer le modèle.
        self.cfg = cfg  # Conserve la configuration.
        self.variante = variante  # Conserve la variante.
        self.train_split = train  # Conserve l'entraînement.
        self.val_split = val  # Conserve la validation.
        self.dossier = Path(dossier)  # Conserve le dossier de sortie.
        self.dossier.mkdir(parents=True, exist_ok=True)  # Crée le dossier de sortie.
        self.graine = graine  # Conserve la graine.
        self.appareil = appareil or torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choisit le device.
        print(f"[{variante.nom} | graine {graine}] Device réel : {self.appareil}", flush=True)  # Journalise le device réel.
        tr = cfg["training"]  # Récupère la section d'entraînement.
        self.tr = tr  # Conserve la section d'entraînement.
        dims = {"image": int(train.images.shape[-1]), "text": int(train.textes.shape[-1])}  # Lit les dimensions réelles des caractéristiques.
        self.modele = TopoFusionModel(cfg, variante, dims).to(self.appareil)  # Construit le modèle.
        self.critere = CompositeLoss(cfg, variante)  # Construit l'objectif composite.
        self.robustesse = FeatureRobustnessModule(tr["feature_noise_std"], tr["feature_mask_prob"], tr["modality_dropout_prob"])  # Construit les perturbations.
        self.optimiseur = torch.optim.AdamW(self.modele.parameter_groups(tr["weight_decay"]), lr=tr["lr"])  # Construit l'optimiseur.
        self.etapes_par_epoque = train.nb_images // int(tr["batch_size"])  # Calcule le nombre d'étapes par époque.
        self.total_etapes = self.etapes_par_epoque * int(tr["epochs"])  # Calcule le nombre total d'étapes.
        self.planning = torch.optim.lr_scheduler.LambdaLR(self.optimiseur, lambda e: learning_rate_factor(e, self.total_etapes, tr["warmup_ratio"], tr["min_lr_ratio"]))  # Construit le planning.
        self.generateur_lots = np.random.default_rng(graine)  # Crée le générateur de lots.
        self.historique: Dict[str, List[Any]] = {"train": [], "val": [], "normes_gradient": []}  # Prépare l'historique.
        self.meilleur = -math.inf  # Initialise le meilleur score.
        self.meilleure_epoque = -1  # Initialise la meilleure époque.
        self.sans_progres = 0  # Initialise le compteur de patience.
        self.etapes_invalides = 0  # Initialise le compteur d'étapes invalides.
        self.epoque_depart = 0  # Initialise l'époque de reprise.

    def _sauvegarder(self, nom: str, epoque: int) -> None:  # Sauvegarde un point de contrôle.
        """Écrit l'état complet : modèle, optimiseur, planning, générateurs et historique."""
        temporaire = self.dossier / f"{nom}.pt.part"  # Prépare une écriture atomique.
        torch.save({  # Écrit le point de contrôle.
            "modele": self.modele.state_dict(),  # État du modèle.
            "optimiseur": self.optimiseur.state_dict(),  # État de l'optimiseur.
            "planning": self.planning.state_dict(),  # État du planning.
            "epoque": epoque,  # Dernière époque terminée.
            "meilleur": self.meilleur,  # Meilleur score de validation.
            "meilleure_epoque": self.meilleure_epoque,  # Meilleure époque.
            "sans_progres": self.sans_progres,  # État de la patience.
            "historique": self.historique,  # Historique complet.
            "rng": capture_rng_state(),  # États des générateurs globaux.
            "rng_lots": self.generateur_lots.bit_generator.state,  # État du générateur de lots.
            "variante": self.variante.nom,  # Nom de la variante.
            "empreinte_config": config_hash(self.cfg),  # Empreinte de la configuration.
        }, temporaire)  # Termine la sauvegarde.
        temporaire.replace(self.dossier / f"{nom}.pt")  # Publie le fichier complet.

    def resume(self) -> bool:  # Reprend un entraînement interrompu.
        """Reprend depuis last.pt si ce fichier existe et correspond à la même configuration."""
        chemin = self.dossier / "last.pt"  # Calcule le chemin du dernier point de contrôle.
        if not chemin.is_file():  # Vérifie la présence du fichier.
            return False  # Signale l'absence de reprise.
        etat = torch.load(chemin, map_location=self.appareil, weights_only=False)  # Relit le point de contrôle, produit localement par ce projet.
        if etat["empreinte_config"] != config_hash(self.cfg):  # Vérifie la configuration.
            raise ValueError(f"Reprise refusée : {chemin} provient d'une autre configuration.")  # Explique le refus.
        self.modele.load_state_dict(etat["modele"])  # Restaure le modèle.
        self.optimiseur.load_state_dict(etat["optimiseur"])  # Restaure l'optimiseur.
        self.planning.load_state_dict(etat["planning"])  # Restaure le planning.
        self.historique = etat["historique"]  # Restaure l'historique.
        self.meilleur = etat["meilleur"]  # Restaure le meilleur score.
        self.meilleure_epoque = etat["meilleure_epoque"]  # Restaure la meilleure époque.
        self.sans_progres = etat["sans_progres"]  # Restaure la patience.
        self.epoque_depart = int(etat["epoque"]) + 1  # Fixe l'époque de reprise.
        restore_rng_state(etat["rng"])  # Restaure les générateurs globaux.
        self.generateur_lots.bit_generator.state = etat["rng_lots"]  # Restaure le générateur de lots.
        print(f"Reprise à l'époque {self.epoque_depart} depuis {chemin}", flush=True)  # Journalise la reprise.
        return True  # Signale la reprise.

    def _normes_par_module(self) -> Dict[str, float]:  # Mesure les normes de gradient.
        """Retourne la norme du gradient de chaque grand module, pour détecter un module mort."""
        carres: Dict[str, float] = {}  # Prépare les sommes de carrés.
        for nom, parametre in self.modele.named_parameters():  # Parcourt les paramètres.
            if parametre.grad is not None:  # Vérifie la présence d'un gradient.
                groupe = nom.split(".")[0]  # Regroupe par module de premier niveau.
                carres[groupe] = carres.get(groupe, 0.0) + float(parametre.grad.pow(2).sum().item())  # Ajoute le carré de la norme.
        return {nom: math.sqrt(valeur) for nom, valeur in carres.items()}  # Retourne les normes.

    def _etape(self, lot: Dict[str, torch.Tensor], phase: int) -> Optional[Dict[str, float]]:  # Exécute une étape d'optimisation.
        """Exécute une étape ; une étape non finie est ignorée puis comptée, sans corrompre les poids."""
        self.optimiseur.zero_grad(set_to_none=True)  # Efface les gradients précédents.
        entrees = modality_inputs(lot)  # Convertit le lot.
        disponibilite = self.robustesse.availability(lot["image_tokens"].size(0), self.modele.modalites, self.appareil) if self.variante.perturbations and phase == 3 else None  # Tire les modalités présentes dans la fusion.
        sorties = self.modele(entrees, disponibilite)  # Exécute le modèle sur les données propres.
        perturbees = None  # Initialise la vue perturbée.
        if self.variante.perturbations and phase == 3:  # Vérifie si la robustesse est active.
            masque_image = torch.ones(lot["image_tokens"].shape[:2], dtype=torch.bool, device=self.appareil)  # Prépare le masque des images.
            perturbees = self.modele({"image": self.robustesse.perturb(lot["image_tokens"], masque_image), "text": self.robustesse.perturb(lot["text_tokens"], lot["text_mask"])}, fusionner=False)  # Exécute le modèle sur les données perturbées.
        perte, details = self.critere(sorties, perturbees, self.modele.temperature, lot["text_ids"], phase)  # Calcule l'objectif.
        if not torch.isfinite(perte):  # Vérifie que l'objectif est fini.
            return self._etape_invalide(f"objectif non fini : {details}")  # Traite l'étape invalide.
        perte.backward()  # Calcule les gradients.
        norme = torch.nn.utils.clip_grad_norm_(self.modele.parameters(), self.tr["max_grad_norm"])  # Écrête les gradients.
        if not torch.isfinite(norme):  # Vérifie que le gradient est fini.
            return self._etape_invalide("gradient non fini")  # Traite l'étape invalide.
        self.optimiseur.step()  # Met à jour les poids.
        self.planning.step()  # Met à jour le taux d'apprentissage.
        details["norme_gradient"] = float(norme.item())  # Conserve la norme avant écrêtage.
        return details  # Retourne le détail des pertes.

    def _etape_invalide(self, motif: str) -> None:  # Traite une étape invalide.
        """Compte une étape invalide et interrompt l'entraînement si l'instabilité se répète."""
        self.etapes_invalides += 1  # Met à jour le compteur.
        print(f"Étape ignorée ({self.etapes_invalides}/{self.tr['max_bad_steps']}) : {motif}", flush=True)  # Journalise le motif.
        if self.etapes_invalides > int(self.tr["max_bad_steps"]):  # Vérifie la limite autorisée.
            raise NumericalInstabilityError(f"Instabilité numérique répétée : {motif}")  # Interrompt l'entraînement.
        return None  # Signale l'absence de résultat.

    def fit(self) -> Dict[str, Any]:  # Exécute tout l'entraînement.
        """Entraîne le modèle et sélectionne la meilleure époque sur la validation."""
        tr = self.tr  # Raccourcit l'accès aux paramètres.
        taille_eval = int(self.cfg["evaluation"]["batch_size"])  # Lit la taille des lots d'évaluation.
        ks = self.cfg["evaluation"]["recall_ks"]  # Lit les rangs de coupure.
        metrique = tr["early_stopping_metric"].replace("val_", "")  # Lit la métrique de sélection.
        fractions = tr["phase_fractions"]  # Lit les fractions des phases.
        premiere_phase3 = next((e for e in range(int(tr["epochs"])) if phase_of_epoch(e, int(tr["epochs"]), fractions) == 3), 0)  # Calcule le début de la phase 3.
        epoque_selection = premiere_phase3 if {"top", "rob"} & set(self.variante.pertes) else 0  # Interdit de sélectionner un état qui n'a pas vu l'objectif complet.
        if not self.historique["val"]:  # Vérifie qu'aucune évaluation initiale n'existe.
            self.historique["val"].append({"epoque": -1, "phase": 0, **evaluate_retrieval(self.modele, self.val_split, taille_eval, self.appareil, ks)})  # Évalue le modèle avant entraînement.
        debut_total = time.time()  # Mémorise l'heure de début.
        for epoque in range(self.epoque_depart, int(tr["epochs"])):  # Parcourt les époques.
            phase = phase_of_epoch(epoque, int(tr["epochs"]), fractions)  # Détermine la phase courante.
            self.modele.train()  # Active le dropout.
            debut = time.time()  # Mémorise l'heure de début de l'époque.
            cumuls: Dict[str, float] = {}  # Prépare les cumuls de pertes.
            nb_etapes = 0  # Initialise le compteur d'étapes valides.
            normes: Dict[str, float] = {}  # Prépare les normes de gradient.
            for lot in self.train_split.iter_train_batches(int(tr["batch_size"]), self.generateur_lots, self.appareil):  # Parcourt les lots.
                details = self._etape(lot, phase)  # Exécute une étape.
                if details is None:  # Vérifie si l'étape a été ignorée.
                    continue  # Passe au lot suivant.
                nb_etapes += 1  # Compte l'étape valide.
                for cle, valeur in details.items():  # Parcourt les termes.
                    cumuls[cle] = cumuls.get(cle, 0.0) + valeur  # Ajoute la valeur au cumul.
                normes = self._normes_par_module()  # Conserve les normes de la dernière étape.
            duree = time.time() - debut  # Calcule la durée de l'époque.
            moyenne_train = {cle: valeur / max(nb_etapes, 1) for cle, valeur in cumuls.items()}  # Calcule les moyennes.
            self.historique["train"].append({"epoque": epoque, "phase": phase, "duree_s": duree, "lr": self.optimiseur.param_groups[0]["lr"], "temperature": float(self.modele.temperature.item()), "porte": float(self.modele.porte.item()), **moyenne_train})  # Enregistre l'entraînement.
            self.historique["normes_gradient"].append({"epoque": epoque, **normes})  # Enregistre les normes de gradient.
            scores = evaluate_retrieval(self.modele, self.val_split, taille_eval, self.appareil, ks)  # Évalue sur la validation.
            perte_val = self._perte_validation(phase)  # Calcule la perte de validation.
            self.historique["val"].append({"epoque": epoque, "phase": phase, **scores, **{f"perte_{c}": v for c, v in perte_val.items()}})  # Enregistre la validation.
            print(f"[{self.variante.nom} | graine {self.graine}] époque {epoque + 1}/{tr['epochs']} phase {phase} perte={moyenne_train.get('total', float('nan')):.4f} val_perte={perte_val.get('total', float('nan')):.4f} val_R@1={scores['mean_recall_at_1']:.4f} val_R@5={scores['mean_recall_at_5']:.4f} ({duree:.0f} s)", flush=True)  # Journalise l'époque.
            score = scores[metrique]  # Lit le score de sélection.
            if epoque >= epoque_selection and score > self.meilleur:  # Vérifie une amélioration après le début de l'objectif complet.
                self.meilleur = score  # Met à jour le meilleur score.
                self.meilleure_epoque = epoque  # Met à jour la meilleure époque.
                self.sans_progres = 0  # Réinitialise la patience.
                self._sauvegarder("best", epoque)  # Sauvegarde le meilleur modèle.
            elif epoque >= epoque_selection:  # Traite une absence d'amélioration.
                self.sans_progres += 1  # Met à jour la patience.
            self._sauvegarder("last", epoque)  # Sauvegarde l'état courant pour la reprise.
            if self.sans_progres >= int(tr["early_stopping_patience"]):  # Vérifie l'arrêt précoce.
                print(f"Arrêt précoce à l'époque {epoque + 1} : meilleure époque {self.meilleure_epoque + 1}.", flush=True)  # Journalise l'arrêt.
                break  # Termine l'entraînement.
        duree_totale = time.time() - debut_total  # Calcule la durée totale.
        self._charger_meilleur()  # Recharge le meilleur modèle.
        resume = {"variante": self.variante.nom, "graine": self.graine, "meilleure_epoque": self.meilleure_epoque, "meilleur_score_val": self.meilleur, "metrique_selection": metrique, "epoque_debut_selection": epoque_selection, "duree_entrainement_s": duree_totale, "etapes_invalides": self.etapes_invalides, **count_parameters(self.modele)}  # Résume l'entraînement.
        (self.dossier / "historique.json").write_text(json.dumps({"resume": resume, **self.historique}, ensure_ascii=False, indent=1), encoding="utf-8")  # Écrit l'historique.
        return resume  # Retourne le résumé.

    @torch.no_grad()  # Désactive le calcul du gradient.
    def _perte_validation(self, phase: int) -> Dict[str, float]:  # Calcule la perte de validation.
        """Calcule l'objectif sur la validation, avec des perturbations à graine fixe pour rester comparable."""
        self.modele.eval()  # Désactive le dropout.
        generateur = torch.Generator(device="cpu").manual_seed(self.graine + 1000)  # Fixe les perturbations de validation.
        cumuls: Dict[str, float] = {}  # Prépare les cumuls.
        nb = 0  # Initialise le compteur de lots.
        for lot in self.val_split.iter_eval_pairs(int(self.tr["batch_size"]), self.appareil):  # Parcourt les paires de validation.
            sorties = self.modele(modality_inputs(lot))  # Exécute le modèle.
            perturbees = None  # Initialise la vue perturbée.
            if self.variante.perturbations and phase == 3:  # Vérifie si la robustesse est active.
                masque_image = torch.ones(lot["image_tokens"].shape[:2], dtype=torch.bool, device=self.appareil)  # Prépare le masque des images.
                perturbees = self.modele({"image": self.robustesse.perturb(lot["image_tokens"], masque_image, generateur), "text": self.robustesse.perturb(lot["text_tokens"], lot["text_mask"], generateur)}, fusionner=False)  # Exécute le modèle sur les données perturbées.
            _, details = self.critere(sorties, perturbees, self.modele.temperature, lot["text_ids"], phase)  # Calcule l'objectif.
            for cle, valeur in details.items():  # Parcourt les termes.
                cumuls[cle] = cumuls.get(cle, 0.0) + valeur  # Ajoute la valeur au cumul.
            nb += 1  # Compte le lot.
        return {cle: valeur / max(nb, 1) for cle, valeur in cumuls.items()}  # Retourne les moyennes.

    def _charger_meilleur(self) -> None:  # Recharge le meilleur modèle.
        """Recharge le meilleur point de contrôle selon la validation."""
        chemin = self.dossier / "best.pt"  # Calcule le chemin du meilleur modèle.
        if chemin.is_file():  # Vérifie la présence du fichier.
            self.modele.load_state_dict(torch.load(chemin, map_location=self.appareil, weights_only=False)["modele"])  # Restaure le meilleur modèle.
            self.modele.eval()  # Place le modèle en mode évaluation.
