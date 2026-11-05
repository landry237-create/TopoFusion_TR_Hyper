# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Perturbations d'entraînement appliquées aux caractéristiques et absence simulée de modalité.

Ces perturbations servent à l'entraînement. L'évaluation de robustesse utilise, elle,
de vraies corruptions d'images et de textes (voir data/corruptions.py).
"""

from __future__ import annotations  # Reporte l'évaluation des annotations de type.

from typing import Dict, Optional, Sequence, Tuple  # Importe les types des signatures.

import torch  # Fournit les tenseurs.


class FeatureRobustnessModule:  # Regroupe les perturbations d'entraînement.
    """Perturbe les tokens gelés et masque des modalités sans jamais créer de fausse donnée."""

    def __init__(self, ecart_type_bruit: float, proba_masque: float, proba_abandon: float) -> None:  # Mémorise les paramètres.
        """Mémorise l'amplitude du bruit et les probabilités de masquage."""
        self.ecart_type_bruit = float(ecart_type_bruit)  # Conserve l'écart-type relatif du bruit.
        self.proba_masque = float(proba_masque)  # Conserve la probabilité de masquer un token.
        self.proba_abandon = float(proba_abandon)  # Conserve la probabilité d'abandonner une modalité.

    def perturb(self, tokens: torch.Tensor, masque: torch.Tensor, generateur: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:  # Perturbe une modalité.
        """Ajoute un bruit relatif à l'échelle des tokens et retire des tokens en conservant toujours le premier."""
        echelle = tokens.detach().std(dim=(1, 2), keepdim=True)  # Mesure l'échelle des tokens de chaque échantillon.
        bruit = torch.randn(tokens.shape, generator=generateur, device=tokens.device, dtype=tokens.dtype)  # Tire le bruit gaussien.
        sortie = tokens + self.ecart_type_bruit * echelle * bruit  # Ajoute le bruit relatif.
        tirages = torch.rand(masque.shape, generator=generateur, device=masque.device)  # Tire une valeur par token.
        nouveau_masque = masque & (tirages >= self.proba_masque)  # Retire des tokens.
        nouveau_masque[:, 0] = masque[:, 0]  # Conserve toujours le premier token de chaque échantillon.
        return sortie, nouveau_masque  # Retourne les tokens et le masque perturbés.

    def availability(self, taille_lot: int, modalites: Sequence[str], appareil: torch.device, generateur: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:  # Tire les modalités présentes.
        """Tire les modalités présentes dans la fusion, avec au moins une modalité réelle par échantillon."""
        presence = torch.rand((taille_lot, len(modalites)), generator=generateur, device=appareil) >= self.proba_abandon  # Tire la présence de chaque modalité.
        vides = ~presence.any(dim=1)  # Détecte les échantillons sans modalité.
        if vides.any():  # Vérifie si un échantillon doit être corrigé.
            secours = torch.randint(0, len(modalites), (taille_lot,), generator=generateur, device=appareil)  # Tire la modalité à conserver.
            presence[vides, secours[vides]] = True  # Rétablit une modalité réelle.
        return {nom: presence[:, rang] for rang, nom in enumerate(modalites)}  # Retourne la présence par modalité.
