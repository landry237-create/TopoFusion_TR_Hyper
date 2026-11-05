# -*- coding: utf-8 -*-  # Déclare l'encodage UTF-8 du fichier source.
"""Vérifie le moteur topologique sur des cas analytiques et par différences finies."""

import gudhi  # Fournit la distance de bottleneck exacte.
import numpy as np  # Construit les nuages de points de référence.
import pytest  # Fournit le cadre de test.
import ripser  # Fournit un second calcul indépendant de la persistance.
import torch  # Fournit les tenseurs différentiables.

from models.topology import (  # Importe les fonctions à vérifier.
    AdaptiveLatentGraph,  # Graphe latent adaptatif.
    ExactSpectralLinear,  # Couche à norme spectrale bornée.
    LipschitzFiltration,  # Filtration de Rips pondérée.
    TopologicalEncoder,  # Encodeur des diagrammes.
    compute_flag_pairs,  # Persistance exacte.
    critical_edges,  # Arêtes critiques.
    differentiable_persistence,  # Persistance différentiable.
    exact_diagrams,  # Diagrammes exacts.
    persistence_landscape,  # Paysages de persistance.
    single_linkage_edges,  # Ultramétrique de liaison simple.
    landscape_l2_distance,  # Distance L2 entre paysages.
    soft_betti_curve,  # Courbes de Betti douces.
    topological_bias,  # Biais topologique.
)  # Termine l'importation.


def _distances(points: np.ndarray) -> np.ndarray:  # Calcule une matrice de distances.
    """Retourne la matrice des distances euclidiennes d'un nuage."""
    return np.linalg.norm(points[:, None] - points[None], axis=-1)  # Calcule les distances par paires.


def _cercle(n: int = 40, rayon: float = 1.0) -> np.ndarray:  # Construit un cercle.
    """Retourne n points régulièrement espacés sur un cercle."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)  # Répartit les angles.
    return rayon * np.stack([np.cos(angles), np.sin(angles)], axis=1)  # Convertit en coordonnées cartésiennes.


def test_h0_deux_amas_analytique():  # Vérifie H0 sur deux amas.
    """Deux amas séparés donnent une composante essentielle et une fusion à la distance entre amas."""
    points = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0]])  # Construit les deux amas.
    diagramme = exact_diagrams(_distances(points))[0]  # Calcule H0.
    assert np.isinf(diagramme[:, 1]).sum() == 1  # Vérifie qu'une seule composante reste essentielle.
    morts = np.sort(diagramme[np.isfinite(diagramme[:, 1]), 1])  # Récupère les morts finies.
    assert morts == pytest.approx([0.1, 0.1, 4.9], abs=1e-9)  # Compare aux valeurs théoriques.


def test_h1_triangle_tue_immediatement():  # Vérifie H1 sur un triangle.
    """Un triangle équilatéral se remplit à l'apparition de sa dernière arête : il n'existe aucun cycle persistant."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])  # Construit le triangle.
    assert exact_diagrams(_distances(points))[1].shape[0] == 0  # Vérifie l'absence de cycle.


def test_h1_cercle_un_cycle_dominant():  # Vérifie H1 sur un cercle.
    """Un cercle échantillonné possède un unique cycle de grande persistance."""
    diagramme = exact_diagrams(_distances(_cercle()))[1]  # Calcule H1.
    durees = np.sort(diagramme[:, 1] - diagramme[:, 0])[::-1]  # Trie les durées de vie.
    assert durees[0] > 1.0  # Vérifie la présence du cycle principal.
    assert len(durees) == 1 or durees[1] < 0.1 * durees[0]  # Vérifie l'absence de second cycle important.


@pytest.mark.parametrize("graine", [0, 1, 2])  # Répète le test sur trois nuages.
def test_gudhi_coincide_avec_ripser(graine):  # Compare deux implémentations indépendantes.
    """Le calcul exact doit coïncider avec Ripser, une implémentation indépendante."""
    points = np.random.default_rng(graine).normal(size=(30, 8))  # Tire un nuage aléatoire.
    distances = _distances(points)  # Calcule les distances.
    notre = exact_diagrams(distances)  # Calcule les diagrammes avec notre moteur.
    reference = ripser.ripser(distances, distance_matrix=True, maxdim=1)["dgms"]  # Calcule les diagrammes avec Ripser.
    for dimension in (0, 1):  # Parcourt H0 et H1.
        a = notre[dimension][np.isfinite(notre[dimension][:, 1])]  # Garde nos points finis.
        b = reference[dimension][np.isfinite(reference[dimension][:, 1])]  # Garde les points finis de Ripser.
        assert a.shape[0] == b.shape[0]  # Compare le nombre de points.
        assert gudhi.bottleneck_distance(a, b) < 1e-4  # Compare les diagrammes.


def test_classes_essentielles_restent_infinies():  # Vérifie l'absence de mort inventée.
    """Une classe essentielle garde une mort infinie : aucune valeur finie n'est inventée."""
    diagramme = exact_diagrams(_distances(_cercle(12)))[0]  # Calcule H0.
    assert np.isinf(diagramme[:, 1]).sum() == 1  # Vérifie la présence d'une seule mort infinie.


def test_valeurs_differentiables_identiques_aux_valeurs_exactes():  # Compare les niveaux A et B.
    """Le niveau différentiable doit redonner exactement les diagrammes du niveau exact."""
    torch.manual_seed(0)  # Fixe la graine.
    filtration = LipschitzFiltration(32, 16, 4)  # Crée la filtration.
    points = torch.nn.functional.normalize(torch.randn(3, 14, 32), dim=-1)  # Tire des points sur la sphère.
    matrice = filtration(points)  # Calcule la filtration.
    lot = differentiable_persistence(matrice)  # Calcule la persistance différentiable.
    for rang in range(3):  # Parcourt les échantillons.
        exact = exact_diagrams(matrice[rang].detach().numpy().astype(np.float64))  # Calcule les diagrammes exacts.
        h1 = torch.stack([lot.h1_birth[rang], lot.h1_death[rang]], dim=1)[lot.h1_mask[rang]].detach().numpy()  # Lit H1 différentiable.
        assert gudhi.bottleneck_distance(h1.astype(np.float64), exact[1]) < 1e-5  # Compare H1.
        h0 = torch.stack([lot.h0_birth[rang], lot.h0_death[rang]], dim=1)[lot.h0_mask[rang]].detach().numpy()  # Lit H0 différentiable.
        assert gudhi.bottleneck_distance(h0.astype(np.float64), exact[0][np.isfinite(exact[0][:, 1])]) < 1e-5  # Compare H0.


def _persistance_totale_exacte(points: np.ndarray) -> float:  # Calcule la persistance totale exacte.
    """Calcule H0 + H1 de la persistance totale du Rips, sans passer par PyTorch."""
    diagrammes = exact_diagrams(_distances(points))  # Calcule les diagrammes exacts.
    total = 0.0  # Initialise le total.
    for dimension in (0, 1):  # Parcourt H0 et H1.
        fini = diagrammes[dimension][np.isfinite(diagrammes[dimension][:, 1])]  # Garde les points finis.
        total += float((fini[:, 1] - fini[:, 0]).sum())  # Ajoute leurs durées de vie.
    return total  # Retourne le total.


def test_gradient_autograd_coincide_avec_differences_finies():  # Vérifie le gradient réel.
    """Le gradient obtenu par relecture doit correspondre au gradient numérique de la persistance exacte."""
    generateur = np.random.default_rng(3)  # Fixe le générateur.
    points = generateur.normal(size=(12, 3))  # Tire un nuage générique, sans égalité de distances.
    tenseur = torch.tensor(points, dtype=torch.float64, requires_grad=True)  # Crée le tenseur différentiable.
    distances = torch.sqrt((tenseur.unsqueeze(0) - tenseur.unsqueeze(1)).pow(2).sum(-1) + 1e-18)  # Calcule les distances.
    matrice = distances * (1.0 - torch.eye(12, dtype=torch.float64))  # Annule la diagonale.
    lot = differentiable_persistence(matrice.unsqueeze(0))  # Calcule la persistance.
    total = ((lot.h0_death - lot.h0_birth) * lot.h0_mask).sum() + ((lot.h1_death - lot.h1_birth) * lot.h1_mask).sum()  # Calcule la persistance totale.
    total.backward()  # Rétropropage le gradient.
    analytique = tenseur.grad.numpy().copy()  # Récupère le gradient automatique.
    assert np.abs(analytique).sum() > 0.0  # Vérifie que le gradient n'est pas nul.
    pas = 1e-6  # Fixe le pas des différences finies.
    for i, j in [(0, 0), (3, 1), (7, 2), (11, 0)]:  # Vérifie plusieurs coordonnées.
        plus = points.copy()  # Copie le nuage.
        plus[i, j] += pas  # Déplace la coordonnée vers le haut.
        moins = points.copy()  # Copie le nuage.
        moins[i, j] -= pas  # Déplace la coordonnée vers le bas.
        numerique = (_persistance_totale_exacte(plus) - _persistance_totale_exacte(moins)) / (2.0 * pas)  # Calcule la dérivée numérique.
        assert analytique[i, j] == pytest.approx(numerique, abs=1e-4)  # Compare les deux dérivées.


def test_gradient_atteint_la_filtration_et_les_points():  # Vérifie la propagation du gradient.
    """Un objectif calculé sur la persistance doit produire un gradient sur les points et sur la filtration."""
    torch.manual_seed(1)  # Fixe la graine.
    filtration = LipschitzFiltration(32, 16, 4)  # Crée la filtration.
    points = torch.nn.functional.normalize(torch.randn(2, 12, 32), dim=-1).requires_grad_(True)  # Crée des points différentiables.
    lot = differentiable_persistence(filtration(points))  # Calcule la persistance.
    ((lot.h1_death - lot.h1_birth) * lot.h1_mask).sum().backward()  # Rétropropage depuis H1 uniquement.
    assert points.grad is not None and points.grad.abs().sum() > 0.0  # Vérifie le gradient sur les points.
    normes = [p.grad.abs().sum().item() for p in filtration.metrique.parameters() if p.grad is not None]  # Lit les gradients de la métrique.
    assert normes and max(normes) > 0.0  # Vérifie le gradient sur la métrique apprise.


def test_norme_spectrale_exacte_bornee_par_un():  # Vérifie la contrainte de Lipschitz.
    """La norme spectrale exacte doit rester bornée par 1, même après des poids volontairement grands."""
    torch.manual_seed(2)  # Fixe la graine.
    couche = ExactSpectralLinear(64, 32)  # Crée la couche.
    with torch.no_grad():  # Désactive le gradient pendant la modification.
        couche.poids.mul_(25.0)  # Augmente fortement les poids.
    assert couche.norme_spectrale() <= 1.0 + 1e-5  # Vérifie la borne.


def test_borne_de_stabilite_verifiee_numeriquement():  # Vérifie la borne démontrée.
    """La distance de bottleneck doit respecter d_B <= (L_phi + L_w) * max ||u_i - u'_i||."""
    torch.manual_seed(4)  # Fixe la graine.
    filtration = LipschitzFiltration(32, 16, 4)  # Crée la filtration.
    borne = filtration.lipschitz_bound()["borne_stabilite"]  # Lit la constante de stabilité.
    assert borne <= 2.0 + 1e-4  # Vérifie la constante attendue.
    for amplitude in (1e-3, 1e-2, 1e-1, 0.5):  # Parcourt plusieurs amplitudes.
        for _ in range(5):  # Répète chaque essai.
            u = torch.nn.functional.normalize(torch.randn(1, 14, 32), dim=-1)  # Tire un nuage propre.
            v = torch.nn.functional.normalize(u + amplitude * torch.randn_like(u), dim=-1)  # Tire un nuage perturbé.
            delta = float((u - v).norm(dim=-1).max())  # Calcule la perturbation maximale.
            with torch.no_grad():  # Désactive le gradient.
                f_u = filtration(u)[0].numpy().astype(np.float64)  # Calcule la filtration propre.
                f_v = filtration(v)[0].numpy().astype(np.float64)  # Calcule la filtration perturbée.
            assert np.abs(f_u - f_v).max() <= borne * delta + 1e-4  # Vérifie la borne en norme infinie.
            for dimension in (0, 1):  # Parcourt H0 et H1.
                a = exact_diagrams(f_u)[dimension]  # Calcule le diagramme propre.
                b = exact_diagrams(f_v)[dimension]  # Calcule le diagramme perturbé.
                a = a[np.isfinite(a[:, 1])]  # Garde les points finis.
                b = b[np.isfinite(b[:, 1])]  # Garde les points finis.
                assert gudhi.bottleneck_distance(a, b) <= borne * delta + 1e-4  # Vérifie la borne de stabilité.


def test_filtration_monotone_et_positive():  # Vérifie la validité de la filtration.
    """Une arête doit apparaître après ses sommets, sinon la filtration n'est pas valide."""
    torch.manual_seed(5)  # Fixe la graine.
    filtration = LipschitzFiltration(32, 16, 4)  # Crée la filtration.
    matrice = filtration(torch.nn.functional.normalize(torch.randn(2, 10, 32), dim=-1))  # Calcule la matrice.
    sommets = torch.diagonal(matrice, dim1=1, dim2=2)  # Lit les valeurs des sommets.
    assert (sommets >= 0.0).all()  # Vérifie la positivité.
    assert (matrice >= sommets.unsqueeze(1) - 1e-6).all() and (matrice >= sommets.unsqueeze(2) - 1e-6).all()  # Vérifie la monotonie.
    assert torch.allclose(matrice, matrice.transpose(1, 2))  # Vérifie la symétrie.
    assert (torch.diff(filtration.echelles) > 0.0).all()  # Vérifie l'ordre des échelles.


def test_masque_ignore_le_padding():  # Vérifie la gestion du padding.
    """Les positions masquées ne doivent changer ni le diagramme ni les indices critiques."""
    torch.manual_seed(6)  # Fixe la graine.
    filtration = LipschitzFiltration(16, 8, 4)  # Crée la filtration.
    points = torch.nn.functional.normalize(torch.randn(1, 10, 16), dim=-1)  # Tire un nuage.
    masque = torch.tensor([[True] * 6 + [False] * 4])  # Masque les quatre derniers points.
    lot = differentiable_persistence(filtration(points), masque)  # Calcule avec le masque.
    reference = differentiable_persistence(filtration(points[:, :6]))  # Calcule sur le nuage réduit.
    assert torch.allclose(lot.h0_death[lot.h0_mask].sort().values, reference.h0_death[reference.h0_mask].sort().values)  # Compare H0.
    for paire in lot.paires[0].h0_death.tolist() + lot.paires[0].h1_birth.tolist():  # Parcourt les arêtes critiques.
        assert max(paire) < 6  # Vérifie qu'aucune position masquée n'est utilisée.


def test_paysage_tente_unique():  # Vérifie le paysage sur un seul point.
    """Un point (b, d) donne une tente de hauteur (d - b) / 2 au milieu de la barre."""
    grille = torch.linspace(0.0, 2.0, 21)  # Crée la grille.
    p = persistence_landscape(torch.tensor([[0.5]]), torch.tensor([[1.5]]), torch.tensor([[True]]), grille, 2)  # Calcule le paysage.
    assert p.shape == (1, 2, 21)  # Vérifie la forme.
    assert p[0, 0].max().item() == pytest.approx(0.5, abs=1e-5)  # Vérifie la hauteur de la tente.
    assert p[0, 0, 10].item() == pytest.approx(0.5, abs=1e-5)  # Vérifie la position du sommet.
    assert p[0, 1].abs().max().item() == 0.0  # Vérifie que la deuxième couche est nulle.


def test_paysage_trie_meme_avec_peu_de_points():  # Vérifie la correction de l'ancien défaut M2.
    """Les couches doivent décroître en tout point, même si le nombre de points est inférieur aux couches."""
    grille = torch.linspace(0.0, 3.0, 31)  # Crée la grille.
    naissance = torch.tensor([[0.0, 1.0]])  # Crée deux naissances.
    mort = torch.tensor([[1.0, 3.0]])  # Crée deux morts.
    p = persistence_landscape(naissance, mort, torch.tensor([[True, True]]), grille, 3)  # Calcule trois couches.
    assert (p[0, 0] >= p[0, 1] - 1e-6).all() and (p[0, 1] >= p[0, 2] - 1e-6).all()  # Vérifie l'ordre des couches.


def test_distance_paysages_nulle_et_differentiable():  # Vérifie la distance entre paysages.
    """La distance entre paysages doit être nulle pour deux diagrammes identiques et transmettre un gradient sinon."""
    torch.manual_seed(9)  # Fixe la graine.
    filtration = LipschitzFiltration(32, 16, 4)  # Crée la filtration.
    u = torch.nn.functional.normalize(torch.randn(2, 12, 32), dim=-1).requires_grad_(True)  # Crée des points différentiables.
    v = torch.nn.functional.normalize(torch.randn(2, 12, 32), dim=-1)  # Crée un autre nuage.
    a = differentiable_persistence(filtration(u))  # Calcule le premier diagramme.
    assert landscape_l2_distance(a, a, 3, 16).item() == pytest.approx(0.0, abs=1e-10)  # Vérifie la nullité.
    distance = landscape_l2_distance(a, differentiable_persistence(filtration(v)), 3, 16)  # Calcule la distance entre deux nuages.
    assert distance.item() > 0.0  # Vérifie que deux nuages distincts sont distingués.
    distance.backward()  # Rétropropage le gradient.
    assert u.grad.abs().sum() > 0.0  # Vérifie le gradient sur les points.


def test_betti_douce_converge_vers_exacte():  # Compare l'approximation au calcul exact.
    """La courbe de Betti douce doit tendre vers la courbe exacte quand la température diminue."""
    naissance = torch.tensor([[0.0, 0.3, 0.6]])  # Crée trois naissances.
    mort = torch.tensor([[1.0, 0.8, 1.5]])  # Crée trois morts.
    masque = torch.tensor([[True, True, True]])  # Valide les trois points.
    echelles = torch.tensor([0.1, 0.5, 0.7, 1.2])  # Crée quatre échelles.
    exacte = ((echelles.unsqueeze(0) >= naissance.T) & (echelles.unsqueeze(0) < mort.T)).sum(dim=0).float()  # Calcule la courbe exacte.
    erreurs = [(soft_betti_curve(naissance, mort, masque, echelles, t)[0] - exacte).abs().max().item() for t in (0.1, 0.01, 0.001)]  # Calcule les erreurs.
    assert erreurs[0] > erreurs[1] > erreurs[2]  # Vérifie la décroissance de l'erreur.
    assert erreurs[2] < 1e-3  # Vérifie la convergence.


def test_ultrametrique_de_liaison_simple():  # Vérifie le biais issu de H0.
    """La hauteur de fusion doit être la valeur minimax, avec l'arête critique attendue."""
    d = np.array([[0.0, 1.0, 4.0, 5.0], [1.0, 0.0, 3.0, 6.0], [4.0, 3.0, 0.0, 2.0], [5.0, 6.0, 2.0, 0.0]])  # Crée une matrice de distances.
    ci, cj = single_linkage_edges(d)  # Calcule les arêtes critiques.
    assert d[ci[0, 1], cj[0, 1]] == 1.0  # Vérifie la fusion directe de 0 et 1.
    assert d[ci[0, 3], cj[0, 3]] == 3.0  # Vérifie le chemin minimax de 0 vers 3.
    assert d[ci[2, 3], cj[2, 3]] == 2.0  # Vérifie la fusion directe de 2 et 3.


@pytest.mark.parametrize("graine", [0, 1, 2])  # Répète le test sur trois nuages.
def test_ultrametrique_coincide_avec_scipy(graine):  # Compare à une implémentation indépendante.
    """L'ultramétrique vectorisée doit coïncider avec la distance cophénétique de la liaison simple de SciPy."""
    from scipy.cluster.hierarchy import cophenet, linkage  # Importe l'implémentation de référence.
    from scipy.spatial.distance import pdist, squareform  # Importe la conversion des distances.

    points = np.random.default_rng(graine).normal(size=(14, 6))  # Tire un nuage aléatoire.
    d = _distances(points)  # Calcule les distances.
    reference = squareform(cophenet(linkage(pdist(points), method="single")))  # Calcule la référence.
    ci, cj = single_linkage_edges(d)  # Calcule nos arêtes critiques.
    lignes, colonnes = np.triu_indices(14, k=1)  # Énumère les paires distinctes.
    assert d[ci[lignes, colonnes], cj[lignes, colonnes]] == pytest.approx(reference[lignes, colonnes], abs=1e-10)  # Compare les hauteurs.


def test_ultrametrique_par_lot_et_masque():  # Vérifie le lot et le masque.
    """Le calcul par lot doit coïncider avec le calcul isolé, et le padding ne doit pas servir de raccourci."""
    generateur = np.random.default_rng(4)  # Fixe le générateur.
    matrices = np.stack([_distances(generateur.normal(size=(9, 4))) for _ in range(3)])  # Tire trois matrices.
    ci, cj = single_linkage_edges(matrices)  # Calcule les arêtes par lot.
    for rang in range(3):  # Parcourt les matrices.
        ci_1, cj_1 = single_linkage_edges(matrices[rang])  # Calcule les arêtes d'une matrice isolée.
        assert (ci[rang] == ci_1).all() and (cj[rang] == cj_1).all()  # Compare les résultats.
    masque = np.array([True] * 6 + [False] * 3)  # Masque les trois derniers points.
    mi, mj = single_linkage_edges(matrices[0], masque)  # Calcule avec le masque.
    ri, rj = single_linkage_edges(matrices[0][:6, :6])  # Calcule sur le nuage réduit.
    assert (mi[:6, :6][~np.eye(6, dtype=bool)] == ri[~np.eye(6, dtype=bool)]).all()  # Compare le premier sommet critique.
    assert (mj[:6, :6][~np.eye(6, dtype=bool)] == rj[~np.eye(6, dtype=bool)]).all()  # Compare le second sommet critique.


def test_paires_coincident_avec_les_intervalles_gudhi_rips_pondere():  # Compare à une lecture indépendante.
    """Sur un Rips pondéré, les paires doivent redonner les intervalles que GUDHI calcule directement."""
    torch.manual_seed(11)  # Fixe la graine.
    filtration = LipschitzFiltration(16, 8, 4)  # Crée la filtration.
    matrice = filtration(torch.nn.functional.normalize(torch.randn(1, 12, 16), dim=-1))[0].detach().numpy().astype(np.float64)  # Calcule une filtration pondérée.
    arbre = gudhi.SimplexTree()  # Crée un arbre indépendant.
    for i in range(12):  # Parcourt les sommets.
        arbre.insert([i], filtration=float(matrice[i, i]))  # Insère le sommet.
    for i in range(12):  # Parcourt les premiers sommets des arêtes.
        for j in range(i + 1, 12):  # Parcourt les seconds sommets des arêtes.
            arbre.insert([i, j], filtration=float(matrice[i, j]))  # Insère l'arête.
    arbre.expansion(2)  # Ajoute les triangles.
    arbre.compute_persistence(homology_coeff_field=2)  # Calcule la persistance.
    notre = exact_diagrams(matrice)  # Calcule nos diagrammes.
    for dimension in (0, 1):  # Parcourt H0 et H1.
        reference = arbre.persistence_intervals_in_dimension(dimension)  # Lit les intervalles de référence.
        reference = reference[np.isfinite(reference[:, 1])] if reference.size else np.empty((0, 2))  # Garde les points finis.
        a = notre[dimension][np.isfinite(notre[dimension][:, 1])]  # Garde nos points finis.
        assert a.shape[0] == reference.shape[0]  # Compare les nombres de points.
        assert gudhi.bottleneck_distance(a, reference) < 1e-6  # Compare les diagrammes, les intervalles de GUDHI étant arrondis en simple précision.


def test_biais_topologique_borne_et_differentiable():  # Vérifie le biais différentiable.
    """Le biais doit appartenir à [0, 1], valoir 1 sur la diagonale et transmettre un gradient."""
    torch.manual_seed(7)  # Fixe la graine.
    filtration = LipschitzFiltration(32, 16, 4)  # Crée la filtration.
    points = torch.nn.functional.normalize(torch.randn(2, 8, 32), dim=-1).requires_grad_(True)  # Crée des points différentiables.
    biais = topological_bias(filtration(points))  # Calcule le biais.
    assert (biais >= -1e-6).all() and (biais <= 1.0 + 1e-6).all()  # Vérifie les bornes.
    assert torch.allclose(torch.diagonal(biais, dim1=1, dim2=2), torch.ones(2, 8))  # Vérifie la diagonale.
    biais.sum().backward()  # Rétropropage le gradient.
    assert points.grad.abs().sum() > 0.0  # Vérifie le gradient sur les points.


def test_encodeur_topologique_transmet_le_gradient_des_diagrammes():  # Corrige l'ancien faux test H12.
    """Les tokens doivent dépendre des diagrammes eux-mêmes, et non d'une moyenne de tokens."""
    torch.manual_seed(8)  # Fixe la graine.
    filtration = LipschitzFiltration(32, 16, 4)  # Crée la filtration.
    encodeur = TopologicalEncoder(32, 2, 2, 3, 16, 4, 0.05)  # Crée l'encodeur.
    points = torch.nn.functional.normalize(torch.randn(2, 12, 32), dim=-1).requires_grad_(True)  # Crée des points différentiables.
    h0, h1, descripteurs = encodeur(differentiable_persistence(filtration(points)), filtration.echelles)  # Encode les diagrammes.
    assert h0.shape == (2, 2, 32) and h1.shape == (2, 2, 32)  # Vérifie les formes.
    assert descripteurs["paysage_h1"].requires_grad  # Vérifie que le paysage reste dans le graphe de calcul.
    descripteurs["paysage_h1"].sum().backward(retain_graph=True)  # Rétropropage depuis le seul paysage de H1.
    assert points.grad.abs().sum() > 0.0  # Vérifie que le gradient traverse la persistance.
    filtration.zero_grad()  # Réinitialise les gradients de la filtration.
    (h0.sum() + h1.sum()).backward()  # Rétropropage depuis les tokens.
    assert filtration.echelles_brutes.grad is not None and filtration.echelles_brutes.grad.abs().sum() > 0.0  # Vérifie que les échelles sont utilisées.


def test_amplification_du_gradient_bornee_par_l_encodeur():  # Empêche le retour de l'explosion du gradient.
    """Des barres minuscules ne doivent pas amplifier le gradient : l'encodeur doit rester proche de la lipschitzianité."""
    torch.manual_seed(10)  # Fixe la graine.
    encodeur = TopologicalEncoder(16, 2, 2, 3, 16, 4, 0.05)  # Crée l'encodeur.
    filtration = LipschitzFiltration(16, 8, 4)  # Crée la filtration.
    centre = torch.nn.functional.normalize(torch.randn(1, 1, 16), dim=-1)  # Tire un centre commun.
    points = torch.nn.functional.normalize(centre + 1e-3 * torch.randn(32, 10, 16), dim=-1).requires_grad_(True)  # Crée des nuages très concentrés, donc des barres minuscules.
    h0, h1, _ = encodeur(differentiable_persistence(filtration(points)), filtration.echelles)  # Encode les diagrammes.
    (h0.pow(2).sum() + h1.pow(2).sum()).backward()  # Calcule les gradients.
    normes = [p.grad.norm().item() for p in filtration.parameters() if p.grad is not None] + [points.grad.norm().item()]  # Mesure les normes de gradient.
    assert all(np.isfinite(n) for n in normes) and max(normes) < 2000.0  # Vérifie l'absence d'explosion ; l'ancien encodage dépassait largement ce niveau.


def test_graphe_symetrique_borne_et_masque():  # Vérifie le graphe latent.
    """L'adjacence doit être symétrique, bornée et sans arête vers le padding."""
    graphe = AdaptiveLatentGraph(32, 16)  # Crée le graphe.
    tokens = torch.randn(2, 6, 32)  # Tire des tokens.
    masque = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])  # Crée le masque.
    adjacence, log_variance = graphe(tokens, masque)  # Calcule le graphe.
    assert adjacence.shape == (2, 6, 6) and log_variance.shape == (2, 6, 6)  # Vérifie les formes.
    assert torch.allclose(adjacence, adjacence.transpose(1, 2), atol=1e-6)  # Vérifie la symétrie.
    assert (adjacence >= 0.0).all() and (adjacence <= 1.0).all()  # Vérifie les bornes.
    assert adjacence[1, 4:].abs().sum() == 0.0  # Vérifie l'absence d'arête vers le padding.


def test_aretes_critiques_valides():  # Vérifie la sélection des arêtes critiques.
    """Les arêtes critiques doivent désigner des paires de points existantes et distinctes."""
    aretes = critical_edges(_distances(_cercle(20)))  # Sélectionne les arêtes critiques.
    assert aretes.shape[1] == 2 and aretes.shape[0] >= 19  # Vérifie la forme et le minimum attendu de H0.
    assert (aretes[:, 0] != aretes[:, 1]).all() and aretes.max() < 20  # Vérifie la validité des indices.


def test_paires_de_petits_nuages():  # Vérifie les cas limites.
    """Un nuage d'un ou deux points ne doit ni planter ni inventer de cycle."""
    assert compute_flag_pairs(np.zeros((1, 1))).h0_essential.size == 1  # Vérifie un nuage d'un point.
    deux = compute_flag_pairs(np.array([[0.0, 1.0], [1.0, 0.0]]))  # Calcule un nuage de deux points.
    assert deux.h0_death.shape == (1, 2) and deux.h1_birth.shape == (0, 2)  # Vérifie les résultats attendus.
