"""Autocorrelation spatiale et decoupages de validation."""

import numpy as np
import pytest

from eokit import validation_spatiale as vs


def grille(n=10, pas=50.0):
    xs, ys = np.meshgrid(np.arange(n) * pas, np.arange(n) * pas)
    return np.column_stack([xs.ravel(), ys.ravel()])


def test_moran_positif_sur_gradient():
    coords = grille()
    valeurs = coords[:, 0]  # variable lisse : forte autocorrelation
    res = vs.moran_i(valeurs, coords, k=8, permutations=99, graine=1)
    assert res["I"] > 0.3
    assert res["p_perm"] < 0.10


def test_moran_valeur_constante():
    coords = grille(5)
    res = vs.moran_i(np.ones(len(coords)), coords, permutations=19)
    assert np.isnan(res["I"])


def test_blocs_grille_identifiants_distincts():
    coords = np.array([[0.0, 0.0], [10.0, 10.0], [1000.0, 1000.0]])
    blocs = vs._blocs_grille(coords, taille_bloc=100.0)
    assert blocs[0] == blocs[1]
    assert blocs[2] != blocs[0]


def test_block_cv_couvre_tous_les_points():
    coords = grille(6)
    folds = vs._folds_block_cv(coords, 5, taille_bloc=100.0, graine=42)[0]
    vus = np.concatenate([test for _, test in folds])
    assert sorted(vus.tolist()) == list(range(coords.shape[0]))
    assert len(vus) == len(set(vus.tolist()))


def test_appliquer_buffer_retire_les_voisins():
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [500.0, 0.0]])
    train = np.array([0, 2])
    test = np.array([1])
    garde = vs._appliquer_buffer(train, test, coords, buffer_m=50.0)
    assert garde.tolist() == [2]


def test_appliquer_buffer_nul_ne_change_rien():
    coords = np.array([[0.0, 0.0], [10.0, 0.0]])
    train = np.array([0])
    test = np.array([1])
    garde = vs._appliquer_buffer(train, test, coords, buffer_m=0.0)
    assert garde.tolist() == [0]


def test_distance_min_mediane():
    coords_train = np.array([[0.0, 0.0], [100.0, 0.0]])
    coords_test = np.array([[10.0, 0.0], [140.0, 0.0]])
    d = vs._distance_min_mediane(coords_train, coords_test)
    # distances minimales 10 et 40 -> mediane 25
    assert d == pytest.approx(25.0)
