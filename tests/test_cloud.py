"""Masque de nuages : SCL, QA et heuristique de brillance."""

from types import SimpleNamespace

import numpy as np

from eokit import cloud


def scene(**bandes):
    return SimpleNamespace(bands=bandes)


def test_methode_aucun():
    s = scene(blue=np.ones((3, 3), dtype="float32"))
    masque = cloud.cloud_mask(s, methode="aucun")
    assert masque.shape == (3, 3)
    assert not masque.any()


def test_auto_sans_bande_classification(capsys):
    s = scene(blue=np.ones((2, 2), dtype="float32"))
    masque = cloud.cloud_mask(s)
    assert not masque.any()
    assert "masque de nuages non appliqué" in capsys.readouterr().out


def test_scl_classes_nuages():
    scl = np.array([[3, 4, 6, 8], [9, 10, 11, 4]], dtype="float32")
    masque = cloud._mask_from_scl(scl)
    attendu = np.array([[True, False, False, True],
                        [True, True, False, False]])
    np.testing.assert_array_equal(masque, attendu)


def test_qa_bits_nuage_et_ombre():
    qa = np.array([[0, 8, 16, 24]], dtype="float32")
    masque = cloud._mask_from_qa(qa)
    np.testing.assert_array_equal(masque, [[False, True, True, True]])


def test_brillance():
    bleu = np.array([[0.9, 0.1, np.nan]], dtype="float32")
    vert = np.array([[0.9, 0.1, 0.9]], dtype="float32")
    rouge = np.array([[0.9, 0.1, 0.9]], dtype="float32")
    masque = cloud.cloud_mask(scene(blue=bleu, green=vert, red=rouge),
                              methode="brillance", seuil_brillance=0.5)
    np.testing.assert_array_equal(masque, [[True, False, False]])


def test_brillance_sans_bandes_visibles():
    import pytest
    with pytest.raises(ValueError):
        cloud.cloud_mask(scene(nir=np.ones((2, 2))), methode="brillance")
