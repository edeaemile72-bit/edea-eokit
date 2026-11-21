"""Formules des indices spectraux, sur des valeurs calculees a la main."""

import numpy as np
import pytest

from eokit import indices


@pytest.fixture
def bandes():
    return {
        "red": np.array([[0.10, 0.20], [0.30, 0.40]], dtype="float32"),
        "green": np.array([[0.30, 0.30], [0.30, 0.30]], dtype="float32"),
        "nir": np.array([[0.50, 0.10], [0.50, 0.10]], dtype="float32"),
        "swir1": np.array([[0.10, 0.10], [0.50, 0.50]], dtype="float32"),
        "swir2": np.array([[0.05, 0.05], [0.05, 0.05]], dtype="float32"),
    }


def test_ndvi(bandes):
    out = indices.ndvi(bandes)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0, 0], 0.4 / 0.6, rtol=1e-6)
    np.testing.assert_allclose(out[1, 1], (0.10 - 0.40) / 0.50, rtol=1e-6)


def test_ndwi_eau_positive(bandes):
    out = indices.ndwi(bandes)
    np.testing.assert_allclose(out[0, 0], (0.30 - 0.50) / 0.80, rtol=1e-6)


def test_mndwi(bandes):
    out = indices.mndwi(bandes)
    np.testing.assert_allclose(out[0, 0], (0.30 - 0.10) / 0.40, rtol=1e-6)
    np.testing.assert_allclose(out[1, 1], (0.30 - 0.50) / 0.80, rtol=1e-6)


def test_mndwi_sans_swir1_replie_sur_nir(bandes, capsys):
    sans_swir = {k: v for k, v in bandes.items() if k != "swir1"}
    out = indices.mndwi(sans_swir)
    attendu = indices.ndwi(sans_swir)
    np.testing.assert_allclose(out, attendu)
    assert "SWIR1 absente" in capsys.readouterr().out


def test_mndwi_sans_green_ni_nir(bandes):
    with pytest.raises(KeyError):
        indices.mndwi({"red": bandes["red"]})


def test_awei(bandes):
    out = indices.awei(bandes)
    attendu = 4 * (0.30 - 0.10) - (0.25 * 0.50 + 2.75 * 0.05)
    np.testing.assert_allclose(out[0, 0], attendu, rtol=1e-6)


def test_denominateur_nul_donne_nan(bandes):
    zeros = {k: np.zeros((2, 2), dtype="float32") for k in bandes}
    out = indices.ndvi(zeros)
    assert np.isnan(out).all()


def test_bande_absente_leve_keyerror(bandes):
    with pytest.raises(KeyError):
        indices.ndvi({"nir": bandes["nir"]})


def test_compute_insensible_a_la_casse(bandes):
    np.testing.assert_allclose(indices.compute("NDVI", bandes),
                               indices.ndvi(bandes))


def test_compute_indice_inconnu(bandes):
    with pytest.raises(ValueError) as e:
        indices.compute("evi", bandes)
    assert "Disponibles" in str(e.value)


def test_compute_many_ignore_bande_manquante(bandes, capsys):
    sans_swir = {k: v for k, v in bandes.items()
                 if k not in ("swir1", "swir2")}
    out = indices.compute_many(["ndvi", "awei"], sans_swir)
    assert list(out) == ["ndvi"]
    assert "AWEI ignoré" in capsys.readouterr().out


def test_available(bandes):
    assert indices.available() == ["awei", "mndwi", "ndvi", "ndwi"]
