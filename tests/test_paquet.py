"""Identite du paquet et chargement des sous-modules."""

import pathlib
import tomllib

import eokit

RACINE = pathlib.Path(__file__).resolve().parent.parent


def test_version():
    assert eokit.__version__ == "1.0.3"


def test_version_coherente_avec_pyproject():
    with open(RACINE / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == eokit.__version__


def test_sous_modules_declares():
    attendus = {"io", "indices", "cloud", "geometry", "viz",
                "validation_spatiale"}
    assert attendus <= set(eokit.__all__)


def test_sous_modules_sans_dependance_geo():
    # ces modules ne dependent que de numpy / scipy
    for nom in ("indices", "cloud", "geometry", "validation_spatiale"):
        assert getattr(eokit, nom).__name__.startswith("eokit.")


def test_module_io_si_rasterio_present():
    import pytest
    pytest.importorskip("rasterio")
    assert getattr(eokit, "io").__name__ == "eokit.io"


def test_import_leger_sans_dependance_geo():
    # import eokit doit reussir meme si rasterio n'est pas installe
    assert eokit.__version__
    assert "indices" in dir(eokit)


def test_attribut_inconnu():
    try:
        eokit.module_inexistant
    except AttributeError as e:
        assert "module_inexistant" in str(e)
    else:
        raise AssertionError("AttributeError attendu")
