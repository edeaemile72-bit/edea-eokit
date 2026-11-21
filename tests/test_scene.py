"""Objet Scene et lecture d'un GeoTIFF (ignore si rasterio absent)."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rasterio")

from eokit.io import Scene  # noqa: E402


def test_scene_attributs():
    bandes = {
        "red": np.zeros((3, 4), dtype="float32"),
        "nir": np.zeros((3, 4), dtype="float32"),
        "scl": np.zeros((3, 4), dtype="float32"),
    }
    transform = SimpleNamespace(a=10.0, c=300000.0, e=-10.0, f=400000.0)
    crs = SimpleNamespace(is_geographic=False)
    scene = Scene(bandes, {"crs": crs, "transform": transform}, "sentinel2",
                  descriptions=["B04", "B08", "SCL"],
                  echelle={"facteur": 0.0001, "bandes_masquees": ["scl"]})
    assert scene.shape == (3, 4)
    assert scene.crs is crs
    assert scene.bandes_masquees == ["scl"]
    assert "shape=(3, 4)" in repr(scene)


def test_scene_sans_echelle():
    bandes = {"red": np.zeros((2, 2))}
    scene = Scene(bandes, {}, "generic")
    assert scene.echelle == {}
    assert scene.bandes_masquees == []


def test_read_scene_sur_un_geotiff():
    import rasterio
    from rasterio.transform import from_origin

    import tempfile
    import os

    chemin = os.path.join(tempfile.mkdtemp(), "scene.tif")
    with rasterio.open(
        chemin, "w", driver="GTiff", height=4, width=5, count=2,
        dtype="float32", crs="EPSG:32631",
        transform=from_origin(300000, 400000, 10, 10),
    ) as dst:
        dst.write(np.full((4, 5), 1500, dtype="float32"), 1)
        dst.write(np.full((4, 5), 3000, dtype="float32"), 2)

    from eokit.io import read_scene
    scene = read_scene(chemin, sensor="generic",
                       band_names=["red", "nir"], scale=None)
    assert scene.shape == (4, 5)
    assert set(scene.bands) == {"red", "nir"}
    np.testing.assert_allclose(scene.bands["red"].mean(), 1500.0)
