"""Surface du pixel, emprise et surfaces en hectares."""

from types import SimpleNamespace

import numpy as np
import pytest

from eokit import geometry


def scene_projetee():
    transform = SimpleNamespace(a=10.0, b=0.0, c=300000.0,
                                d=0.0, e=-10.0, f=400000.0)
    crs = SimpleNamespace(is_geographic=False)
    return SimpleNamespace(transform=transform, crs=crs, shape=(4, 5))


def scene_geographique():
    transform = SimpleNamespace(a=0.0001, b=0.0, c=2.4,
                                d=0.0, e=-0.0001, f=6.4)
    crs = SimpleNamespace(is_geographic=True)
    return SimpleNamespace(transform=transform, crs=crs, shape=(10, 10))


def test_pixel_area_projetee():
    assert geometry.pixel_area_m2(scene_projetee()) == pytest.approx(100.0)


def test_pixels_en_metres_projetee():
    assert geometry.pixels_en_metres(scene_projetee()) == pytest.approx((10.0, 10.0))


def test_pixel_area_geographique_en_metres_carres():
    # pixel de 0.0001 degre vers 6.4 N : environ 12.3 m x 11.1 m
    aire = geometry.pixel_area_m2(scene_geographique())
    assert 100.0 < aire < 170.0


def test_pixels_en_metres_geographique():
    largeur, hauteur = geometry.pixels_en_metres(scene_geographique())
    assert 10.5 < largeur < 13.0
    assert 10.5 < hauteur < 11.5


def test_bbox():
    xmin, ymin, xmax, ymax = geometry.bbox(scene_projetee())
    assert (xmin, ymin, xmax, ymax) == (300000.0, 399960.0, 300050.0, 400000.0)


def test_surface_ha():
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    gdf = geopandas.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(0, 0, 100, 100), box(0, 0, 200, 100)],
        crs="EPSG:32631",
    )
    surfaces = geometry.surface_ha(gdf)
    np.testing.assert_allclose(surfaces.values, [1.0, 2.0], rtol=1e-6)
