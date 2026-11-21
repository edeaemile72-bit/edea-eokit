"""Utilitaires géométriques (complément léger à shapely / geopandas).

Dans le notebook, la plupart des opérations géométriques passent
directement par geopandas (gdf.geometry.area, representative_point(), ...).
Ce module fournit quelques raccourcis optionnels.

v1.0.2 : ``pixel_area_m2`` renvoie désormais une surface en mètres carrés
même lorsque la scène est en CRS géographique (degrés). Auparavant elle
renvoyait des degrés carrés, soit un résultat inutilisable pour un calcul
de surface en hectares.
"""

import math

import numpy as np


def surface_ha(gdf):
    """Surface (hectares) par entité d'un GeoDataFrame projeté (CRS mètres)."""
    return gdf.geometry.area / 10000.0


def bbox(scene):
    """Emprise (xmin, ymin, xmax, ymax) d'une scène à partir de son transform."""
    t = scene.transform
    h, w = scene.shape
    xs = [t.c, t.c + t.a * w]
    ys = [t.f, t.f + t.e * h]
    return (min(xs), min(ys), max(xs), max(ys))


def _metres_par_degre(latitude):
    """Longueur (m) d'un degré de latitude et de longitude à cette latitude.

    Formule ellipsoïdale approchée (WGS84), valable pour un pixel.
    """
    phi = math.radians(latitude)
    m_lat = (111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
             - 0.0023 * math.cos(6 * phi))
    m_lon = (111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
             + 0.118 * math.cos(5 * phi))
    return m_lat, m_lon


def _surface_pixel_geographique(scene):
    """Surface (m²) d'un pixel pour une scène en degrés."""
    t = scene.transform
    h, w = scene.shape
    # Centre du pixel central : c'est là que la distorsion est représentative.
    x0 = t.c + (w / 2.0) * t.a
    y0 = t.f + (h / 2.0) * t.e

    # pyproj, s'il est disponible, calcule la surface exacte sur l'ellipsoïde.
    try:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        x1, y1 = x0 + t.a, y0 + t.e
        aire, _ = geod.polygon_area_perimeter(
            [x0, x1, x1, x0], [y0, y0, y1, y1])
        return abs(float(aire))
    except Exception:
        pass

    m_lat, m_lon = _metres_par_degre(y0)
    return abs(t.a) * m_lon * abs(t.e) * m_lat


def pixel_area_m2(scene):
    """Surface d'un pixel (m²) d'après la résolution du transform.

    Gère les deux cas :

    - CRS projeté (mètres) : produit direct des pas du transform ;
    - CRS géographique (degrés) : conversion en mètres à la latitude du
      centre de la scène, ce qui donne la surface réelle du pixel.
    """
    t = scene.transform
    crs = getattr(scene, "crs", None)
    est_geographique = bool(getattr(crs, "is_geographic", False))
    if est_geographique:
        return _surface_pixel_geographique(scene)
    return abs(t.a) * abs(t.e)


def pixels_en_metres(scene):
    """Taille du pixel (largeur, hauteur) en mètres, quel que soit le CRS."""
    t = scene.transform
    crs = getattr(scene, "crs", None)
    if bool(getattr(crs, "is_geographic", False)):
        h, w = scene.shape
        latitude = t.f + (h / 2.0) * t.e
        m_lat, m_lon = _metres_par_degre(latitude)
        return abs(t.a) * m_lon, abs(t.e) * m_lat
    return abs(t.a), abs(t.e)


__all__ = ["surface_ha", "bbox", "pixel_area_m2", "pixels_en_metres"]
