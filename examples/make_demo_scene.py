"""Create the synthetic Sentinel-2 like scene used by the demonstration notebook.

Run from the repository root::

    python examples/make_demo_scene.py

Output
------
- ``examples/_demo/demo_s2.tif``    : 6 reflectance bands + 1 SCL band, EPSG:32631
- ``examples/_demo/demo_points.gpkg`` : 500 labelled points (water / other)

The scene contains a river, a lake, a cloud above the lake, its shadow, and a
small saturated patch, so that the cloud mask and the area computation can be
tested on realistic situations. Nothing here is a real observation: the values
are chosen to reproduce the spectral contrast of a Level-2A product.
"""

from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_demo")
os.makedirs(OUT, exist_ok=True)

TAILLE = 400          # pixels
PIXEL = 10.0          # metres
X0, Y0 = 450_000.0, 750_000.0
CRS = "EPSG:32631"

rng = np.random.default_rng(42)

yy, xx = np.mgrid[0:TAILLE, 0:TAILLE].astype(float)

# ---------------------------------------------------------------- geometry
# river: a sine shaped band of about 4 pixels wide
centre = 200 + 45 * np.sin(xx / 55.0)
riviere = np.abs(yy - centre) < 4

# lake: an ellipse in the north east
lac = (((xx - 300) / 45.0) ** 2 + ((yy - 95) / 32.0) ** 2) < 1.0

eau = riviere | lac

# cloud over the north west edge of the lake, and its shadow to the south west
nuage = (((xx - 245) / 55.0) ** 2 + ((yy - 52) / 30.0) ** 2) < 1.0
ombre = (((xx - 205) / 55.0) ** 2 + ((yy - 88) / 30.0) ** 2) < 1.0
ombre &= ~nuage

# saturated patch in the south
sature = (((xx - 90) / 18.0) ** 2 + ((yy - 330) / 18.0) ** 2) < 1.0

# ---------------------------------------------------------------- spectra
# surface reflectance, as floats, converted to integers 0-10000 at the end
sol = {"blue": 0.055, "green": 0.085, "red": 0.105, "nir": 0.300,
       "swir1": 0.250, "swir2": 0.150}
veg = {"blue": 0.030, "green": 0.070, "red": 0.040, "nir": 0.380,
       "swir1": 0.180, "swir2": 0.070}
wat = {"blue": 0.060, "green": 0.055, "red": 0.030, "nir": 0.015,
       "swir1": 0.008, "swir2": 0.004}
nua = {"blue": 0.560, "green": 0.580, "red": 0.600, "nir": 0.590,
       "swir1": 0.520, "swir2": 0.480}
sat = {"blue": 0.900, "green": 0.920, "red": 0.950, "nir": 0.880,
       "swir1": 0.850, "swir2": 0.800}

# vegetation on the northern half, bare soil elsewhere
vegetation = (yy < 180) & ~eau

bands = {}
for nom, val_sol in sol.items():
    arr = np.full((TAILLE, TAILLE), val_sol, dtype=float)
    arr = np.where(vegetation, veg[nom], arr)
    arr = np.where(eau, wat[nom], arr)
    arr = np.where(nuage, nua[nom], arr)
    arr = np.where(sature, sat[nom], arr)
    # shadow: water becomes darker, land loses about 40 percent of its signal
    arr = np.where(ombre & eau, arr * 0.55, arr)
    arr = np.where(ombre & ~eau, arr * 0.60, arr)
    # multiplicative noise, 2 percent
    arr = arr * rng.normal(1.0, 0.02, size=arr.shape)
    bands[nom] = np.clip(arr * 10_000.0, 0, 10_000).astype(np.int16)

# ---------------------------------------------------------------- SCL band
# 4 vegetation, 5 bare soil, 6 water, 3 cloud shadow, 9 cloud, 1 saturated
scl = np.full((TAILLE, TAILLE), 5, dtype=np.uint8)
scl[vegetation] = 4
scl[eau] = 6
scl[ombre] = 3
scl[nuage] = 9
scl[sature] = 1

noms = ["blue", "green", "red", "nir", "swir1", "swir2", "scl"]
desc = ["B02 blue", "B03 green", "B04 red", "B08 nir",
        "B11 swir1", "B12 swir2", "scl"]

profil = dict(driver="GTiff", height=TAILLE, width=TAILLE, count=len(noms),
              dtype="int16", crs=CRS, transform=from_origin(X0, Y0 + TAILLE * PIXEL,
                                                            PIXEL, PIXEL),
              compress="deflate", nodata=None)

chemin_tif = os.path.join(OUT, "demo_s2.tif")
with rasterio.open(chemin_tif, "w", **profil) as dst:
    for i, nom in enumerate(noms, start=1):
        dst.write(bands[nom] if nom in bands else scl, i)
        dst.set_band_description(i, desc[i - 1])
    dst.update_tags(sensor="sentinel2", product="synthetic Level-2A subset")

# ---------------------------------------------------------------- points
verite = eau.copy()
pts, cls = [], []
for classe, n in ((1, 220), (0, 280)):
    candidats = np.argwhere(verite == bool(classe))
    choisis = candidats[rng.choice(len(candidats), size=n, replace=False)]
    for r, c in choisis:
        x = X0 + (c + 0.5) * PIXEL
        y = Y0 + TAILLE * PIXEL - (r + 0.5) * PIXEL
        pts.append(Point(x, y))
        cls.append("eau" if classe else "autre")

gdf = gpd.GeoDataFrame({"id": range(1, len(pts) + 1), "classe": cls},
                       geometry=pts, crs=CRS)
chemin_pts = os.path.join(OUT, "demo_points.gpkg")
gdf.to_file(chemin_pts, layer="points", driver="GPKG")

print(f"scene  : {chemin_tif}  ({TAILLE} x {TAILLE} px, {PIXEL:g} m)")
print(f"points : {chemin_pts}  ({len(gdf)} points, {int((gdf['classe'] == 'eau').sum())} water)")
print(f"water  : {eau.sum()} pixels, soit {eau.sum() * PIXEL * PIXEL / 10_000:.1f} ha de verite terrain")
