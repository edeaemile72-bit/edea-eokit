# EDEA-EOKit

**A tested and reproducible Python chain for surface water and flood mapping from optical satellite imagery.**

EDEA-EOKit (`eokit`) reads a multiband GeoTIFF, applies the correct radiometric scaling for the sensor, masks clouds, computes water indices, converts pixels into hectares using the real ground size of a pixel, and measures how optimistic a validation design can be when training and test samples are drawn at random.

The library is deliberately small: about 1 100 lines of readable Python including the docstrings, six sub-modules, and 41 automated tests that all pass on Python 3.10 to 3.13. It is meant to be opened, read and audited by a student or a colleague, and re-run on another image without editing a single line.

Version 1.0.3, Python 3.10 or later, MIT licence.

---

## Why this package exists

A water index is a one-line formula. The fragile part is the chain of conventions around it. Almost every water mapping script makes four decisions that are never written down, and each of them can move the final area by tens of percent:

| Convention | What usually happens | What `eokit` does |
|---|---|---|
| **Radiometric scaling** | One factor copied from a tutorial is applied to every band, including classification bands | The factor and offset are deduced from the product and the sensor, classification bands are always left as integers, and the applied rule is stored in `scene.echelle` |
| **Ground size of a pixel** | The nominal resolution (10, 20 or 30 m) is hard-coded | `pixel_area_m2()` reads the transform, and handles degree grids by computing the local metre size at the scene latitude |
| **Cloud mask** | A cloud probability band is divided by 10,000 and the mask stops working | SCL or QA bands are used when present, with a documented fallback and an explicit threshold |
| **Validation design** | Points are split at random, adjacent pixels end up in training and test | `comparer_strategies()` runs the same classifier under five designs and reports the optimistic bias in OA points |

---

## Installation

```bash
# from a clone of this repository
pip install -e .

# or with the test dependencies
pip install -e ".[dev]"
```

Dependencies: `numpy`, `rasterio`, `geopandas`, `shapely`, `pyproj`, `scikit-image`, `scikit-learn`, `pandas`, `matplotlib`, `openpyxl`. GDAL is provided by the `rasterio` wheels on Windows, Linux and macOS.

Sub-modules are imported lazily: `import eokit` does not load `rasterio` or `geopandas`, and a missing optional dependency is reported only when the sub-module that needs it is used.

---

## Quick start

```python
from eokit.io import read_scene
from eokit.cloud import cloud_mask
from eokit.indices import compute
from eokit.geometry import pixel_area_m2

scene = read_scene("subset_s2.tif", sensor="sentinel2")
print(scene)                      # Scene(sensor='sentinel2', shape=(...), bands=[...])
print(scene.echelle["regle"])     # scaling rule actually applied
print(scene.bandes_masquees)      # bands kept as integer codes (scl, qa, ...)

cloud = cloud_mask(scene)         # True where the pixel is a cloud or a shadow
mndwi = compute("mndwi", scene.bands)
water = (mndwi > 0.1) & ~cloud

area_ha = water.sum() * pixel_area_m2(scene) / 10_000
print(f"{area_ha:.1f} ha of water")
```

If the file carries a classification band after the reflectance bands (a common layout when you
export an SCL layer from a stack), name the bands explicitly:

```python
scene = read_scene("stack.tif", sensor="sentinel2",
                   band_names=["blue", "green", "red", "nir", "swir1", "swir2", "scl"])
```

A complete, runnable walk-through is in [`examples/01_water_mapping.ipynb`](examples/01_water_mapping.ipynb). It builds a small synthetic scene with `examples/make_demo_scene.py`, so the notebook runs without downloading anything.

---

## Modules

| Module | Role | Main entry points |
|---|---|---|
| `eokit.io` | Read a GeoTIFF into a `Scene` (named bands, NaN nodata, explicit scaling) | `read_scene()`, `appliquer_echelle()`, `Scene` |
| `eokit.indices` | Spectral water and vegetation indices | `ndvi()`, `ndwi()`, `mndwi()`, `awei()`, `compute()`, `compute_many()`, `available()` |
| `eokit.cloud` | Cloud and shadow mask from SCL or QA, heuristic fallback | `cloud_mask()` |
| `eokit.geometry` | Areas and extents, including degree grids | `surface_ha()`, `bbox()`, `pixel_area_m2()`, `pixels_en_metres()` |
| `eokit.viz` | Display helpers built on matplotlib | `show_index()`, `stretch()`, `rgb_composite()` |
| `eokit.validation_spatiale` | Supervised classification and spatial validation designs | `echantillonner_features()`, `moran_i()`, `comparer_strategies()`, `exporter_excel()`, `graphique_biais()` |

### `read_scene()`

```python
read_scene(path, sensor="sentinel2", band_names=None, scale="auto",
           offset=None, skip_scale=None)
```

`sensor` accepts `sentinel2`, `landsat8`, `landsat9`, `landsat7`, `landsat5`, `drone` and `generic`. In `scale="auto"` mode the rule is deduced as follows:

- `SR_Bx` descriptions in the file mean Landsat Collection 2 surface reflectance (2.75e-05, offset -0.2);
- Sentinel-2 with values above 1.5 means integers 0-10000, divided by 10000;
- Landsat means 2.75e-05 with offset -0.2;
- values already inside [0, 1] are not divided again.

A factor can also be forced per band, or scaling can be disabled:

```python
scene = read_scene("scene.tif", scale={"blue": 1e-4, "green": 1e-4})
scene = read_scene("scene.tif", scale=None)
```

### `pixel_area_m2()`

Returns the area of one pixel in square metres in both cases:

- a projected CRS, where the pixel step is already in metres;
- a geographic CRS (EPSG:4326), where the step is in degrees: the metre size is then computed at the mean latitude of the scene, and the value differs from the square of a nominal resolution.

```python
from eokit.geometry import pixel_area_m2, pixels_en_metres
print(pixel_area_m2(scene), pixels_en_metres(scene))
```

### Water indices

```python
from eokit.indices import compute, compute_many, available

print(available())                              # ('ndvi', 'ndwi', 'mndwi', 'awei')
idx = compute_many(("ndwi", "mndwi", "awei"), scene.bands)
```

| Index | Formula | Bands used |
|---|---|---|
| NDWI | (green - nir) / (green + nir) | green, nir |
| MNDWI | (green - swir1) / (green + swir1) | green, swir1 |
| AWEI | 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2) | green, nir, swir1, swir2 |
| NDVI | (nir - red) / (nir + red) | nir, red |

A missing band raises an explicit error that names the band and the index.

### Spatial validation

```python
from eokit.validation_spatiale import comparer_strategies

table = comparer_strategies(scene, points, "classe",
                            noms_indices=("ndvi", "ndwi", "mndwi"),
                            k_folds=5, buffer_m=100.0, taille_bloc=500.0)
print(table[["OA", "Kappa", "Moran_I_residus", "Biais_OA"]])
```

The same Random Forest is trained in every design; only the train and test split changes. The reference design is a spatial block cross-validation. `Biais_OA` is the difference in overall accuracy, in points, relative to that reference, so a positive value quantifies how optimistic the design is. The table also reports the median nearest-neighbour distance between training and test points, and Moran's I of the residuals with its permutation p-value.

---

## Tests

```bash
pytest
```

41 tests, spread over six files:

| File | Tests | Covered behaviour |
|---|---|---|
| `test_indices.py` | 12 | formulas, band lookup, missing bands, NaN propagation |
| `test_moran.py` | 7 | Moran's I, permutation p-value, weight construction |
| `test_paquet.py` | 7 | lazy import, version, public API |
| `test_cloud.py` | 6 | SCL and QA masks, fallback threshold |
| `test_geometry.py` | 6 | pixel area in metres and in degrees, extent |
| `test_scene.py` | 3 | scaling rules, masked bands, nodata |

The three tests that open a GeoTIFF use `pytest.importorskip` and are skipped automatically when `rasterio` or `geopandas` is not installed.

---

## Data

The repository does not ship satellite imagery. `examples/make_demo_scene.py` writes a small synthetic Sentinel-2 like scene (six reflectance bands and an SCL band, 10 m pixels, UTM zone 31N) with a river and a lake, which is enough to run the notebook and the tests of the demonstration section.

To work on real data, download a Level-2A tile from the Copernicus Open Access Hub or a Landsat Collection 2 surface reflectance scene from USGS EarthExplorer, and clip a subset around your area of interest.

---

## Citation

If you use this package in a publication, please cite the software paper and the archived release:

```bibtex
@article{edea2026eokit,
  title   = {EDEA-EOKit: a tested and reproducible Python chain for surface water and flood mapping from optical satellite imagery},
  author  = {EDEA, Obognon Emile and PLAGBETO, Hermann Arnaud and OLOUKOI, Joseph},
  journal = {SoftwareX},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

A `CITATION.cff` file is provided so that GitHub displays the citation block. The DOI of the archived release will be added here once minted on Zenodo.

---

## Contributing

Bug reports and pull requests are welcome. Please open an issue with a minimal reproducible example, the sensor, the CRS of the scene, and the versions reported by `python -c "import eokit; print(eokit.__version__)"`.

## Licence

MIT. See [LICENSE.txt](LICENSE.txt).

## Contact

Dr. Ir. EDEA O. Emile
Enseignant-chercheur, Expert en géomatique et analyse spatiale
Laboratoire de Biogéographie et Expertise Environnementale (LABEE), Université d'Abomey-Calavi, Benin
edeaemile72@gmail.com
