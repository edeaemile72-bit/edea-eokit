# Data folder

No satellite imagery is distributed with this repository, to keep it light and to
avoid redistributing third party products.

## Synthetic scene

The demonstration notebook runs on a synthetic scene generated locally:

```bash
python examples/make_demo_scene.py
```

This writes `examples/_demo/demo_s2.tif`: 400 x 400 pixels, 10 m, EPSG:32631,
six reflectance bands (blue, green, red, nir, swir1, swir2) stored as integers
0-10000, one SCL classification band, a river and a lake. Several deliberate
features are planted in it: a bright cloud above a water body, a shadow next to
it, and a saturated patch. The scene is the one used in Section 3 of the paper.

## Real data

| Source | Product | Access |
|---|---|---|
| Copernicus Data Space Ecosystem | Sentinel-2 Level-2A (surface reflectance, 10/20 m) | https://dataspace.copernicus.eu |
| USGS EarthExplorer | Landsat 8/9 Collection 2 Level-2 (surface reflectance, 30 m) | https://earthexplorer.usgs.gov |

Clip a subset around your area of interest, keep the band descriptions in the
GeoTIFF when you export it (QGIS and GDAL do it by default), and pass the right
`sensor=` value to `read_scene()`.
