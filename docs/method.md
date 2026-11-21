# Method notes

Short notes on the four conventions that the library makes explicit. They are the material of
Section 1 of the accompanying paper.

## 1. Radiometric scaling

| Product | Factor | Offset | Stored as |
|---|---|---|---|
| Sentinel-2 Level-2A | 1/10000 | 0 | integers 0-10000 |
| Landsat 8/9 Collection 2 Level-2 | 2.75e-05 | -0.2 | integers, surface reflectance |
| Drone, generic | none unless values exceed 1.5 | 0 | floats in [0, 1] |

Classification bands (SCL, QA_PIXEL, masks) are never scaled. A mask divided by 10000 becomes a
field of values below 0.001, and every threshold comparison against 3, 8 or 9 then fails silently.

## 2. Ground size of a pixel

For a projected CRS, the area is the product of the two transform steps. For a geographic CRS, the
steps are in degrees, and the metre size of one degree of longitude shrinks with latitude:

```
dx = 111 320 * cos(latitude)   metres per degree of longitude
dy = 110 574                   metres per degree of latitude (mean value)
```

`pixel_area_m2()` uses the mean latitude of the scene. On a 0.00009 degree grid at 6.4 degrees of
latitude, this gives 100.1 m2 instead of the square of a nominal 10 m resolution; at 13.5 degrees
(Niamey), 96.0 m2.

## 3. Cloud mask

`cloud_mask()` returns True where the pixel is a cloud, a shadow or a cirrus:

- Sentinel-2 SCL: classes 3 (shadow), 8 and 9 (cloud), 10 (cirrus);
- Landsat QA_PIXEL: bits 3 (cloud) and 4 (cloud shadow);
- otherwise no mask at all, with a printed note, unless `methode="brillance"` is requested.

## 4. Validation design

Five designs can be compared with the same classifier and the same features:

| Design | Train and test separation |
|---|---|
| random k-fold | a few metres, since neighbours are split across folds |
| random k-fold with an exclusion buffer | at least the buffer width |
| spatial hold-out (KMeans on coordinates) | whole clusters held out |
| spatial block cross-validation | blocks of a regular grid held out |

The reported bias is the difference in overall accuracy, in points, against the block design.
Moran's I of the out-of-sample residuals measures the spatial structure that remains in the errors.
