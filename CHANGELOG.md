# Changelog

All notable changes to EDEA-EOKit are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.3] - 2026-11-21

### Added
- Automated test suite: 41 tests over six files (indices, scène, cloud, geometry, Moran, package).
- Lazy loading of the sub-modules (PEP 562): `import eokit` no longer requires `rasterio` or
  `geopandas`, and a missing dependency is reported only when the sub-module is used.
- `CITATION.cff`, MIT licence, continuous integration workflow, synthetic demonstration scene and
  executable notebook.

### Changed
- Package metadata: Python 3.10 or later, explicit dependency floors, project URLs, classifiers.
- Documentation rewritten around the four conventions that the library makes explicit.

## [1.0.2] - 2026-11-19

### Fixed
- The radiometric scaling is never applied to classification or quality bands (SCL, QA, masks);
  their codes stay integers, so the cloud mask keeps working.
- The scaling factor and offset are deduced from the product and the sensor instead of being
  hard-coded, and the rule actually applied is stored in `scene.echelle`.

### Added
- `pixel_area_m2()` handles geographic CRS by computing the local metre size at the scene latitude.
- `comparer_strategies()` reports the optimistic bias of four validation designs against a spatial
  block cross-validation, with Moran's I of the residuals.

## [1.0.1] - 2026-11-15

### Fixed
- AWEI and MNDWI raise an explicit error naming the missing band instead of returning NaN silently.
- Nodata values are converted to NaN before any index computation.

## [1.0.0] - 2026-11-10

### Added
- First public version: `io`, `indices`, `cloud`, `geometry`, `viz` and `validation_spatiale`
  sub-modules, and the water and flood mapping notebook.
