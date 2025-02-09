# Change Log

## Unreleased

### Added

### Changed

### removed

## 0.3

Released on February 9, 2025.

### Added

* Major revamp of `PolyData`, with a switch to an object-oriented API.
* Introduction of `point_neighborhoods`.
* Cosmetic improvements to the documentation.

## 0.2

Released on June 14, 2024.

### Added

* Add taichi as an optional dependency and set up the framework for future optional dependencies (https://github.com/scikit-shapes/scikit-shapes/pull/55)
* Varifold Loss for triangle meshes (https://github.com/scikit-shapes/scikit-shapes/pull/53)
* `stiff_edges` attribute to `PolyData` with `knn_graph` and `k_ring_graph` methods (https://github.com/scikit-shapes/scikit-shapes/pull/53)
* `cotan_weights` for triangle meshes (https://github.com/scikit-shapes/scikit-shapes/pull/53)
* multiscale registration example (https://github.com/scikit-shapes/scikit-shapes/pull/53)


### Changed

* Fix Lp loss formula
* Fix as_isometric_as_possible metric formula (https://github.com/scikit-shapes/scikit-shapes/pull/53)

### removed

## 0.1

Released on April 19, 2024.

### Added

* First version of the package
