"""EDEA-EOKit — boîte à outils de télédétection pour la cartographie
des zones en eau et des inondations.

Auteur : Dr. Ir. EDEA O. Emile, Enseignant-chercheur
         Géomatique, Télédétection et Analyse Spatiale

Sous-modules
------------
- io        : lecture d'une scène satellite -> objet Scene (bandes nommées).
- indices   : calcul d'indices spectraux d'eau (MNDWI, NDWI, AWEI, NDVI...).
- cloud     : masque de nuages (SCL/QA si présent, sinon heuristique).
- geometry  : utilitaires géométriques (complément à shapely/geopandas).
- viz       : utilitaires de visualisation (complément à matplotlib).
- validation_spatiale : classification supervisée + stratégies de validation
              spatiale et biais optimiste (Moran I, distance train/test).

Depuis la v1.0.3, les sous-modules sont chargés à la demande : importer
eokit ne demande plus rasterio ni geopandas. Une dépendance manquante
n'est signalée qu'au moment où le sous-module concerné est utilisé.
"""

import importlib

_SOUS_MODULES = ("io", "indices", "cloud", "geometry", "viz",
                 "validation_spatiale")

__version__ = "1.0.3"
__author__ = (
    "Dr. Ir. EDEA O. Emile, Enseignant-chercheur — "
    "Géomatique, Télédétection et Analyse Spatiale"
)
__all__ = list(_SOUS_MODULES) + ["__version__", "__author__"]


def __getattr__(nom):
    """Charge un sous-module seulement quand on y accède (PEP 562)."""
    if nom in _SOUS_MODULES:
        module = importlib.import_module("." + nom, __name__)
        globals()[nom] = module
        return module
    raise AttributeError(f"module {__name__!r} n'a pas d'attribut {nom!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SOUS_MODULES))
