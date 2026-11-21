"""Indices spectraux pour la détection de l'eau et de la végétation.

Toutes les fonctions prennent le dictionnaire ``bands`` (nom -> tableau 2D)
et renvoient un tableau 2D. Les indices normalisés sont dans [-1, 1].

Bandes utilisées
----------------
- MNDWI = (green - swir1) / (green + swir1)
- NDWI  = (green - nir)   / (green + nir)
- AWEI (non-shadow) = 4*(green - swir1) - (0.25*nir + 2.75*swir2)
- NDVI  = (nir - red)     / (nir + red)
"""

import numpy as np


def _get(bands, name):
    if name not in bands:
        raise KeyError(name)
    return bands[name]


def _norm_diff(a, b):
    """Différence normalisée (a-b)/(a+b), robuste à la division par 0."""
    a = a.astype("float64")
    b = b.astype("float64")
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / denom
    out[denom == 0] = np.nan
    return out


def ndvi(bands):
    return _norm_diff(_get(bands, "nir"), _get(bands, "red"))


def ndwi(bands):
    """NDWI de McFeeters : (green - nir) / (green + nir)."""
    return _norm_diff(_get(bands, "green"), _get(bands, "nir"))


def mndwi(bands):
    """MNDWI de Xu : (green - swir1) / (green + swir1).

    Repli : si la bande SWIR1 est absente (ex. produit à 4 bandes 10 m),
    on calcule un indice d'eau équivalent avec le NIR (revient au NDWI)
    et on avertit l'utilisateur — cela évite d'interrompre la chaîne.
    """
    if "swir1" in bands:
        return _norm_diff(bands["green"], bands["swir1"])
    if "nir" in bands:
        print("[eokit] Attention : bande SWIR1 absente — MNDWI calculé "
              "avec le NIR (équivalent NDWI). Fournissez une image avec "
              "SWIR (>= 6 bandes) pour un vrai MNDWI.")
        return _norm_diff(bands["green"], bands["nir"])
    raise KeyError("green/swir1/nir")


def awei(bands):
    """AWEI (Automated Water Extraction Index), variante sans ombre."""
    green = _get(bands, "green")
    nir = _get(bands, "nir")
    swir1 = _get(bands, "swir1")
    swir2 = _get(bands, "swir2")
    return 4.0 * (green - swir1) - (0.25 * nir + 2.75 * swir2)


_REGISTRY = {
    "ndvi": ndvi,
    "ndwi": ndwi,
    "mndwi": mndwi,
    "awei": awei,
}


def available():
    """Liste des noms d'indices disponibles."""
    return sorted(_REGISTRY.keys())


def compute(name, bands):
    """Calcule un seul indice par son nom."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Indice inconnu : {name!r}. "
                         f"Disponibles : {available()}")
    return _REGISTRY[key](bands)


def compute_many(names, bands):
    """Calcule plusieurs indices et renvoie un dict {nom: tableau}.

    Un indice dont une bande requise est absente est ignoré (avec un
    avertissement) plutôt que de faire échouer toute la chaîne.
    """
    out = {}
    for name in names:
        key = name.lower()
        try:
            out[key] = compute(key, bands)
        except KeyError as e:
            print(f"[eokit] Indice {key.upper()} ignoré : bande manquante "
                  f"{e}. Bandes disponibles : {list(bands.keys())}")
    return out


__all__ = ["compute", "compute_many", "available",
           "ndvi", "ndwi", "mndwi", "awei"]
