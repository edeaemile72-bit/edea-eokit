"""Masque de nuages.

Comportement par défaut (sûr)
----------------------------
1. Si la scène contient une bande de classification, elle est utilisée :
   - 'scl'      : Sentinel-2 Scene Classification (classes nuages/ombres).
   - 'qa_pixel' : Landsat Collection 2 QA (bits nuages/ombres).
2. Sinon, AUCUN masque n'est appliqué : la fonction renvoie un masque
   « tout à False » (0 % de nuages) et affiche une note. On ne devine PAS
   les nuages par la brillance par défaut, car cette heuristique sur-détecte
   et risque d'effacer les surfaces en eau (qui sont sombres).

Heuristique de brillance (optionnelle)
--------------------------------------
Elle n'est utilisée que si vous la demandez explicitement :
    cloud.cloud_mask(scene, methode='brillance', seuil_brillance=0.5)
"""

import numpy as np


def _mask_from_scl(scl):
    # classes nuages / ombres / cirrus : 3 (ombre), 8-9 (nuages), 10 (cirrus)
    return np.isin(np.rint(scl), [3, 8, 9, 10])


def _mask_from_qa(qa):
    qa = np.rint(qa).astype("int64")
    cloud_bit = (qa >> 3) & 1      # nuage
    shadow_bit = (qa >> 4) & 1     # ombre de nuage
    return (cloud_bit | shadow_bit).astype(bool)


def _mask_from_brillance(bands, seuil):
    needed = ["blue", "green", "red"]
    if not all(b in bands for b in needed):
        raise ValueError("Heuristique de brillance impossible : bandes "
                         "visibles (blue/green/red) absentes.")
    blue, green, red = bands["blue"], bands["green"], bands["red"]
    brillant = ((blue > seuil) & (green > seuil) & (red > seuil))
    return np.where(np.isnan(blue), False, brillant)


def cloud_mask(scene, sensor="sentinel2", methode="auto", seuil_brillance=0.5):
    """Renvoie un masque booléen 2D (True = nuage).

    Paramètres
    ----------
    scene : eokit.io.Scene
    sensor : str
    methode : {'auto', 'brillance', 'aucun'}
        - 'auto' (défaut) : SCL/QA si présent ; sinon AUCUN masque (0 %).
        - 'brillance'      : force l'heuristique de brillance (à vos risques).
        - 'aucun'          : ne masque jamais rien.
    seuil_brillance : float
        Seuil de réflectance (0-1) pour la méthode 'brillance'.
    """
    bands = scene.bands
    ref = bands.get("blue", next(iter(bands.values())))
    zeros = np.zeros(ref.shape, dtype=bool)

    if methode == "aucun":
        return zeros

    if methode == "brillance":
        return _mask_from_brillance(bands, seuil_brillance)

    # methode == 'auto'
    if "scl" in bands:
        return _mask_from_scl(bands["scl"])
    if "qa_pixel" in bands:
        return _mask_from_qa(bands["qa_pixel"])

    print("[eokit] Aucune bande de classification (SCL/QA) dans l'image : "
          "masque de nuages non appliqué (0 %). Pour forcer une détection "
          "approximative : cloud_mask(scene, methode='brillance').")
    return zeros


__all__ = ["cloud_mask"]
