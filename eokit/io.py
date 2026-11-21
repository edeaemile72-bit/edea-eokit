"""Lecture des scènes satellite -> objet ``Scene`` avec bandes nommées.

Le format d'entrée attendu est un GeoTIFF multibande. L'ordre des bandes
dans le fichier est associé à des noms logiques (blue, green, red, nir,
swir1, swir2, ...) selon le capteur et le nombre de bandes.

Deux règles importantes, revues en v1.0.2 :

1. La mise à l'échelle des réflectances ne s'applique **jamais** aux bandes de
   classification (SCL, QA, masques), qui doivent rester en valeurs entières.
2. Le facteur et le décalage sont explicites : ils sont déduits du capteur et
   des descriptions de bandes du fichier, et restent modifiables par
   ``scale`` / ``offset``. Le facteur propre à chaque produit est ainsi
   respecté (1/10000 pour Sentinel-2 L2A, 2,75e-05 avec décalage -0,2 pour
   Landsat Collection 2 niveau SR).

Si votre fichier a un ordre de bandes particulier, vous pouvez passer
``band_names=[...]`` à :func:`read_scene` pour forcer la correspondance.
"""

import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Correspondance ordre des bandes -> noms logiques, par capteur.
# La clé est le nombre de bandes présentes dans le GeoTIFF.
# ---------------------------------------------------------------------------
_SENTINEL2 = {
    4:  ["blue", "green", "red", "nir"],
    5:  ["blue", "green", "red", "nir", "swir1"],
    6:  ["blue", "green", "red", "nir", "swir1", "swir2"],
    10: ["blue", "green", "red", "red_edge1", "red_edge2", "red_edge3",
         "nir", "nir_narrow", "swir1", "swir2"],
    12: ["coastal", "blue", "green", "red", "red_edge1", "red_edge2",
         "red_edge3", "nir", "nir_narrow", "water_vapor", "swir1", "swir2"],
    13: ["coastal", "blue", "green", "red", "red_edge1", "red_edge2",
         "red_edge3", "nir", "nir_narrow", "water_vapor", "cirrus",
         "swir1", "swir2"],
}

_LANDSAT = {
    4:  ["blue", "green", "red", "nir"],
    5:  ["blue", "green", "red", "nir", "swir1"],
    6:  ["blue", "green", "red", "nir", "swir1", "swir2"],
    7:  ["coastal", "blue", "green", "red", "nir", "swir1", "swir2"],
}

_SENSOR_MAPS = {
    "sentinel2": _SENTINEL2,
    "landsat8": _LANDSAT,
    "landsat9": _LANDSAT,
    "landsat7": _LANDSAT,
    "landsat5": _LANDSAT,
}

# Noms de base tentés pour les premières bandes quand le nombre de bandes
# ne correspond à aucune configuration connue (drone / generic).
_FALLBACK_BASE = ["blue", "green", "red", "nir", "swir1", "swir2"]

# ---------------------------------------------------------------------------
# Mise à l'échelle : facteurs usuels des produits.
# ---------------------------------------------------------------------------
SENTINEL2_L2A_SCALE = 1.0 / 10000.0      # réflectance entière 0-10000 -> 0-1
LANDSAT_C2_SR_SCALE = 2.75e-05           # Landsat Collection 2 niveau SR
LANDSAT_C2_SR_OFFSET = -0.2              # décalage associé

# Mots-clés qui désignent une bande de classification ou de qualité.
# Ces bandes ne sont jamais mises à l'échelle : leurs valeurs sont des codes.
_MASQUE_MOTS = (
    "scl", "qa", "msk", "mask", "class", "cloud", "aot", "wvp",
    "tci", "saa", "sza", "vza", "angle", "nodata",
)

# Masque d'identification : noms logiques connus pour être des masques.
_MASQUE_NOMS = ("scl", "qa_pixel", "qa_radsat", "cloud", "masque", "mask")


def _est_bande_masque(nom, description=""):
    """Vrai si la bande est une bande de classification ou de qualité."""
    nom_clair = (nom or "").strip().lower()
    if nom_clair in _MASQUE_NOMS:
        return True
    texte = f"{nom_clair} {str(description or '').lower()}"
    return any(mot in texte for mot in _MASQUE_MOTS)


def _default_names(sensor, count):
    """Retourne la liste de noms de bandes pour un capteur et un nombre."""
    sensor = (sensor or "generic").lower()
    mapping = _SENSOR_MAPS.get(sensor)
    if mapping and count in mapping:
        return list(mapping[count])
    # Repli : nomme les premières bandes avec les noms usuels, puis band_i.
    names = list(_FALLBACK_BASE[:count])
    for i in range(len(names), count):
        names.append(f"band_{i + 1}")
    return names


def _maximum_global(arr):
    """Maximum des valeurs finies, ou None si le tableau est vide."""
    fini = arr[np.isfinite(arr)]
    return float(fini.max()) if fini.size else None


def _echelle_par_defaut(capteur, descriptions, val_max):
    """Facteur et décalage déduits du capteur et des descriptions de bandes.

    Retourne ``(facteur, decalage, justification)``.
    """
    desc = " ".join(str(d).upper() for d in descriptions if d)
    capteur = (capteur or "generic").lower()

    # 1) Les descriptions du fichier sont les plus fiables : un produit Landsat
    #    Collection 2 niveau SR porte des bandes SR_B2 ... SR_B7.
    if "SR_B" in desc or "SR_" in desc:
        return (LANDSAT_C2_SR_SCALE, LANDSAT_C2_SR_OFFSET,
                "Landsat Collection 2 SR (descriptions SR_Bx du fichier)")

    # 2) Sinon, le capteur déclaré.
    if capteur.startswith("sentinel2") or capteur.startswith("sentinel-2"):
        if val_max is not None and val_max <= 1.5:
            return 1.0, 0.0, "valeurs déjà en réflectance (max <= 1,5)"
        return (SENTINEL2_L2A_SCALE, 0.0,
                "Sentinel-2 L2A (entiers 0-10000, facteur 1/10000)")
    if capteur.startswith("landsat"):
        return (LANDSAT_C2_SR_SCALE, LANDSAT_C2_SR_OFFSET,
                "Landsat Collection 2 niveau SR")

    # 3) Capteur inconnu : on ne devine que si les valeurs sont visiblement
    #    des entiers échelonnés, et on le signale.
    if val_max is not None and val_max > 1.5:
        return (SENTINEL2_L2A_SCALE, 0.0,
                "capteur générique, facteur 1/10000 appliqué par défaut")
    return 1.0, 0.0, "aucune mise à l'échelle nécessaire"


def appliquer_echelle(arr, noms, descriptions=None, scale="auto", offset=None,
                      skip_scale=None, capteur="sentinel2"):
    """Met à l'échelle les réflectances, bande par bande, sans toucher aux masques.

    Paramètres
    ----------
    arr : np.ndarray
        Tableau (bandes, hauteur, largeur) déjà converti en float et où les
        valeurs nodata ont été remplacées par NaN.
    noms : list[str]
        Nom logique de chaque bande.
    descriptions : list[str] | None
        Descriptions lues dans le GeoTIFF (``src.descriptions``).
    scale : 'auto' | float | dict
        'auto' (défaut) déduit le facteur du capteur et des descriptions.
        Un nombre impose le même facteur à toutes les bandes non masquées.
        Un dictionnaire permet de régler bande par bande (nom -> facteur).
        None désactive toute mise à l'échelle.
    offset : None | float | dict
        Décalage additif, appliqué après le facteur. None vaut 0, sauf en mode
        'auto' où le décalage du produit détecté est utilisé.
    skip_scale : list[str] | None
        Noms de bandes à ne jamais mettre à l'échelle (en plus des bandes de
        classification détectées automatiquement).
    capteur : str
        Nom du capteur, utilisé en mode 'auto'.

    Retourne
    --------
    (arr_echelle, infos) où ``infos`` est un dictionnaire décrivant la règle
    appliquée : facteur, décalage, justification, bandes masquées, bandes
    laissées telles quelles.
    """
    arr = np.asarray(arr, dtype="float64")
    nb_bandes = arr.shape[0]
    noms = list(noms)[:nb_bandes]
    while len(noms) < nb_bandes:
        noms.append(f"band_{len(noms) + 1}")
    descriptions = list(descriptions or [])
    descriptions += [""] * (nb_bandes - len(descriptions))
    skip = set(skip_scale or ())

    val_max = _maximum_global(arr)
    if scale == "auto":
        facteur, decalage, justification = _echelle_par_defaut(
            capteur, descriptions, val_max)
        if offset is not None and not isinstance(offset, dict):
            decalage = float(offset)
            justification += f" ; décalage imposé ({decalage:g})"
    elif scale is None:
        facteur, decalage, justification = 1.0, 0.0, "mise à l'échelle désactivée"
    else:
        facteur = 1.0 if isinstance(scale, dict) else float(scale)
        decalage = 0.0 if offset is None else (
            float(offset) if not isinstance(offset, dict) else 0.0)
        justification = "facteur fourni par l'utilisateur"

    sortie = arr.copy()
    bandes_masquees = []
    bandes_inchangees = []
    facteurs = {}

    for i in range(nb_bandes):
        nom, desc = noms[i], descriptions[i]
        if _est_bande_masque(nom, desc) or nom in skip:
            bandes_masquees.append(nom)
            continue

        if isinstance(scale, dict):
            if nom not in scale:
                bandes_inchangees.append(nom)
                continue
            f = float(scale[nom])
            o = float(offset[nom]) if isinstance(offset, dict) and nom in offset else (
                0.0 if offset is None or isinstance(offset, dict) else float(offset))
        else:
            f, o = facteur, decalage

        if f == 1.0 and o == 0.0:
            bandes_inchangees.append(nom)
            continue

        sortie[i] = arr[i] * f + o
        facteurs[nom] = (f, o)

    infos = {
        "facteur": facteur,
        "decalage": decalage,
        "regle": justification,
        "bandes_masquees": bandes_masquees,
        "bandes_inchangees": bandes_inchangees,
        "facteurs": facteurs,
        "valeurs_max_avant": val_max,
    }
    return sortie, infos


class Scene:
    """Conteneur simple pour une scène satellite.

    Attributs
    ---------
    bands : dict[str, np.ndarray]
        Dictionnaire nom_de_bande -> tableau 2D (float64). Les valeurs
        nodata sont converties en NaN. Les bandes de réflectance sont
        ramenées dans l'ordre de grandeur physique (voir ``echelle``).
    profile : dict
        Profil rasterio (driver, crs, transform, width, height, ...).
    crs, transform : issus du profil.
    shape : (hauteur, largeur).
    sensor : nom du capteur.
    descriptions : list[str]
        Descriptions de bandes lues dans le fichier.
    echelle : dict
        Règle de mise à l'échelle réellement appliquée (facteur, décalage,
        justification, bandes masquées).
    """

    def __init__(self, bands, profile, sensor, descriptions=None, echelle=None):
        self.bands = bands
        self.profile = profile
        self.crs = profile.get("crs")
        self.transform = profile.get("transform")
        self.sensor = sensor
        self.descriptions = list(descriptions or [])
        self.echelle = dict(echelle or {})
        first = next(iter(bands.values()))
        self.shape = (first.shape[0], first.shape[1])

    @property
    def bandes_masquees(self):
        """Noms des bandes de classification laissées en valeurs entières."""
        return list(self.echelle.get("bandes_masquees", []))

    def __repr__(self):
        return (f"Scene(sensor={self.sensor!r}, shape={self.shape}, "
                f"bands={list(self.bands.keys())})")


def read_scene(path, sensor="sentinel2", band_names=None, scale="auto",
               offset=None, skip_scale=None):
    """Lit un GeoTIFF multibande et renvoie un objet :class:`Scene`.

    Paramètres
    ----------
    path : str
        Chemin du GeoTIFF.
    sensor : str
        'sentinel2' | 'landsat8' | 'landsat9' | 'landsat7' | 'landsat5'
        | 'drone' | 'generic'. Détermine la correspondance des bandes.
    band_names : list[str] | None
        Force les noms de bandes (dans l'ordre du fichier). Prioritaire
        sur la correspondance automatique.
    scale, offset : 'auto' | float | dict | None
        Mise à l'échelle des réflectances. Voir :func:`appliquer_echelle`.
        Par défaut 'auto' : le facteur du produit est déduit des descriptions
        de bandes et du capteur. Les bandes de classification (SCL, QA,
        masques) en sont toujours exclues.
    skip_scale : list[str] | None
        Noms de bandes supplémentaires à ne pas mettre à l'échelle.
    """
    with rasterio.open(path) as src:
        count = src.count
        arr = src.read().astype("float64")   # (bandes, h, w)
        profile = src.profile.copy()
        nodata = src.nodata
        descriptions = list(src.descriptions or [])

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    names = list(band_names) if band_names else _default_names(sensor, count)
    if len(names) < count:
        names += [f"band_{i + 1}" for i in range(len(names), count)]

    arr, echelle = appliquer_echelle(
        arr, names, descriptions=descriptions,
        scale=scale, offset=offset, skip_scale=skip_scale, capteur=sensor)

    bands = {names[i]: arr[i] for i in range(count)}
    return Scene(bands, profile, sensor, descriptions=descriptions,
                 echelle=echelle)


__all__ = ["read_scene", "Scene", "appliquer_echelle",
           "SENTINEL2_L2A_SCALE", "LANDSAT_C2_SR_SCALE", "LANDSAT_C2_SR_OFFSET"]
