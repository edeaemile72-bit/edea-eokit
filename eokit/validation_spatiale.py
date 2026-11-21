"""Validation spatiale et biais optimiste en classification supervisee.

Sous-module complementaire d'EDEA-EOKit. Il quantifie l'effet de
l'autocorrelation spatiale sur le biais optimiste des indicateurs de
precision, en comparant plusieurs strategies de validation sur une
classification supervisee d'images Sentinel-2.

Auteur : Dr. Ir. EDEA O. Emile, Enseignant-chercheur
         Expert en geomatique et analyse spatiale

Principe
--------
Le classificateur (Random Forest) reste IDENTIQUE pour toutes les
strategies ; seul le decoupage train/test change. L'ecart de precision
entre la validation aleatoire et la validation spatialement separee
mesure le biais optimiste induit par l'autocorrelation spatiale.

Strategies comparees
--------------------
- 'aleatoire'   : k-fold aleatoire classique (strategie naive, biaisee).
- 'buffer'      : k-fold avec retrait des points d'entrainement situes
                  dans un buffer autour des points de test.
- 'hold_out'    : hold-out spatial (clusters KMeans sur les coordonnees,
                  chaque cluster teste a tour de role).
- 'block_cv'    : spatial block cross-validation (grille de blocs
                  reguliers) -> strategie de REFERENCE.

Entrees attendues
-----------------
- scene : objet eokit.io.Scene (lu par eokit.io.read_scene).
- points : GeoDataFrame geopandas de points etiquetes (verite terrain),
           avec un champ de classe.
"""

import numpy as np
import pandas as pd

from . import indices as _indices


# ---------------------------------------------------------------------------
# 1) Echantillonnage des variables (bandes + indices) aux points
# ---------------------------------------------------------------------------
def echantillonner_features(scene, points, champ_classe,
                            noms_indices=("ndvi", "ndwi", "mndwi"),
                            noms_bandes=None):
    """Extrait la matrice de variables X, les classes y et les coordonnees.

    Parametres
    ----------
    scene : eokit.io.Scene
    points : geopandas.GeoDataFrame (geometrie de points)
    champ_classe : str
        Nom du champ contenant la classe (entier ou chaine).
    noms_indices : tuple[str]
        Indices eokit a ajouter comme variables (ignores si bande absente).
    noms_bandes : list[str] | None
        Bandes a utiliser ; par defaut toutes les bandes de la scene.

    Retour
    ------
    X : np.ndarray (n, p)   variables (bandes + indices)
    y : np.ndarray (n,)     classes
    coords : np.ndarray (n, 2)  coordonnees (x, y) dans le CRS metrique
    noms_features : list[str]   noms des colonnes de X
    """
    import geopandas as gpd  # import local : dependance seulement si utilise

    if not isinstance(points, gpd.GeoDataFrame):
        raise TypeError("'points' doit etre un GeoDataFrame geopandas.")
    if champ_classe not in points.columns:
        raise KeyError(f"Champ de classe absent : {champ_classe!r}. "
                       f"Colonnes : {list(points.columns)}")

    # Reprojection des points vers le CRS de la scene (coherence spatiale).
    gdf = points
    if scene.crs is not None and gdf.crs is not None and gdf.crs != scene.crs:
        gdf = gdf.to_crs(scene.crs)
        print(f"[eokit] Points reprojetes vers le CRS de la scene ({scene.crs}).")

    # Avertissement si CRS geographique (degres) : fausse les distances/Moran.
    try:
        if gdf.crs is not None and gdf.crs.is_geographic:
            print("[eokit] ATTENTION : CRS geographique (degres). Reprojetez "
                  "vos donnees dans un CRS metrique (ex. UTM) avant d'analyser "
                  "les distances et le Moran I.")
    except Exception:
        pass

    # Bandes retenues + arrays d'indices precalcules.
    noms_bandes = list(noms_bandes) if noms_bandes else list(scene.bands.keys())
    arrs = {b: scene.bands[b] for b in noms_bandes}
    idx_arrs = _indices.compute_many(list(noms_indices), scene.bands)
    noms_features = noms_bandes + list(idx_arrs.keys())

    t = scene.transform
    h, w = scene.shape
    inv = ~t  # transform inverse (coord -> pixel), via affine

    X_list, y_list, xy_list = [], [], []
    for geom, cls in zip(gdf.geometry.values, gdf[champ_classe].values):
        if geom is None or geom.is_empty:
            continue
        mx, my = float(geom.x), float(geom.y)
        col, row = inv * (mx, my)
        col, row = int(col), int(row)
        if row < 0 or col < 0 or row >= h or col >= w:
            continue  # point hors emprise de la scene
        vals = [arrs[b][row, col] for b in noms_bandes]
        vals += [idx_arrs[k][row, col] for k in idx_arrs]
        if not np.all(np.isfinite(vals)):
            continue  # pixel nodata / NaN -> point ecarte
        X_list.append(vals)
        y_list.append(cls)
        xy_list.append((mx, my))

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list)
    coords = np.asarray(xy_list, dtype=float)
    print(f"[eokit] Echantillons retenus : {X.shape[0]} | "
          f"variables : {X.shape[1]} ({', '.join(noms_features)})")
    return X, y, coords, noms_features


# ---------------------------------------------------------------------------
# 2) Autocorrelation spatiale : indice de Moran I (poids k plus proches voisins)
# ---------------------------------------------------------------------------
def _poids_knn(coords, k=8):
    """Matrice de voisinage k-NN (listes d'indices voisins), row-standardisee."""
    from scipy.spatial import cKDTree
    n = coords.shape[0]
    k = min(k, n - 1)
    tree = cKDTree(coords)
    # +1 car le point lui-meme est son plus proche voisin.
    _, idx = tree.query(coords, k=k + 1)
    voisins = [row[1:] for row in idx]  # retire soi-meme
    return voisins


def moran_i(valeurs, coords, k=8, permutations=999, graine=42):
    """Indice de Moran I global avec test par permutations.

    Parametres
    ----------
    valeurs : array (n,)   variable numerique (ex. residus 0/1).
    coords  : array (n, 2) coordonnees metriques.
    k : int                nombre de voisins (poids k-NN row-standardises).
    permutations : int     nombre de permutations pour la p-value empirique.

    Retour
    ------
    dict : {'I', 'E_I', 'z', 'p_perm', 'k'}
    """
    x = np.asarray(valeurs, dtype=float)
    n = x.size
    if n < 3 or np.allclose(x, x[0]):
        return {"I": np.nan, "E_I": np.nan, "z": np.nan,
                "p_perm": np.nan, "k": k}
    voisins = _poids_knn(coords, k=k)
    z = x - x.mean()
    denom = np.sum(z ** 2)

    def _calc_I(zc):
        num = 0.0
        W = 0.0
        for i, vs in enumerate(voisins):
            m = len(vs)
            if m == 0:
                continue
            wij = 1.0 / m  # row-standardisation
            num += wij * np.sum(zc[i] * zc[vs])
            W += wij * m
        return (n / W) * (num / denom) if (W > 0 and denom > 0) else np.nan

    I = _calc_I(z)
    E_I = -1.0 / (n - 1)

    # Test par permutations : on reaffecte aleatoirement les valeurs.
    rng = np.random.default_rng(graine)
    I_perm = np.empty(permutations)
    for p in range(permutations):
        zp = rng.permutation(z)
        I_perm[p] = _calc_I(zp)
    sd = np.nanstd(I_perm)
    zscore = (I - np.nanmean(I_perm)) / sd if sd > 0 else np.nan
    # p-value bilaterale empirique.
    extremes = np.sum(np.abs(I_perm - np.nanmean(I_perm)) >= abs(I - np.nanmean(I_perm)))
    p_perm = (extremes + 1) / (permutations + 1)
    return {"I": float(I), "E_I": float(E_I), "z": float(zscore),
            "p_perm": float(p_perm), "k": k}


# ---------------------------------------------------------------------------
# 3) Distance train-test
# ---------------------------------------------------------------------------
def _distance_min_mediane(coords_train, coords_test):
    """Distance mediane entre chaque point de test et le point train le plus proche."""
    from scipy.spatial import cKDTree
    if len(coords_train) == 0 or len(coords_test) == 0:
        return np.nan
    tree = cKDTree(coords_train)
    d, _ = tree.query(coords_test, k=1)
    return float(np.median(d))


# ---------------------------------------------------------------------------
# 4) Generation des decoupages train/test par strategie
# ---------------------------------------------------------------------------
def _folds_aleatoire(coords, k_folds, graine):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=graine)
    return list(kf.split(coords)), None


def _blocs_grille(coords, taille_bloc):
    """Attribue a chaque point un identifiant de bloc selon une grille reguliere."""
    xmin, ymin = coords.min(axis=0)
    gx = np.floor((coords[:, 0] - xmin) / taille_bloc).astype(int)
    gy = np.floor((coords[:, 1] - ymin) / taille_bloc).astype(int)
    return gx * 100000 + gy  # identifiant de bloc unique


def _folds_groupes(coords, groupes, k_folds, graine):
    """k-fold sur des groupes spatiaux (aucun groupe a cheval train/test)."""
    from sklearn.model_selection import GroupKFold
    uniq = np.unique(groupes)
    n_splits = min(k_folds, len(uniq))
    if n_splits < 2:
        # Un seul groupe : repli sur un hold-out aleatoire.
        return _folds_aleatoire(coords, k_folds, graine)[0], groupes
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(coords, groups=groupes)), groupes


def _folds_block_cv(coords, k_folds, taille_bloc, graine):
    groupes = _blocs_grille(coords, taille_bloc)
    return _folds_groupes(coords, groupes, k_folds, graine)


def _folds_hold_out(coords, k_folds, graine):
    """Clusters spatiaux (KMeans) testes a tour de role : hold-out spatial."""
    from sklearn.cluster import KMeans
    n_clusters = min(k_folds, coords.shape[0])
    km = KMeans(n_clusters=n_clusters, random_state=graine, n_init=10)
    groupes = km.fit_predict(coords)
    return _folds_groupes(coords, groupes, k_folds, graine)


def _appliquer_buffer(train_idx, test_idx, coords, buffer_m):
    """Retire du train les points situes a moins de buffer_m d'un point de test."""
    from scipy.spatial import cKDTree
    if buffer_m <= 0 or len(test_idx) == 0:
        return train_idx
    tree = cKDTree(coords[test_idx])
    d, _ = tree.query(coords[train_idx], k=1)
    garde = train_idx[d > buffer_m]
    return garde


# ---------------------------------------------------------------------------
# 5) Evaluation d'une strategie (predictions hors-echantillon poolees)
# ---------------------------------------------------------------------------
def _evaluer_strategie(X, y, coords, folds, buffer_m=0.0,
                       n_estimators=300, graine=42, k_moran=8,
                       permutations=999):
    """Entraine le RF sur chaque fold, agrege les predictions hors-echantillon."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score)

    n = X.shape[0]
    y_true_all, y_pred_all, idx_all = [], [], []
    distances = []
    for train_idx, test_idx in folds:
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        if buffer_m > 0:
            train_idx = _appliquer_buffer(train_idx, test_idx, coords, buffer_m)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        if len(np.unique(y[train_idx])) < 2:
            continue  # une seule classe en apprentissage -> fold ignore
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     random_state=graine, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)
        idx_all.append(test_idx)
        distances.append(_distance_min_mediane(coords[train_idx], coords[test_idx]))

    if not y_true_all:
        return None
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    idx = np.concatenate(idx_all)

    oa = accuracy_score(y_true, y_pred) * 100.0
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    dist_med = float(np.nanmedian(distances)) if distances else np.nan

    # Moran I des residus (1 = erreur) sur les points predits.
    residus = (y_true != y_pred).astype(float)
    mi = moran_i(residus, coords[idx], k=k_moran, permutations=permutations,
                 graine=graine)
    return {"OA": oa, "Kappa": kappa, "F1_macro": f1,
            "Dist_min_m": dist_med, "Moran_I_residus": mi["I"],
            "Moran_p": mi["p_perm"], "n_test": int(y_true.size)}


# ---------------------------------------------------------------------------
# 6) Fonction principale : comparaison des strategies -> tableau des biais
# ---------------------------------------------------------------------------
def comparer_strategies(scene, points, champ_classe,
                        noms_indices=("ndvi", "ndwi", "mndwi"),
                        noms_bandes=None, k_folds=5, buffer_m=100.0,
                        taille_bloc=500.0, n_estimators=300,
                        strategie_reference="block_cv",
                        graine=42, k_moran=8, permutations=499):
    """Compare les strategies de validation et renvoie le tableau des biais.

    Le classificateur (Random Forest) est identique partout ; seul le
    decoupage train/test change. Le biais optimiste est l'ecart d'OA par
    rapport a la strategie de reference spatiale.

    Retour
    ------
    pandas.DataFrame indexe par strategie, colonnes :
    OA, Kappa, F1_macro, Dist_min_m, Moran_I_residus, Moran_p, Biais_OA.
    """
    X, y, coords, _ = echantillonner_features(
        scene, points, champ_classe, noms_indices=noms_indices,
        noms_bandes=noms_bandes)
    if X.shape[0] < k_folds * 2:
        raise ValueError(f"Trop peu d'echantillons ({X.shape[0]}) pour "
                         f"{k_folds} folds.")

    plans = {
        "aleatoire": _folds_aleatoire(coords, k_folds, graine)[0],
        "buffer": _folds_aleatoire(coords, k_folds, graine)[0],
        "hold_out": _folds_hold_out(coords, k_folds, graine)[0],
        "block_cv": _folds_block_cv(coords, k_folds, taille_bloc, graine)[0],
    }
    buffers = {"aleatoire": 0.0, "buffer": buffer_m,
               "hold_out": 0.0, "block_cv": 0.0}
    libelles = {
        "aleatoire": "Aleatoire k-fold",
        "buffer": f"Buffer d'exclusion ({buffer_m:.0f} m)",
        "hold_out": "Hold-out spatial (KMeans)",
        "block_cv": f"Spatial block CV ({taille_bloc:.0f} m)",
    }

    lignes = {}
    for cle, folds in plans.items():
        print(f"[eokit] Evaluation : {libelles[cle]} ...")
        res = _evaluer_strategie(X, y, coords, folds, buffer_m=buffers[cle],
                                 n_estimators=n_estimators, graine=graine,
                                 k_moran=k_moran, permutations=permutations)
        if res is not None:
            lignes[libelles[cle]] = res

    df = pd.DataFrame(lignes).T
    # Biais optimiste = OA strategie - OA reference spatiale.
    ref_lib = libelles.get(strategie_reference)
    if ref_lib in df.index:
        df["Biais_OA"] = df["OA"] - df.loc[ref_lib, "OA"]
    df = df.round({"OA": 2, "Kappa": 3, "F1_macro": 3, "Dist_min_m": 1,
                   "Moran_I_residus": 3, "Moran_p": 3, "Biais_OA": 2})
    return df


# ---------------------------------------------------------------------------
# 7) Exports optionnels (Excel + graphique) avec identite de l'auteur
# ---------------------------------------------------------------------------
_SIGNATURE = ("Dr. Ir. EDEA O. Emile - Enseignant-chercheur, "
              "Expert en geomatique et analyse spatiale")


def exporter_excel(df, chemin):
    """Exporte le tableau des biais en .xlsx (identite de l'auteur incluse)."""
    import openpyxl
    from openpyxl.utils.dataframe import dataframe_to_rows
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Biais_validation"
    ws["A1"] = _SIGNATURE
    ws["A2"] = "Comparaison des strategies de validation - biais optimiste"
    ws.append([])
    ws.append(["Strategie"] + list(df.columns))
    for nom, row in df.iterrows():
        ws.append([nom] + [row[c] for c in df.columns])
    wb.save(chemin)
    print(f"[eokit] Tableau exporte : {chemin}")
    return chemin


def graphique_biais(df, chemin=None, colonne="OA"):
    """Diagramme en barres de l'indicateur par strategie."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df.index.astype(str), df[colonne].values, color="#2c7fb8")
    ax.set_ylabel(colonne)
    ax.set_title(f"{colonne} par strategie de validation")
    ax.set_xticklabels(df.index.astype(str), rotation=20, ha="right")
    fig.text(0.01, 0.01, _SIGNATURE, fontsize=7, color="gray")
    fig.tight_layout()
    if chemin:
        fig.savefig(chemin, dpi=150)
        print(f"[eokit] Graphique exporte : {chemin}")
    return fig


__all__ = ["echantillonner_features", "moran_i", "comparer_strategies",
           "exporter_excel", "graphique_biais"]

