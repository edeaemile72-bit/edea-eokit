"""Utilitaires de visualisation (complément léger à matplotlib).

Le notebook trace directement avec matplotlib ; ces fonctions sont des
raccourcis optionnels pour afficher un indice ou une composition colorée.
"""

import numpy as np


def _lazy_plt():
    import matplotlib.pyplot as plt
    return plt


def show_index(arr, titre="", cmap="RdYlBu", vmin=-1, vmax=1):
    """Affiche un indice normalisé avec une barre de couleur."""
    plt = _lazy_plt()
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(titre)
    plt.colorbar(im, fraction=0.046)
    plt.axis("off")
    plt.show()


def stretch(arr, p_low=2, p_high=98):
    """Étirement de contraste par percentiles -> [0, 1] pour l'affichage."""
    a = arr.astype("float64")
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return a
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        return np.clip(a, 0, 1)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def rgb_composite(scene, bands=("red", "green", "blue")):
    """Composition colorée (H, W, 3) étirée pour l'affichage."""
    chans = [stretch(scene.bands[b]) for b in bands]
    return np.dstack(chans)


__all__ = ["show_index", "stretch", "rgb_composite"]
