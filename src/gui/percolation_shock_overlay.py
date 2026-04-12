"""
percolation_shock_overlay.py — Colour-map provider for shock distance rendering.

Pure function: maps a normalised distance array → brush array.
No allocations beyond the returned array.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg


def shock_brushes(shock_norm: np.ndarray, alive_mask: np.ndarray) -> np.ndarray:
    """
    Map normalised shock distance [0, 1] to a red→orange→yellow gradient.

    Parameters
    ----------
    shock_norm : (N,) float64, values in [0, 1].
    alive_mask : (N,) bool.

    Returns
    -------
    brushes : (N,) object array of pg.mkBrush instances.
    """
    N = len(shock_norm)
    brushes = np.empty(N, dtype=object)

    for i in range(N):
        if not alive_mask[i]:
            brushes[i] = pg.mkBrush(100, 100, 100, 80)
            continue

        d = shock_norm[i]
        if d <= 0:
            # LCC origin → bright red
            brushes[i] = pg.mkBrush(248, 81, 73, 220)
        elif d < 0.5:
            # Red → orange
            r = 248
            g = int(81 + d * 2 * (180 - 81))
            brushes[i] = pg.mkBrush(r, g, 30, 200)
        else:
            # Orange → yellow fading
            r = int(248 - (d - 0.5) * 2 * 50)
            g = int(180 + (d - 0.5) * 2 * 75)
            alpha = int(200 - (d - 0.5) * 2 * 100)
            brushes[i] = pg.mkBrush(r, g, 50, max(alpha, 60))

    return brushes
