from skimage import feature

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from skimage.util import img_as_float


def sigma_setter(image, σ=1.0, bounds=None):
    """
    Zobrazí obrázek s překrytými hranami (Canny) a nechá uživatele
    doladit sigma/low/high. Po stisku 'Done' vrátí zvolené parametry.

    :param image: 2D numpy (grayscale)
    :param σ: počáteční sigma (float)
    :param bounds: [left, right, top, bottom] pro ořez pohledu
    :return: (sigma, low, high)
    """
    # Převod na float v [0,1] – Canny v skimage předpokládá float
    img = img_as_float(image)

    # Výchozí prahy, když nejsou dané: něco rozumného v [0,1]
    low0 = 0.1
    high0 = 0.3

    plt.ioff()
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap="gray")
    ax.set_title("Canny preview")
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[3], bounds[2])  # [top, bottom] -> y je obráceně

    # Překryv hran (boolean) jako další imshow
    edges = feature.canny(img, sigma=σ, low_threshold=low0, high_threshold=high0)
    edges_im = ax.imshow(edges, alpha=0.6)  # jednoduchý overlay
    plt.subplots_adjust(left=0.1, bottom=0.28)

    # --- Slidery ---
    ax_sigma = plt.axes([0.15, 0.18, 0.7, 0.03])
    ax_low   = plt.axes([0.15, 0.13, 0.7, 0.03])
    ax_high  = plt.axes([0.15, 0.08, 0.7, 0.03])

    s_sigma = Slider(ax_sigma, "sigma", 0.0, 10.0, valinit=float(σ))
    s_low   = Slider(ax_low,   "low",   0.0, 1.0,  valinit=low0)
    s_high  = Slider(ax_high,  "high",  0.0, 1.0,  valinit=high0)

    # --- Tlačítka ---
    ax_done  = plt.axes([0.15, 0.02, 0.2, 0.04])
    ax_reset = plt.axes([0.38, 0.02, 0.2, 0.04])

    b_done  = Button(ax_done, "Done")
    b_reset = Button(ax_reset, "Reset")

    # Stav pro návratové hodnoty
    ret = {"sigma": float(σ), "low": low0, "high": high0}

    def recompute(*_):
        # garantuj high >= low
        low = s_low.val
        high = s_high.val
        if high < low:
            high = low
            s_high.set_val(high)  # vizuálně srovnáme

        sig = s_sigma.val
        e = feature.canny(img, sigma=sig, low_threshold=low, high_threshold=high)
        edges_im.set_data(e)

        ret["sigma"] = float(sig)
        ret["low"]   = float(low)
        ret["high"]  = float(high)
        fig.canvas.draw_idle()

    # napojení handlerů
    s_sigma.on_changed(recompute)
    s_low.on_changed(recompute)
    s_high.on_changed(recompute)

    def on_reset(event):
        s_sigma.reset()
        s_low.reset()
        s_high.reset()
    b_reset.on_clicked(on_reset)

    def on_done(event):
        plt.close(fig)
    b_done.on_clicked(on_done)

    # první přepočet, aby overlay odpovídal
    recompute()

    # blokující zobrazení okna
    fig.show()
    plt.pause(0.001)
    # dokud okno žije, čekáme (funguje v běžném skriptu; v Jupyteru použij %matplotlib qt)
    while plt.fignum_exists(fig.number):
        plt.pause(0.05)

    print(f"Proceeding with sigma = {ret['sigma'] :6.2f}")
    plt.ion()
    return ret["sigma"], ret["low"], ret["high"]


def extract_edges(image, σ=1.0, low=None, high=None, indices=True):
    """
    Spočítá hrany Canny.

    :param image: numpy grayscale image
    :param σ: sigma pro Canny
    :param low: dolní práh v [0,1] (pokud None, skimage si zvolí heuristiku)
    :param high: horní práh v [0,1]
    :param indices: True => vrátí souřadnice (x, y); False => vrátí boolean masku
    :return: np.ndarray souřadnic (N, 2) nebo boolean maska
    """
    img = img_as_float(image)
    edges = feature.canny(img, sigma=float(σ),
                          low_threshold=low, high_threshold=high)
    if not indices:
        return edges
    # stejné chování jako dřív: transpozice a np.argwhere => (x, y)
    return np.argwhere(edges.T)
