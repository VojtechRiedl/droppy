def auto_crop(image, pad=25, σ=1, low=None, high=None):
    """
    Memory-safe Hough: dávky poloměrů + volitelné zmenšení, BEZ posunu o max_R.
    Vrací bounds = [left, right, top, bottom].
    """
    min_top = min_left = 0
    H, W = image.shape
    max_bottom, max_right = H, W
    print('Performing auto-cropping, please wait...')

    # 1) Hrany
    edges = extract_edges(image, σ=σ, low=low, high=high, indices=False)

    # 2) Dočasné zmenšení (kvůli paměti/rychlosti)
    target_max_dim = 2000
    scale = max(1, int(np.ceil(max(H, W) / target_max_dim)))
    if scale > 1:
        edges_s = resize(edges.astype(float),
                         (H // scale, W // scale),
                         preserve_range=True,
                         anti_aliasing=False) > 0.5
    else:
        edges_s = edges

    # 3) Poloměry (v původním měřítku) + škálovaná sada pro menší obraz
    radii_Δ = 10
    hough_radii = np.arange(min(H, W) // 10, max(H, W), radii_Δ)
    if hough_radii.size == 0:
        raise RuntimeError("No Hough radii to search. Check image size or radii settings.")

    hough_radii_s = np.unique(np.maximum((hough_radii // scale).astype(int), 1))

    # 4) Hough po dávkách (šetří RAM)
    def _hough_circle_in_chunks(edges_bin, radii, chunk=8):
        best = None  # (accum, cx, cy, r)
        for i in range(0, len(radii), chunk):
            rs = radii[i:i+chunk]
            hspaces = hough_circle(edges_bin, rs, normalize=False)  # (len(rs), H, W)
            accums, cx, cy, r_found = hough_circle_peaks(
                hspaces, rs, total_num_peaks=1, normalize=False
            )
            if len(accums) > 0:
                cand = (accums[0], cx[0], cy[0], r_found[0])
                if (best is None) or (cand[0] > best[0]):
                    best = cand
            del hspaces
        return best

    best = _hough_circle_in_chunks(edges_s, hough_radii_s, chunk=8)
    if best is None:
        raise RuntimeError("No circle candidate found by Hough. Check edges/thresholds or radius range.")

    _, cx_s, cy_s, r_s = best

    # 5) Přepočet zpět do původního měřítka (žádný max_R! žádné z = (...))
    cx = int(round(cx_s * scale))
    cy = int(round(cy_s * scale))
    r  = int(round(max(1, r_s * scale)))

    # 6) Baseline (na plném rozlišení); když hledáme ve spodním řezu, přičteme offset
    accums_l, angles, dists = hough_line_peaks(*hough_line(edges), num_peaks=1)
    baseline_y = None
    if len(angles) > 0:
        theta = angles[0]
        rho = dists[0]
        if np.abs(theta) < np.deg2rad(80):
            slice_top = max(cy - r - pad, 0)
            accums_l, angles, dists = hough_line_peaks(*hough_line(edges[slice_top:, :]), num_peaks=1)
            if len(angles) > 0 and np.abs(angles[0]) >= np.deg2rad(80):
                rho = dists[0] + slice_top  # přepočet na absolutní y
            else:
                rho = None
        baseline_y = None if rho is None else int(round(rho))

    # Fallback: když se baseline nenašla, vezmi spodní hranu kružnice
    if baseline_y is None:
        baseline_y = min(H - 1, cy + r)

    # 7) Bounds z centra + poloměru (už žádný max_R/posuny)
    left   = cx - r - pad
    right  = cx + r + pad
    top    = cy - r - pad
    bottom = baseline_y + pad

    # Klip do obrazu
    left   = max(min_left,  min(W - 1, left))
    right  = max(min_left,  min(W - 1, right))
    top    = max(min_top,   min(H - 1, top))
    bottom = max(min_top,   min(H - 1, bottom))

    # Ujisti se o správném pořadí
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top

    bounds = np.array([left, right, top, bottom], dtype=int)
    return bounds