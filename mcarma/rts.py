"""mcarma.rts -- refit-free RTS-smoother reconstruction helpers.

Posterior reconstruction of a latent light curve on a dense grid, from a
fitted-or-true CARMA state space plus the observed data.  Factored out of the
per-corpus figure scripts (fits/21sim/rts_bigindep.py, fits/21sim/rts_lsst.py)
so every reconstruction figure shares one code path: the +/-2 sigma band per
reference band and the stacked-panel plot.  The state space (F, G, H, V) is
built by the caller (from a flat Jones theta for the independent corpus, from
the physical truth for the LSST corpus); this module only smooths and draws.

Imports mcarma.smoother, which loads JAX, so -- like the smoother -- this
module is NOT imported by mcarma.__init__.  Import it explicitly on a compute
node or a laptop with JAX available.
"""
import numpy as np

from .smoother import smooth

# shared palette / marker style so the figures do not drift per script
_BAND_COLOR = "#3b6fb0"
_MEAN_COLOR = "#1f3f6f"
_OBS_MFC = "#d24b3a"

# Font sizes in points, used when a caller passes no ``sizes``. A caller that
# knows how wide the figure will be printed should override these: a label's
# size on the page is its size here times the printed width over the canvas
# width, so a figure drawn at 11 in and printed at 7 in loses a third of it.
# See fits/21sim/figstyle.py, which inverts that ratio.
_DEFAULT_SIZES = dict(suptitle=12.0, title=11.0, label=10.0, tick=10.0,
                      legend=9.0, annot=9.0)


def _sizes(sizes):
    sz = dict(_DEFAULT_SIZES)
    sz.update(sizes or {})
    return sz


def reconstruct_band(t, y, band, R, F, G, H, V, ref_band, t_eval, mu=None):
    """Posterior mean + std of one reference band on a dense grid, refit-free.

    Runs the RTS smoother over the observed (t, y, band, R) light curve at the
    supplied state space (F, G, H, V) and evaluates the reference band on the
    dense t_eval grid.  The observed values are assumed per-band centered
    unless an observation mean ``mu`` (shape (d, 1)) is supplied.

    Returns dict with keys ``yhat`` and ``std`` (each len(t_eval)) plus the
    reference-band observations ``t_obs`` and ``y_obs``.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    band = np.asarray(band, int)
    R = np.asarray(R, float)
    t_eval = np.asarray(t_eval, float)
    d = H.shape[0]
    C_list = [np.eye(d)[b:b + 1, :] for b in band]
    R_list = [np.array([[r]]) for r in R]
    if mu is None:
        mu = np.zeros((d, 1))
    bands_eval = np.full(len(t_eval), int(ref_band), dtype=int)
    out = smooth(t, y, F, G, H, V, C_list, R_list, mu,
                 t_eval=t_eval, bands_eval=bands_eval)
    m = band == int(ref_band)
    return {"yhat": out["yhat_eval"], "std": out["std_eval"],
            "t_obs": t[m], "y_obs": y[m], "R_obs": np.asarray(R, float)[m]}


def plot_order_panels(panels, out, suptitle, ylabel, obs_label,
                      xlabel="time since start (days)", figsize=None,
                      sizes=None):
    """Stacked reconstruction panels, one per generative order or regime.

    Each entry of ``panels`` is a dict with the dense time axis ``x``, the
    posterior ``yhat`` and ``std`` on ``x``, the reference-band observations
    ``t_obs``/``y_obs`` (same time origin as ``x``), and an annotation
    ``title``.  Draws the +/-2 sigma band, the smoother mean, and the
    observations, and writes ``out``.

    ``sizes`` overrides ``_DEFAULT_SIZES`` for a caller that knows the printed
    width.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sz = _sizes(sizes)
    n = len(panels)
    if figsize is None:
        figsize = (9.2, 2.75 * n + 0.5)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, pan in zip(axes, panels):
        x = np.asarray(pan["x"], float)
        yhat = np.asarray(pan["yhat"], float)
        std = np.asarray(pan["std"], float)
        ax.fill_between(x, yhat - 2 * std, yhat + 2 * std, color=_BAND_COLOR,
                        alpha=0.35, linewidth=0, label=r"$\pm2\sigma$ band")
        ax.plot(x, yhat, "-", color=_MEAN_COLOR, lw=1.3, label="RTS mean")
        ax.plot(np.asarray(pan["t_obs"], float), np.asarray(pan["y_obs"], float),
                "o", ms=3.4, mfc=_OBS_MFC, mec="0.25", mew=0.3, alpha=0.9,
                label=obs_label)
        ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
        ax.set_ylabel(ylabel, fontsize=sz["label"])
        ax.tick_params(axis="both", labelsize=sz["tick"])
        ax.text(0.015, 0.87, pan["title"], transform=ax.transAxes,
                fontsize=sz["annot"],
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))
    axes[0].legend(loc="upper right", fontsize=sz["legend"], framealpha=0.9,
                   ncol=3)
    axes[-1].set_xlabel(xlabel, fontsize=sz["label"])
    # A figure that goes into a manuscript carries its description in the
    # caption, not in a title inside the image, so suptitle is optional.
    if suptitle:
        fig.suptitle(suptitle, fontsize=sz["suptitle"])
    fig.tight_layout(rect=[0, 0, 1, 0.97 if suptitle else 1.0])
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def plot_order_grid(panels, out, col_titles, row_labels, suptitle, ylabel,
                    obs_labels, xlabel="time since start (days)",
                    figsize=None, sharey_row=True, sizes=None, sharex=True,
                    sharey=None, legend_anchor=(0.5, 0.918)):
    """Reconstruction panels on a (order x study) grid.

    ``panels`` is a list of columns, each a list of per-order panel dicts in the
    same format ``plot_order_panels`` takes.  With ``sharex=True`` every column
    is drawn on a common x range so the observing designs are compared at the
    same days per inch, which is the whole point of the side-by-side: the
    columns differ in how many points fall in the same window, not in how much
    time is on show.  Pass ``sharex="col"`` when a column is deliberately shown
    over a different span (a survey run out to its full length, say); the caller
    then owes the reader the span in that column's title, because the days per
    inch no longer match.  Rows share a y range by default so the +/-2 sigma
    bands are comparable across the row.

    ``obs_labels`` is one legend label per column (the reference band can differ
    between studies).  ``sizes`` overrides ``_DEFAULT_SIZES`` for a caller that
    knows the printed width.  ``sharey`` overrides ``sharey_row`` outright and
    takes anything ``plt.subplots`` takes; pass ``"col"`` when it is the column
    and not the row that carries one process under different observing designs,
    so the comparison the column exists to make is read off a common y range.
    Returns ``out``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sz = _sizes(sizes)
    ncol = len(panels)
    nrow = len(panels[0])
    if any(len(c) != nrow for c in panels):
        raise ValueError("every column must carry the same number of rows")
    if figsize is None:
        figsize = (5.6 * ncol, 2.5 * nrow + 0.8)
    shy = sharey if sharey is not None else ("row" if sharey_row else False)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=sharex,
                             sharey=shy, squeeze=False)
    for j, col in enumerate(panels):
        for i, pan in enumerate(col):
            ax = axes[i][j]
            x = np.asarray(pan["x"], float)
            yhat = np.asarray(pan["yhat"], float)
            std = np.asarray(pan["std"], float)
            ax.fill_between(x, yhat - 2 * std, yhat + 2 * std,
                            color=_BAND_COLOR, alpha=0.35, linewidth=0,
                            label=r"$\pm2\sigma$ band")
            ax.plot(x, yhat, "-", color=_MEAN_COLOR, lw=1.2, label="RTS mean")
            ax.plot(np.asarray(pan["t_obs"], float),
                    np.asarray(pan["y_obs"], float), "o", ms=2.8,
                    mfc=_OBS_MFC, mec="0.25", mew=0.25, alpha=0.9,
                    label=obs_labels[j])
            ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
            # Anchored at the TOP of the box, not at the first line's baseline:
            # with the default baseline anchor a two-line note grows upward out
            # of the axes and, in the first row, into the column title above it.
            ax.text(0.015, 0.985, pan["title"], transform=ax.transAxes,
                    va="top", ha="left", fontsize=sz["annot"],
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="0.7"))
            ax.tick_params(axis="both", labelsize=sz["tick"])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=sz["title"])
            if j == 0:
                # The row label only. ``ylabel`` goes on the figure below:
                # stacked on one axes the two strings are taller than a row,
                # so consecutive rows' labels ran into each other.
                ax.set_ylabel(row_labels[i], fontsize=sz["label"])
            if i == nrow - 1:
                ax.set_xlabel(xlabel, fontsize=sz["label"])
    # Headroom for the annotation box, which is opaque and sits in the top left.
    # Without it the box hides whatever the reconstruction does at the top of
    # its range, which on an oscillating row is a peak rather than dead space.
    # 0.32 rather than 0.22 because the box is now anchored at its top edge and
    # so sits a fifth of its own height lower than it did.
    # One axes per shared group, or the widening compounds once per member and
    # the last one ends up nearly twice as tall as its data.
    if shy == "row":
        reps = [axes[i][0] for i in range(nrow)]
    elif shy == "col":
        reps = [axes[0][j] for j in range(ncol)]
    elif shy is True or shy == "all":
        reps = [axes[0][0]]
    else:
        reps = [ax for row in axes for ax in row]
    for ax in reps:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.32 * (hi - lo))
    # One figure-level legend rather than one per axes: the per-axes placement
    # collides with the annotation box in the top row, and the three mark types
    # mean the same thing in every panel.
    handles, labels = axes[0][0].get_legend_handles_labels()
    for j in range(1, ncol):            # a column may name its marks apart
        hj, lj = axes[0][j].get_legend_handles_labels()
        for h, lab in zip(hj, lj):
            if lab not in labels:
                handles, labels = handles + [h], labels + [lab]
    # The band across the top holds three things in this order: the suptitle,
    # the legend, then the two-line column titles. tight_layout subtracts the
    # suptitle's own height from the rect, so the rect only has to reserve the
    # legend; setting it as low as the column titles should sit leaves a dead
    # band the width of the suptitle between the legend and the panels. With no
    # suptitle there is nothing above the legend to clear, so the caller raises
    # ``legend_anchor`` to the top of the figure; left at the default it lands
    # on the column titles instead.
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=sz["legend"], framealpha=0.9,
               bbox_to_anchor=legend_anchor)
    fig.suptitle(suptitle, fontsize=sz["suptitle"])
    fig.supylabel(ylabel, fontsize=sz["label"], x=0.012)
    fig.tight_layout(rect=[0.03, 0, 1, 0.955])
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out
