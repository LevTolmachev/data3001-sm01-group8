"""
DATA3001 Group S01-G8 - Step 6: every figure used in the presentation and the report
=====================================================================================
Run AFTER steps 2 to 5 and 8 to 10. Reads their CSV / NPZ outputs and writes PNGs (300 dpi) to
<BASE>/models/figures. Figures are drawn at the size they appear on a slide, with text large enough
to read when projected.  Coastlines: Natural Earth (public domain), data/geo/south_america.geojson.
Run:  python step6_figures.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrowPatch, Ellipse, Rectangle

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
import step2_models_M0_M1_M2 as s2
from step2_models_M0_M1_M2 import NCELL, NX, NY, EXITS, CELL_LAT, CELL_LON, OUT, MIN_DRIFTERS, BASE, as_grid
import step4_drogue_season_gam_nn as s4

FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
S3, S4, S5, S8, S10, S12 = (os.path.join(OUT, d) for d in ("step3", "step4", "step5", "step8", "step10", "step12"))

INK, MUTED, GRID, BG = "#1B2A3A", "#5B6B7A", "#DDE3EA", "#F7F8FA"
BLUE, ORANGE, PURPLE, GREEN, GREY = "#1F6FB2", "#E07B1F", "#7A2E7E", "#1E9E6A", "#8C96A3"
LAND, NODATA = "#C9D1DB", "#EDF0F4"
font = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": font, "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
                     "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 1.2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK, "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG})
EXT = [s2.LON_MIN, s2.LON_MAX, s2.LAT_MIN, s2.LAT_MAX]
XC = np.linspace(s2.LON_MIN + 0.5, s2.LON_MAX - 0.5, NX)
YC = np.linspace(s2.LAT_MAX - 0.5, s2.LAT_MIN + 0.5, NY)
GEO = json.load(open(os.path.join(os.path.dirname(_here), "data", "geo", "south_america.geojson")))


def save(fig, name, tight=True):
    if tight:
        fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight", pad_inches=0.06)
    else:
        fig.savefig(os.path.join(FIG, name), dpi=300)
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------------- editable overlays
# Arrows, boxes, labels and (for the plain maps) the land itself are NOT drawn into the PNGs. They are
# written to figures/overlays.json in figure fractions, and the deck builder draws them as native,
# editable PowerPoint shapes on top of the image. Sizes are in points as they appear on the slide.
OVERLAYS = {}


class Overlay:
    def __init__(self, fig, name, image=True):
        self.fig, self.name, self.image, self.items = fig, name, image, []

    def _pt(self, ax, x, y, trans="data", dx_pt=0.0, dy_pt=0.0):
        t = ax.transData if trans == "data" else ax.transAxes
        X, Y = t.transform((x, y))
        X += dx_pt / 72 * self.fig.dpi
        Y += dy_pt / 72 * self.fig.dpi
        return X, Y

    def _frac(self, X, Y):
        return [round(X / self.fig.bbox.width, 5), round(1 - Y / self.fig.bbox.height, 5)]

    def arrow(self, ax, p0, p1, color, lw=4.0, rad=0.0, head=True, dash=None):
        self.items.append(("curve", ax, dict(p0=p0, p1=p1, rad=rad, color=color, lw=lw, head=head, dash=dash)))

    def line(self, ax, p0, p1, color, lw=1.2):
        self.arrow(ax, p0, p1, color, lw=lw, head=False)

    def rect(self, ax, x0, y0, x1, y1, edge=None, lw=2.0, fill=None, alpha=0.0, dash=None, ellipse=False):
        self.items.append(("rect", ax, dict(x0=x0, y0=y0, x1=x1, y1=y1, edge=edge, lw=lw, fill=fill, alpha=alpha, dash=dash, ellipse=ellipse)))

    def text(self, ax, x, y, text, size=14, color=INK, bold=True, italic=False, ha="center", va="center", bg=None,
             trans="data", dx_pt=0.0, dy_pt=0.0, lines=None):
        self.items.append(("text", ax, dict(x=x, y=y, text=text, size=size, color=color, bold=bold, italic=italic, ha=ha, va=va,
                                            bg=bg, trans=trans, dx_pt=dx_pt, dy_pt=dy_pt, lines=lines)))

    def title(self, ax, text, size=17):
        ax.set_title(" ", loc="left", pad=8, fontsize=size)
        self.text(ax, 0, 1, text, size=size, ha="left", va="bottom", trans="axes", dy_pt=3)

    def dot(self, ax, x, y, d_pt, fill, edge="FFFFFF", lw=1.5, text=None, tsize=14):
        self.items.append(("dot", ax, dict(x=x, y=y, d=d_pt, fill=fill, edge=edge, lw=lw, text=text, tsize=tsize)))

    def poly(self, ax, parts, fill, edge="FFFFFF", lw=0.75, name=None):
        self.items.append(("poly", ax, dict(parts=parts, fill=fill, edge=edge, lw=lw, name=name)))

    def export(self):
        self.fig.canvas.draw()
        out = []
        h = lambda c: None if c is None else c.lstrip("#").upper()
        for kind, ax, o in self.items:
            if kind == "curve":
                A, B = np.array(self._pt(ax, *o["p0"])), np.array(self._pt(ax, *o["p1"]))
                mid, dvec = (A + B) / 2, B - A
                Cc = mid + o["rad"] * np.array([dvec[1], -dvec[0]])
                out.append({"type": "curve", "p0": self._frac(*A), "c": self._frac(*Cc) if o["rad"] else None, "p1": self._frac(*B),
                            "color": h(o["color"]), "lw": o["lw"], "head": o["head"], "dash": o["dash"]})
            elif kind == "rect":
                a, b = self._frac(*self._pt(ax, o["x0"], o["y0"])), self._frac(*self._pt(ax, o["x1"], o["y1"]))
                out.append({"type": "ellipse" if o["ellipse"] else "rect", "x0": min(a[0], b[0]), "y0": min(a[1], b[1]),
                            "x1": max(a[0], b[0]), "y1": max(a[1], b[1]), "edge": h(o["edge"]), "lw": o["lw"],
                            "fill": h(o["fill"]), "alpha": o["alpha"], "dash": o["dash"]})
            elif kind == "text":
                p = self._frac(*self._pt(ax, o["x"], o["y"], o["trans"], o["dx_pt"], o["dy_pt"]))
                out.append({"type": "text", "x": p[0], "y": p[1], "text": o["text"], "lines": o["lines"], "size": o["size"], "color": h(o["color"]),
                            "bold": o["bold"], "italic": o["italic"], "ha": o["ha"], "va": o["va"], "bg": h(o["bg"])})
            elif kind == "dot":
                p = self._frac(*self._pt(ax, o["x"], o["y"]))
                out.append({"type": "dot", "x": p[0], "y": p[1], "d": o["d"], "fill": h(o["fill"]), "edge": h(o["edge"]), "lw": o["lw"],
                            "text": o["text"], "tsize": o["tsize"]})
            elif kind == "poly":
                parts = [[self._frac(*self._pt(ax, x, y)) for x, y in part] for part in o["parts"]]
                out.append({"type": "poly", "parts": parts, "fill": h(o["fill"]), "edge": h(o["edge"]), "lw": o["lw"], "name": o["name"]})
        W, H = self.fig.get_size_inches()
        OVERLAYS[self.name] = {"aspect": round(W / H, 5), "image": self.name + ".png" if self.image else None, "items": out}
        if self.image:
            save(self.fig, self.name + ".png", tight=False)
        else:
            plt.close(self.fig)
            print("overlay only", self.name)


def geo_parts(clip=None, tol=0.1, min_area=0.2):
    """Simplified country outlines (shapely), optionally clipped to (x0, y0, x1, y1)."""
    from shapely.geometry import shape, box
    out = {}
    for f in GEO["features"]:
        g = shape(f["geometry"]).buffer(0)
        if clip is not None:
            g = g.intersection(box(*clip))
        if g.is_empty:
            continue
        g = g.simplify(tol, preserve_topology=True)
        polys = [g] if g.geom_type == "Polygon" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]
        parts = [list(p.exterior.coords) for p in polys if p.area >= min_area]
        if parts:
            out[f["properties"]["NAME"]] = parts
    return out


def land(ax, color=LAND, edge="white", lw=0.8, zorder=3, highlight=None, hcolor=None):
    for f in GEO["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        hl = (highlight,) if isinstance(highlight, str) else (highlight or ())
        c = hcolor if f["properties"]["NAME"] in hl else color
        for p in polys:
            ax.add_patch(Polygon(np.array(p[0]), closed=True, facecolor=c, edgecolor=edge, lw=lw, zorder=zorder))


def map_ax(ax, title="", ylab=True, xlab=True, pad=0.0):
    ax.set_title(title, loc="left", pad=8)
    ax.set_aspect(1 / np.cos(np.deg2rad(40)))
    ax.set_facecolor(NODATA)
    ax.set_xlim(s2.LON_MIN - pad, s2.LON_MAX + pad); ax.set_ylim(s2.LAT_MIN - pad, s2.LAT_MAX + pad)
    ax.set_xticks([-55, -50, -45]); ax.set_xticklabels(["55°W", "50°W", "45°W"] if xlab else [])
    ax.set_yticks([-45, -40, -35]); ax.set_yticklabels(["45°S", "40°S", "35°S"] if ylab else [])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    land(ax)


def grid_img(ax, v, **kw):
    return ax.imshow(np.ma.masked_invalid(as_grid(v)), origin="upper", extent=EXT, interpolation="nearest", zorder=1, **kw)


def hpd(vec, q):
    v = vec[:NCELL] / max(vec[:NCELL].sum(), 1e-12)
    order = np.argsort(v)[::-1]
    keep = order[: np.searchsorted(np.cumsum(v[order]), q) + 1]
    m = np.zeros(NCELL); m[keep] = 1
    return as_grid(m)


def yline(ax):
    ax.grid(axis="y", color=GRID, lw=1.0); ax.set_axisbelow(True)


def arrow(ax, pts, color, lw=4.5, z=6, rad=0.0, head=22):
    ax.add_patch(FancyArrowPatch(pts[0], pts[1], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=head,
                                 lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0))


def label(ax, x, y, txt, color=INK, size=14, ha="center", va="center", weight="bold", z=8):
    ax.text(x, y, txt, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=z,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.92))


# ----------------------------------------------------------------------------- load
daily, tr, ids, SIG, ALP = s4.setup()
fit_ids = ids["train"] | ids["val"]
OP = {d: np.load(os.path.join(S5, f"operator_final_{d}.npz"), allow_pickle=True) for d in ("drogued", "undrogued")}
masked, ocean = OP["drogued"]["masked"], OP["drogued"]["ocean"]
cellfate = pd.read_csv(os.path.join(S5, "rq1_cell_fate.csv"))
summ = pd.read_csv(os.path.join(S5, "rq1_summary.csv"))
pos = pd.read_csv(os.path.join(S5, "final_test_position.csv"))
names = list(OP["drogued"]["release_names"])
days = list(OP["drogued"]["show_days"])
REL = LogNorm(1e-4, 1e-1)

# ----------------------------------------------------------------------------- 0. title map (South America): all native shapes
fig, ax = plt.subplots(figsize=(4.9, 6.9))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(-84, -22); ax.set_ylim(-61, 14); ax.set_aspect(1.12); ax.axis("off")
ov = Overlay(fig, "title_map", image=False)
HL = ("Brazil", "Uruguay", "Argentina")
for nm, parts in geo_parts(tol=0.12, min_area=0.25).items():
    ov.poly(ax, parts, "#9FB1C8" if nm in HL else "#DDE2E9", "#FFFFFF", 0.75, nm)
ov.rect(ax, -60, -50, -40, -30, edge=INK, lw=2.25)
ov.text(ax, -39.2, -49.4, "Study region R", size=15, ha="left", va="bottom")
ov.arrow(ax, (-37.0, -14.0), (-51.5, -36.3), ORANGE, lw=4.0, rad=-0.22)
ov.arrow(ax, (-64.0, -53.0), (-55.5, -39.5), PURPLE, lw=4.0, rad=-0.25)
ov.arrow(ax, (-52.5, -38.5), (-40.8, -40.5), INK, lw=3.0)
ov.dot(ax, -54, -38, 11, INK, "FFFFFF", 1.5)
ov.text(ax, -53.5, -7.5, "Brazil", size=16, color="#FFFFFF")
ov.text(ax, -65.0, -36.2, "Argentina", size=13, color="#FFFFFF")
ov.line(ax, (-71.0, -28.3), (-56.0, -32.7), INK, lw=1.25)
ov.text(ax, -71.5, -27.6, "Uruguay", size=14, ha="right", va="center")
ov.text(ax, -35.5, -16.6, "Brazil Current", size=15, color=ORANGE, ha="left", va="center")
ov.text(ax, -35.5, -19.6, "warm, southward", size=13, color=MUTED, bold=False, ha="left", va="center")
ov.text(ax, -62.5, -55.6, "Malvinas Current", size=15, color=PURPLE, ha="left", va="center")
ov.text(ax, -62.5, -58.6, "cold, northward", size=13, color=MUTED, bold=False, ha="left", va="center")
ov.text(ax, -57.2, -44.0, "Confluence", size=14, ha="left", va="center")
ov.export()

# ----------------------------------------------------------------------------- 1. locator: all native shapes
fig, ax = plt.subplots(figsize=(2.6, 3.6))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(-83, -33); ax.set_ylim(-57, 13.5); ax.set_aspect(1.12); ax.axis("off")
ov = Overlay(fig, "locator", image=False)
for nm, parts in geo_parts(tol=0.2, min_area=0.6).items():
    ov.poly(ax, parts, "#C3CBD6", "#F7F8FA", 0.5, nm)
ov.rect(ax, -60, -50, -40, -30, edge=PURPLE, lw=2.5, fill=PURPLE, alpha=0.7)
ov.text(ax, -50, -40, "R", size=20, color=PURPLE)
ov.export()

# ----------------------------------------------------------------------------- 2. motivation map: all native shapes
VIEW = (-68, -52, -38, -25)
fig, ax = plt.subplots(figsize=(6.3, 6.3 * 27 / (30 * np.cos(np.deg2rad(40)))))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(VIEW[0], VIEW[2]); ax.set_ylim(VIEW[1], VIEW[3]); ax.set_aspect(1 / np.cos(np.deg2rad(40))); ax.axis("off")
ov = Overlay(fig, "fig_motivation_map", image=False)
ov.rect(ax, VIEW[0], VIEW[1], VIEW[2], VIEW[3], fill="#DCE9F5", alpha=0.0, edge=None, lw=0)
TINT = {"Argentina": "#C3CCD8", "Uruguay": "#A9B8CA", "Brazil": "#D5DBE3", "Paraguay": "#E3E7EC", "Chile": "#E3E7EC", "Bolivia": "#E3E7EC"}
for nm, parts in geo_parts(clip=VIEW, tol=0.04, min_area=0.02).items():
    ov.poly(ax, parts, TINT.get(nm, "#D5DBE3"), "#FFFFFF", 1.25, nm)
ov.rect(ax, -60, -50, -40, -30, edge=INK, lw=2.0, dash="dash")
ov.text(ax, -40.5, -49.5, "Study region R", size=14, ha="right", va="bottom")
for (lo, la, nm, ha, dx, dy) in [(-58.4, -34.6, "Buenos Aires", "right", -0.6, 0.6), (-56.2, -34.9, "Montevideo", "left", 0.2, 1.15),
                                 (-57.55, -38.0, "Mar del Plata", "right", -0.6, 0)]:
    ov.dot(ax, lo, la, 6, INK, None, 0)
    ov.text(ax, lo + dx, la + dy, nm, size=13, bold=False, ha=ha, va="center", bg="#FFFFFF")
# pins: where each event took place. 1 Punta del Este; 2 where the ARA San Juan was found (45°57'S 59°46'W);
# 3 about 300 km off Mar del Plata (approximate); 4 the mouth of the Rio de la Plata (ports of Buenos Aires and Montevideo)
for n_, lo, la, col in [(1, -54.95, -35.0, ORANGE), (2, -59.77, -45.95, BLUE), (3, -54.4, -38.9, PURPLE), (4, -56.9, -36.3, GREEN)]:
    ov.dot(ax, lo, la, 22, col, "FFFFFF", 2.0, text=str(n_), tsize=13)
ov.text(ax, -51.2, -27.3, "Brazil", size=14, color=MUTED, bold=True)
ov.text(ax, -64.3, -40.5, "Argentina", size=14, color=MUTED, bold=True)
ov.text(ax, -56.0, -32.6, "Uruguay", size=12, color=INK, bold=True)
ov.text(ax, -57.2, -26.3, "Paraguay", size=11, color=MUTED, bold=False)
ov.export()

# ----------------------------------------------------------------------------- 3. region: tracks and mean flow
d = daily.sort_values(["id", "time"])
same = (d["id"].values[1:] == d["id"].values[:-1]) & (np.diff(d["time"].values) == np.timedelta64(1, "D"))
xy = np.column_stack([d["lon"].values, d["lat"].values])
segs = np.stack([xy[:-1], xy[1:]], axis=1)[same]
drg = d["drogued"].values[:-1][same].astype(bool)
fig, a = plt.subplots(figsize=(4.6, 5.9), layout="constrained")
ov = Overlay(fig, "fig_tracks")
for mask, col in [(drg, BLUE), (~drg, ORANGE)]:
    a.add_collection(LineCollection(segs[mask], colors=col, linewidths=0.3, alpha=0.4, zorder=1))
map_ax(a, "", pad=2.0)
ov.title(a, "Every drifter trajectory")
a.set_facecolor(BG)
ov.rect(a, -60, -50, -40, -30, edge=INK, lw=2.25)
a.legend(handles=[Line2D([], [], color=BLUE, lw=4, label="Drogued (follows water at 15 m)"),
                  Line2D([], [], color=ORANGE, lw=4, label="Undrogued (at the surface)")],
         loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1, frameon=False)
ov.export()
m_all = s4.moves_for(tr[tr["id"].isin(fit_ids)], "all")
g = m_all.groupby("from_cell").agg(de=("de_km", "mean"), dn=("dn_km", "mean")).reindex(range(NCELL))
spd = np.array(np.hypot(g["de"], g["dn"]).values, dtype=float)
spd[masked] = np.nan
fig, a = plt.subplots(figsize=(5.6, 5.9), layout="constrained")
ov = Overlay(fig, "fig_meanflow")
im = grid_img(a, spd, cmap="Blues", vmin=0, vmax=np.nanpercentile(spd, 99))
okq = ~masked
a.quiver(CELL_LON[okq], CELL_LAT[okq], g["de"].values[okq], g["dn"].values[okq], color=INK, scale=420, width=0.0035,
         headwidth=3.5, pivot="mid", zorder=4, alpha=0.75)
map_ax(a, "")
ov.title(a, "Mean daily drift per 1° cell")
ov.arrow(a, (-48.3, -30.6), (-53.6, -37.2), ORANGE, lw=5.0, rad=-0.15)
ov.arrow(a, (-58.8, -47.0), (-55.6, -39.6), PURPLE, lw=5.0, rad=-0.25)
ov.arrow(a, (-52.5, -39.3), (-41.5, -40.3), INK, lw=5.0)
ov.text(a, -45.4, -32.0, "Brazil Current", size=14, color=ORANGE, bg="#FFFFFF", lines=["Brazil Current", "warm, southward"])
ov.text(a, -51.6, -46.3, "Malvinas Current", size=14, color=PURPLE, bg="#FFFFFF", lines=["Malvinas Current", "cold, northward"])
ov.text(a, -45.2, -42.4, "Both turn east", size=14, bg="#FFFFFF")
cb = fig.colorbar(im, ax=a, orientation="vertical", shrink=0.8, pad=0.02)
cb.set_label("Mean drift (km per day)"); cb.outline.set_visible(False)
ov.export()

# ----------------------------------------------------------------------------- 4. data support (one map)
r = daily[daily["inR"] & daily["id"].isin(fit_ids)].copy()
r["cell"] = s2.cell_of(r["lat"].values, r["lon"].values)
cnt = r.groupby("cell")["id"].nunique().reindex(range(NCELL)).values.astype(float)
cnt[~ocean] = np.nan
fig, ax = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
ov = Overlay(fig, "fig_data_support")
im = grid_img(ax, cnt, cmap="Blues", norm=LogNorm(3, np.nanmax(cnt)))
mk = masked & ocean
ax.scatter(CELL_LON[mk], CELL_LAT[mk], marker="x", s=55, color=ORANGE, lw=2.2, zorder=5)
map_ax(ax, "")
ov.title(ax, "Distinct drifters per 1° cell")
cb = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.02, ticks=[3, 10, 30, 100]); cb.outline.set_visible(False)
cb.ax.set_yticklabels(["3", "10", "30", "100"]); cb.ax.minorticks_off()
ax.legend(handles=[Line2D([], [], marker="x", color=ORANGE, ls="", ms=10, mew=2.4, label=f"Fewer than {MIN_DRIFTERS} drifters: masked")],
          loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False)
ov.export()


# ----------------------------------------------------------------------------- 5. release forecast
def release_panel(a, kind, ri, t, title, ylab, ov=None):
    z = OP[kind]
    c = int(z["release_cells"][ri])
    snap = z["release_snaps"][ri][days.index(t)]
    gr = snap[:NCELL].copy()
    gr[(gr < 1e-4) | masked] = np.nan
    im = grid_img(a, gr, cmap="Blues", norm=REL)
    a.contour(XC, YC, hpd(snap, 0.5), levels=[0.5], colors=ORANGE, linewidths=2.6, zorder=4)
    a.contour(XC, YC, hpd(snap, 0.9), levels=[0.5], colors=ORANGE, linewidths=1.8, linestyles="--", zorder=4)
    map_ax(a, "", ylab=ylab)
    if ov is not None and title:
        ov.title(a, title.format(pct=snap[:NCELL].sum()), size=16)
    a.plot(CELL_LON[c], CELL_LAT[c], marker="*", color=INK, ms=20, mec="white", mew=1.2, zorder=7)
    return im, snap


fig, ax = plt.subplots(1, 4, figsize=(12.3, 4.5), layout="constrained")
ov = Overlay(fig, "fig_release_confluence")
for j, (t, lab) in enumerate([(7, "1 week"), (30, "1 month"), (90, "3 months"), (365, "1 year")]):
    im, snap = release_panel(ax[j], "drogued", 0, t, lab + ": {pct:.0%} in R", j == 0, ov)
left_e = float(summ[(summ["operator"] == "drogued") & (summ["start"] == names[0]) & (summ["day"] == 365) & (summ["quantity"] == "left via E")]["value"].iloc[0])
ov.arrow(ax[3], (-47.5, -46.8), (-40.6, -46.8), INK, lw=4.5)
ov.text(ax[3], -50.0, -43.9, "", size=14, bg="#FFFFFF", lines=[f"{left_e:.0%} left", "to the east"])
cb = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.85, pad=0.012, ticks=[1e-4, 1e-3, 1e-2, 1e-1]); cb.outline.set_visible(False)
cb.ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%"]); cb.ax.minorticks_off(); cb.set_label("Probability per cell")
fig.legend(handles=[Line2D([], [], marker="*", color=INK, ls="", ms=16, label="Release point"),
                    Line2D([], [], color=ORANGE, lw=2.6, label="Area holding 50%"),
                    Line2D([], [], color=ORANGE, lw=1.8, ls="--", label="Area holding 90%")],
           loc="outside lower center", ncol=3, frameon=False)
ov.export()

# ----------------------------------------------------------------------------- 6. half-time and relative density
cf = cellfate[cellfate["operator"] == "drogued"].reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(9.2, 5.3), layout="constrained")
ov = Overlay(fig, "fig_fate_density")
md = cf["median_days_in_R"].values.copy(); md[masked] = np.nan
im = grid_img(ax[0], md, cmap="viridis", vmin=0, vmax=np.nanmax(md))
map_ax(ax[0], "")
ov.title(ax[0], "Days until half has left R")
cb = fig.colorbar(im, ax=ax[0], orientation="horizontal", shrink=0.9, pad=0.03); cb.outline.set_visible(False)
cb.set_label("Days, by release cell")
ov.arrow(ax[0], (-49.5, -47.8), (-40.6, -47.8), "#FFFFFF", lw=4.5)
ov.text(ax[0], -46.0, -45.5, "Leaves to the east", size=14, bg="#FFFFFF")
a = ax[1]
rho = cf["rho_day90"].values.copy(); rho[masked] = np.nan
im = grid_img(a, rho, cmap="RdBu_r", norm=LogNorm(0.2, 5))
sure = (~masked) & ((cf["rho_day90_p05"] > 1) | (cf["rho_day90_p95"] < 1)).values
a.scatter(CELL_LON[sure], CELL_LAT[sure], s=10, color=INK, lw=0, zorder=4)
map_ax(a, "", ylab=False)
ov.title(a, "Where it gathers (3 months)")
ov.rect(a, -52.55, -42.3, -40.05, -33.7, edge=INK, lw=3.0, ellipse=True)
ov.text(a, -46.3, -32.2, "Gathers here", size=14, bg="#FFFFFF")
cb = fig.colorbar(im, ax=a, orientation="horizontal", shrink=0.9, pad=0.03, ticks=[0.2, 0.5, 1, 2, 5])
cb.ax.minorticks_off(); cb.ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"]); cb.outline.set_visible(False)
cb.set_label("Relative density (blue loses, red gathers)")
ov.export()

# ----------------------------------------------------------------------------- 7. calibration and skill (test)
M4 = "M4 + other drogue type"
fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6), layout="constrained")
x, hz = np.arange(4), [1, 7, 30, 90]
a = ax[0]
for sname, col, off, lab in [("test / drogued", BLUE, -0.06, "At 15 m (drogued)"), ("test / undrogued", ORANGE, 0.06, "At the surface (undrogued)")]:
    q = pos[(pos["set"] == sname) & (pos["model"] == M4)].set_index("horizon_days").loc[hz]
    k = "in_90pct_area_%"
    a.errorbar(x + off, q[k], yerr=[q[k] - q[k + "_p05"], q[k + "_p95"] - q[k]], color=col, marker="o", ms=10, lw=2.6, capsize=4, label=lab)
a.axhline(90, color=INK, ls="--", lw=1.8)
a.set_yticks([70, 75, 80, 85, 90, 95, 100]); a.set_yticklabels([f"{v}%" for v in (70, 75, 80, 85, 90, 95, 100)])
a.text(-0.22, 89.6, "target", ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
a.set_xlim(-0.3, 3.35); a.set_ylim(70, 100); a.set_xticks(x); a.set_xticklabels(["1 day", "1 week", "1 month", "3 months"])
a.set_title("Share of test drifters inside the 90% area", loc="left"); a.legend(frameon=False, loc="lower left"); yline(a)
a = ax[1]
q = pos[(pos["set"] == "test / drogued") & (pos["model"] == M4)].set_index("horizon_days").loc[hz]
qu = pos[(pos["set"] == "test / undrogued") & (pos["model"] == M4)].set_index("horizon_days").loc[hz]
a.plot(x, q["median_persistence_error_km"], marker="o", ms=10, lw=3, color=GREY)
a.plot(x, q["median_centre_error_km"], marker="o", ms=10, lw=3, color=BLUE)
a.plot(x, qu["median_centre_error_km"], marker="o", ms=10, lw=3, color=ORANGE)
a.text(2.92, q["median_persistence_error_km"].iloc[-1] + 12, "Assuming no movement", color=MUTED, va="bottom", ha="right", fontsize=14, fontweight="bold")
a.text(-0.2, 500, "M4, 15 m", color=BLUE, va="top", ha="left", fontsize=14, fontweight="bold")
a.text(-0.2, 445, "M4, surface", color=ORANGE, va="top", ha="left", fontsize=14, fontweight="bold")
a.set_xlim(-0.3, 3.35); a.set_ylim(0, 580); a.set_xticks(x); a.set_xticklabels(["1 day", "1 week", "1 month", "3 months"])
a.set_yticks([0, 100, 200, 300, 400, 500]); a.set_yticklabels([f"{v} km" for v in (0, 100, 200, 300, 400, 500)])
a.set_title("Median distance error", loc="left"); yline(a)
save(fig, "fig_test_calibration.png")

# ----------------------------------------------------------------------------- 8. the model ladder (test)
allm = pd.read_csv(os.path.join(S12, "complexity_test_pooled.csv")).query("set == 'test / all'")
ROWS = [("M0b displacement", "M0  One pattern everywhere"), ("M1 empirical", "M1  Raw counts"),
        ("M5 GAM", "M5  GAM"), ("M6 neural net", "M6  Neural network"),
        ("M7 memory, counts", "M7  Counts + memory"), ("M7 memory, GAM", "M7  GAM + memory"), ("M7 memory, network", "M7  Network + memory")]
fig, ax = plt.subplots(1, 3, figsize=(12.3, 4.9), layout="constrained", sharey=True)
fig.get_layout_engine().set(wspace=0.12)
for a, k in zip(ax, (7, 30, 90)):
    a.axhspan(3.5, 6.5, color="#ECE3EE", zorder=0, lw=0)
    for i, (mname, _) in enumerate(ROWS):
        r_ = allm[(allm["model"] == mname) & (allm["horizon_days"] == k)].iloc[0]
        col = GREEN if r_["gain_p05"] > 0 else (ORANGE if r_["gain_p95"] < 0 else GREY)
        lo, hi, v = max(r_["gain_p05"], -0.5), r_["gain_p95"], r_["gain_vs_M2"]
        if v < -0.5:
            a.annotate("", xy=(-0.49, i), xytext=(-0.36, i), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3.5))
            a.text(-0.34, i, f"{v:.2f}", color=ORANGE, va="center", ha="left", fontsize=14, fontweight="bold")
        else:
            a.plot([lo, hi], [i, i], color=col, lw=5, solid_capstyle="round", zorder=3)
            a.plot(v, i, "o", color=col, ms=13, mec="white", mew=1.6, zorder=4)
    a.axvline(0, color=PURPLE, lw=2.6, zorder=2)
    a.set_title({7: "1 week ahead", 30: "1 month ahead", 90: "3 months ahead"}[k], loc="center", pad=56)
    a.set_xlim(-0.5, 0.2); a.set_ylim(len(ROWS) - 0.5, -0.5)
    a.text(-0.22, 1.085, "WORSE", transform=a.get_xaxis_transform(), color="white", ha="center", va="center", fontsize=13.5, fontweight="bold",
           bbox=dict(boxstyle="larrow,pad=0.3", fc=ORANGE, ec=ORANGE))
    a.text(0.085, 1.085, "BETTER", transform=a.get_xaxis_transform(), color="white", ha="center", va="center", fontsize=13.5, fontweight="bold",
           bbox=dict(boxstyle="rarrow,pad=0.3", fc=GREEN, ec=GREEN))
    a.grid(axis="x", color=GRID, lw=1.0); a.set_axisbelow(True)
    a.set_yticks(range(len(ROWS))); a.set_yticklabels([lab for _, lab in ROWS], color=INK, fontsize=15)
    a.tick_params(axis="y", length=0); a.set_xticks([-0.4, -0.2, 0, 0.2])
    a.set_xlabel("Log score − M2's")
ax[0].text(0.014, -0.42, "M2", color=PURPLE, fontweight="bold", fontsize=14, ha="left", va="top")
save(fig, "fig_ladder_test.png")

# ----------------------------------------------------------------------------- 9. RQ3 grid size: three maps (numbers go on the slide)
trn_ids = ids["train"]
rt = daily[daily["inR"] & daily["id"].isin(trn_ids)]
g3 = pd.read_csv(os.path.join(S12, "grid_pooled.csv"))
GRIDNUM = {}
fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.1), layout="constrained")
ov = Overlay(fig, "fig_rq3_grid")
for a, deg, gname in zip(ax, (0.5, 1.0, 2.0), ("square 0.5°", "square 1°", "square 2°")):
    n = int(round(20 / deg))
    ix = np.clip(np.floor((rt["lon"].values - s2.LON_MIN) / deg).astype(int), 0, n - 1)
    iy = np.clip(np.floor((s2.LAT_MAX - rt["lat"].values) / deg).astype(int), 0, n - 1)
    c = pd.DataFrame({"c": iy * n + ix, "id": rt["id"].values}).groupby("c")["id"].nunique().reindex(range(n * n)).values.astype(float)
    okc = np.where(np.isnan(c), np.nan, (c >= MIN_DRIFTERS).astype(float))
    a.imshow(np.ma.masked_invalid(okc.reshape(n, n)), origin="upper", extent=EXT, interpolation="nearest", zorder=1,
             cmap=matplotlib.colors.ListedColormap(["#F2B36B", BLUE]), vmin=0, vmax=1)
    for v in np.arange(-60, -39.99, deg):
        a.axvline(v, color="white", lw=0.35 if deg < 1 else 0.6, zorder=2); a.axhline(v + 10, color="white", lw=0.35 if deg < 1 else 0.6, zorder=2)
    row = g3[g3["grid"] == gname]
    GRIDNUM[f"{deg:g}"] = {"usable": float(row["usable_%"].iloc[0]), "cells": n * n,
                           "cover30": float(row[row["horizon_days"] == 30]["in_90pct_area_%"].iloc[0])}
    map_ax(a, "", ylab=(deg == 0.5))
    ov.title(a, f"{deg:g}° cells ({n * n:,})", size=17)
ax[1].legend(handles=[Line2D([], [], marker="s", color=BLUE, ls="", ms=14, label=f"{MIN_DRIFTERS} or more drifters"),
                      Line2D([], [], marker="s", color="#F2B36B", ls="", ms=14, label=f"1 to {MIN_DRIFTERS - 1}"),
                      Line2D([], [], marker="s", color=NODATA, mec="#B8C2CE", ls="", ms=14, label="none (shelf or land)")],
             loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, handletextpad=0.2, columnspacing=1.2)
ov.export()

# ----------------------------------------------------------------------------- 9b. backup: Markov check
mk = pd.read_csv(os.path.join(S12, "timestep_markov_pooled.csv"))
mk = mk[mk["step"] == "direct"]
fig, a = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
a.plot(mk["horizon_days"], mk["tv_split_half_reference"], marker="o", ms=10, lw=3, color=GREY)
a.plot(mk["horizon_days"], mk["tv_iterated_vs_direct"], marker="o", ms=10, lw=3, color=BLUE)
a.text(91, 0.78, "Two random halves of the data\n(what sampling noise alone gives)", color=MUTED, ha="right", va="bottom", fontsize=14, fontweight="bold")
a.text(91, 0.33, "One-day matrix applied repeatedly\nvs matrix fitted to the long move", color=BLUE, ha="right", va="top", fontsize=14, fontweight="bold")
a.set_ylim(0, 1); a.set_xticks([6, 12, 30, 60, 90]); a.set_xlabel("Horizon (days)"); a.set_ylabel("Difference between forecasts (0 to 1)"); yline(a)
save(fig, "fig_markov.png")

# ----------------------------------------------------------------------------- 9c. numbers for the method illustration (real operator)
z = OP["drogued"]; Pm = z["P"]; c0 = int(z["release_cells"][0])
iy0, ix0 = divmod(c0, NX)
demo = {"row": [[float(Pm[c0, (iy0 + dy) * NX + ix0 + dx]) for dx in (-1, 0, 1)] for dy in (-1, 0, 1)], "days": {}}
pvec = np.zeros(Pm.shape[0]); pvec[c0] = 1.0
win = [(iy0 + dy, ix0 + 1 + dx) for dy in range(-4, 5) for dx in range(-4, 5)]
for t in range(0, 8):
    if t in (0, 1, 7):
        demo["days"][str(t)] = [float(pvec[y * NX + x]) if (0 <= y < NY and 0 <= x < NX) else None for (y, x) in win]
    pvec = pvec @ Pm
demo["land"] = [bool(masked[y * NX + x] and not ocean[y * NX + x]) if (0 <= y < NY and 0 <= x < NX) else True for (y, x) in win]
demo["grid"] = GRIDNUM
with open(os.path.join(FIG, "numbers_slides.json"), "w") as f:
    json.dump(demo, f)

# ----------------------------------------------------------------------------- 10. drogued vs undrogued
fig, ax = plt.subplots(1, 3, figsize=(12.3, 4.7), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1, 1]})
ov = Overlay(fig, "fig_drogue")
a = ax[0]
dd = np.arange(366)
hd = summ[(summ["quantity"] == "days until half has left")].set_index("operator")["value"]
for kind, col in (("drogued", BLUE), ("undrogued", ORANGE)):
    z = OP[kind]
    a.fill_between(dd, 100 * z["even_curve_p05"], 100 * z["even_curve_p95"], color=col, alpha=0.25, lw=0)
    a.plot(dd, 100 * z["even_curve"], color=col, lw=3.2)
    a.plot([hd[kind]], [50], "o", color=col, ms=12, mec="white", mew=1.6, zorder=5)
a.axhline(50, color=MUTED, ls="--", lw=1.4)
a.text(128, 66, f"At 15 m:\nhalf gone by day {hd['drogued']:.0f}", color=BLUE, fontweight="bold", fontsize=14.5)
a.text(8, 2, f"At the surface:\nhalf gone by day {hd['undrogued']:.0f}", color=ORANGE, fontweight="bold", fontsize=14.5)
a.set_xlim(0, 365); a.set_ylim(0, 100); a.set_xticks([0, 90, 180, 270, 365])
a.set_title("Share still in R", loc="left"); a.set_xlabel("Days after release"); a.set_ylabel("%"); yline(a)
for a, kind, nm in zip(ax[1:], ("drogued", "undrogued"), ("At 15 m", "At the surface")):
    im, snap = release_panel(a, kind, 0, 90, nm + ", 3 months: {pct:.0%}", kind == "drogued", ov)
ov.export()
with open(os.path.join(FIG, "overlays.json"), "w") as f:
    json.dump(OVERLAYS, f)
print("All figures written to", FIG, "| overlays:", ", ".join(OVERLAYS))
