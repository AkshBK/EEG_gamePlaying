"""ERPs to events during continuous video game play (OpenNeuro ds003517, run-02).

Same cleaning and averaging as erp_oddball_gambling.py:
    1-3. load, filter 0.1-40 Hz + average reference, ICA (ICLabel eye/heart/muscle removed)
    4.   epochs -0.5..1.0 s, baseline -200..0 ms, drop epochs > 200 µV peak-to-peak
    5.   average the epochs of each event type within each subject
    6.   grand average = mean of the subjects' ERPs (every subject counts equally)

Game events (the six the authors analysed, at Cz):
    S 90 shoot, S 91 crash into wall, S 92 crash into enemy,
    S 93 collect star, S 94 collect ammo, S 97 missile hits enemy
Comparison ("null"): 500 random moments of gameplay per subject, treated the same way.
Effect = event − null, mean of 300-500 ms at Cz.

Usage:
    python erp_gameplay.py <data_root> <out_dir> sub-001 ... sub-017 [--cache DIR]
--cache DIR keeps each subject's averages so the figure can be redrawn without rerunning ICA.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy import stats

from erp_oddball_gambling import clean_raw, epoch, REJECT_PTP, C_MAIN, MUTED, BAND

GAME_EVENTS = {"S 90": 90, "S 91": 91, "S 92": 92, "S 93": 93, "S 94": 94, "S 97": 97}
GAME_NAMES = {"shoot": 90, "wall": 91, "enemy": 92, "star": 93, "ammo": 94, "hit": 97}
TITLE = {"shoot": "Shoot (button press)", "hit": "Missile hits enemy", "star": "Collect star",
         "ammo": "Collect ammo", "wall": "Crash into wall", "enemy": "Crash into enemy"}
ORDER = ["shoot", "hit", "star", "ammo", "wall", "enemy"]
CH, WIN, N_NULL = "Cz", (0.30, 0.50), 500
C_NULL = "#9a9a9a"


def null_epochs(data, ev_samples, seed=42):
    """Random moments between the first and last game event, epoched like the real events."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(ev_samples.min(), ev_samples.max(), N_NULL))
    ev = np.column_stack([t, np.zeros(N_NULL, int), np.ones(N_NULL, int)])
    ep = mne.Epochs(data, ev, {"null": 1}, tmin=-0.5, tmax=1.0, baseline=(-0.2, 0.0), picks="eeg",
                    preload=True, reject_by_annotation=False)
    return ep[np.where(np.ptp(ep.get_data(), axis=2).max(1) <= REJECT_PTP)[0]]


def subject_erps(root, sub, cache):
    path = os.path.join(cache, f"{sub}_game-ave.fif") if cache else None
    if path and os.path.exists(path):
        evs = mne.read_evokeds(path)
        return {e.comment: e for e in evs}, "cached"
    data, n_rm, n_ic = clean_raw(root, sub, "02")
    ep = epoch(data, GAME_EVENTS, GAME_NAMES)
    ev, _ = mne.events_from_annotations(data, event_id=GAME_EVENTS)
    erps = {k: ep[k].average() for k in ORDER if k in ep.event_id}
    erps["null"] = null_epochs(data, ev[:, 0]).average()
    for k, e in erps.items():
        e.comment = k
    if path:
        mne.write_evokeds(path, list(erps.values()), overwrite=True)
    return erps, f"ICA removed {n_rm} of {n_ic} components"


def figure(erps, out):
    subs = list(erps)
    n = len(subs)
    uv = lambda e: e.copy().pick([CH]).data[0] * 1e6
    t = erps[subs[0]]["null"].times * 1000
    sel = (t >= -200) & (t <= 800)
    w = (t >= WIN[0] * 1000) & (t <= WIN[1] * 1000)
    ga_null = mne.grand_average([erps[s]["null"] for s in subs])

    res, diffs = {}, {}
    for k in ORDER:
        have = [s for s in subs if k in erps[s]]
        d = [mne.combine_evoked([erps[s][k], erps[s]["null"]], [1, -1]) for s in have]
        eff = np.array([uv(x)[w].mean() for x in d])
        res[k] = dict(n=len(have), trials=np.mean([erps[s][k].nave for s in have]), eff=eff,
                      t=stats.ttest_1samp(eff, 0), ga=mne.grand_average([erps[s][k] for s in have]),
                      se=np.std([uv(erps[s][k]) for s in have], axis=0, ddof=1) / np.sqrt(len(have)))
        diffs[k] = mne.grand_average(d)

    fig = plt.figure(figsize=(17, 10.5), dpi=150)
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.85], hspace=0.6, wspace=0.35)
    allv = [(uv(res[k]["ga"]) + sg * res[k]["se"])[sel] for k in ORDER for sg in (-1, 1)] + [uv(ga_null)[sel]]
    lo, hi = min(map(np.min, allv)), max(map(np.max, allv))
    ylim = (lo - 0.15 * (hi - lo), hi + 0.1 * (hi - lo))
    for i, k in enumerate(ORDER):
        r = res[k]
        ax = fig.add_subplot(gs[i // 3, (i % 3) * 2:(i % 3) * 2 + 2])
        g = uv(r["ga"])
        ax.axvspan(WIN[0] * 1000, WIN[1] * 1000, color=BAND, zorder=0)
        ax.fill_between(t[sel], (g - r["se"])[sel], (g + r["se"])[sel], color=C_MAIN, alpha=0.18, lw=0)
        ax.plot(t[sel], uv(ga_null)[sel], color=C_NULL, lw=1.4, label="random moments (null)")
        ax.plot(t[sel], g[sel], color=C_MAIN, lw=2.0, label=f"{k} (± SE across subjects)")
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6, ls="--")
        ax.set(xlim=(-200, 800), ylim=ylim)
        p = r["t"].pvalue
        ax.set_title(f"{TITLE[k]}: {r['eff'].mean():+.1f} µV vs null, {(r['eff'] > 0).sum()}/{r['n']} subjects > 0"
                     f"\n{r['n']} subjects, {r['trials']:.0f} events each on average, "
                     f"p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}", fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel(f"µV at {CH}")
        if i >= 3:
            ax.set_xlabel("Time after the game event (ms)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    maps = {k: diffs[k].copy().crop(*WIN).data.mean(1) * 1e6 for k in ORDER}
    lim = max(np.abs(m).max() for m in maps.values())
    for i, k in enumerate(ORDER):
        ax = fig.add_subplot(gs[2, i])
        im, _ = mne.viz.plot_topomap(maps[k], diffs[k].info, axes=ax, show=False, cmap="RdBu_r",
                                     vlim=(-lim, lim), contours=6, sensors=False,
                                     mask=np.array([c == CH for c in diffs[k].ch_names]),
                                     mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k", markersize=6))
        ax.set_title(f"{k} − null", fontsize=10)
    cax = fig.add_axes([0.92, 0.08, 0.008, 0.18])
    fig.colorbar(im, cax=cax).set_label("µV")
    fig.text(0.5, 0.31, f"Scalp maps: event − null, {WIN[0] * 1000:.0f}–{WIN[1] * 1000:.0f} ms "
             f"(white dot = {CH}, same colour scale for all)", ha="center", fontsize=11)
    fig.suptitle(f"Video game play: which events stand out from ongoing play? "
                 f"(grand average of {n} subjects at {CH}, filter 0.1–40 Hz + ICA)", fontsize=14, y=0.97)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_root"); p.add_argument("out_dir"); p.add_argument("subjects", nargs="+")
    p.add_argument("--cache", default=None, help="folder for each subject's averages")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    erps, lines = {}, []
    for sub in args.subjects:
        erps[sub], note = subject_erps(args.data_root, sub, args.cache)
        counts = ", ".join(f"{k} {erps[sub][k].nave}" for k in ORDER + ["null"] if k in erps[sub])
        lines.append(f"{sub}: {note}; epochs kept: {counts}")
        print(lines[-1], flush=True)

    res = figure(erps, os.path.join(args.out_dir, f"3_gameplay_events_{CH}.png"))
    lines.append(f"\nevent − null at {CH}, {WIN[0] * 1000:.0f}-{WIN[1] * 1000:.0f} ms:")
    for k in ORDER:
        r = res[k]
        lines.append(f"  {k:6s} {r['eff'].mean():+5.1f} µV, {(r['eff'] > 0).sum()}/{r['n']} subjects > 0, "
                     f"t({r['n'] - 1}) = {r['t'].statistic:.1f}, p = {r['t'].pvalue:.2g}; per subject: "
                     + " ".join(f"{x:+.1f}" for x in r["eff"]))
        print(lines[-1])
    with open(os.path.join(args.out_dir, "summary_gameplay.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
