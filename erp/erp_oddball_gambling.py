"""ERPs for the oddball and gambling tasks (OpenNeuro ds003517, run-01), with ICA.

Per subject:
    1. load, add back the reference channel CPz, drop FT9 FT10 TP9 TP10
    2. band-pass 0.1-40 Hz on the continuous recording, average reference
    3. ICA (fitted on a 1-100 Hz copy); remove components ICLabel calls
       eye or heart (> 0.8) or muscle (> 0.95)
    4. epochs -0.5..1.0 s, baseline -200..0 ms, drop epochs > 200 µV peak-to-peak
    5. AVERAGE the trials of each condition -> one ERP per condition per subject
Across subjects:
    6. GRAND AVERAGE = mean of the subjects' ERPs (every subject counts equally)

Figures:
    1_oddball_P3_Pz.png        rare vs standard at Pz, each subject + grand average
    2_gambling_RewP_FCz.png    win vs loss at FCz, each subject + grand average

Usage:
    python erp_oddball_gambling.py <data_root> <out_dir> sub-001 sub-002 sub-003 sub-004 [--shift-ms 40]
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

mne.set_log_level("error")

EVENTS = {"S 22": 1, "S 25": 2, "S 51": 3, "S 52": 4}
NAMES = {"standard": 1, "rare": 2, "loss": 3, "win": 4}
VENTRAL = ["FT9", "FT10", "TP9", "TP10"]
REJECT_PTP = 200e-6
ICA_KW = dict(n_components=0.99, method="picard", random_state=42, max_iter=1000,
              fit_params=dict(ortho=False, extended=True))
IC_RULE = {"eye blink": 0.80, "heart beat": 0.80, "muscle artifact": 0.95}

# task: (title, condition of interest, comparison, channel, window, component)
TASKS = {
    "oddball": ("Oddball", "rare", "standard", "Pz", (0.30, 0.50), "P3"),
    "gambling": ("Gambling feedback", "win", "loss", "FCz", (0.22, 0.32), "reward positivity"),
}
C_MAIN, C_REF, C_DIFF = "#eb6834", "#2a78d6", "#1d2230"
MUTED, BAND = "#5d6270", "#e4e2d9"


def preprocess(root, sub, shift_ms):
    f = glob.glob(os.path.join(root, sub, "eeg", f"{sub}_*run-01_eeg.set"))[0]
    raw = mne.io.read_raw_eeglab(f, preload=True)
    raw.set_channel_types({"VEOG": "eog", "HEOG": "eog"})
    raw = mne.add_reference_channels(raw, "CPz")
    raw.set_montage("standard_1005", on_missing="warn")
    raw.drop_channels(VENTRAL)

    data = raw.copy().filter(0.1, 40.0, picks="eeg").set_eeg_reference("average")

    ic = raw.copy().filter(1.0, 100.0, picks="eeg").notch_filter([60, 120], picks="eeg")
    ic.set_eeg_reference("average")
    seg = mne.make_fixed_length_epochs(ic, duration=1.0, preload=True)
    seg.drop_bad(reject=dict(eeg=500e-6))                # gross junk only; blinks stay in
    ica = ICA(**ICA_KW).fit(seg, picks="eeg")
    lab = label_components(seg, ica, method="iclabel")
    ica.exclude = [i for i, (l, p) in enumerate(zip(lab["labels"], lab["y_pred_proba"]))
                   if l in IC_RULE and p > IC_RULE[l]]
    ica.apply(data)

    ev, _ = mne.events_from_annotations(data, event_id=EVENTS)
    ev[:, 0] += int(round(shift_ms / 1000 * data.info["sfreq"]))
    ids = {k: v for k, v in NAMES.items() if v in ev[:, 2]}
    ep = mne.Epochs(data, ev, ids, tmin=-0.5, tmax=1.0, baseline=(-0.2, 0.0), picks="eeg",
                    preload=True, reject_by_annotation=False)
    keep = np.where(np.ptp(ep.get_data(), axis=2).max(1) <= REJECT_PTP)[0]
    return ep[keep], len(ica.exclude), ica.n_components_


def figure(task, erps, out):
    """erps: {sub: {cond: Evoked}} for the subjects that have this task."""
    title, a, b, ch, win, comp = TASKS[task]
    subs = list(erps)
    n = len(subs)
    diffs = [mne.combine_evoked([erps[s][a], erps[s][b]], [1, -1]) for s in subs]
    ga = {c: mne.grand_average([erps[s][c] for s in subs]) for c in (a, b)}
    ga_d = mne.grand_average(diffs)
    t = ga_d.times * 1000
    sel = (t >= -200) & (t <= 800)
    uv = lambda ev: ev.copy().pick([ch]).data[0] * 1e6
    effect = lambda ev: uv(ev)[(t >= win[0] * 1000) & (t <= win[1] * 1000)].mean()

    fig = plt.figure(figsize=(17, 9), dpi=150)
    gs = fig.add_gridspec(2, 12, height_ratios=[1, 1.35], hspace=0.45, wspace=1.2)
    w_sub = 12 // n                                   # top row: subjects share the full width
    lims = [min(min(uv(erps[s][c]).min() for s in subs for c in (a, b)), -2),
            max(max(uv(erps[s][c]).max() for s in subs for c in (a, b)), 2)]
    pad = 0.08 * (lims[1] - lims[0])
    ylim = (lims[0] - pad, lims[1] + pad)

    def decorate(ax):
        ax.axvspan(win[0] * 1000, win[1] * 1000, color=BAND, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6, ls="--")
        ax.set_xlim(-200, 800)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # top row: one panel per subject
    for i, s in enumerate(subs):
        ax = fig.add_subplot(gs[0, i * w_sub:(i + 1) * w_sub])
        ax.plot(t[sel], uv(erps[s][b])[sel], color=C_REF, lw=1.3, label=f"{b} ({erps[s][b].nave} trials)")
        ax.plot(t[sel], uv(erps[s][a])[sel], color=C_MAIN, lw=1.6, label=f"{a} ({erps[s][a].nave} trials)")
        decorate(ax)
        ax.set(title=f"{s}: {a} − {b} = {effect(diffs[i]):+.1f} µV", xlabel="Time (ms)", ylim=ylim)
        if i == 0:
            ax.set_ylabel(f"µV at {ch}")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    # bottom row: grand average + scalp map
    ax = fig.add_subplot(gs[1, :9])
    se = np.std([uv(d) for d in diffs], axis=0, ddof=1) / np.sqrt(n) if n > 1 else 0 * t
    ax.plot(t[sel], uv(ga[b])[sel], color=C_REF, lw=2, label=b)
    ax.plot(t[sel], uv(ga[a])[sel], color=C_MAIN, lw=2.4, label=a)
    ax.fill_between(t[sel], (uv(ga_d) - se)[sel], (uv(ga_d) + se)[sel], color=C_DIFF, alpha=0.12, lw=0)
    ax.plot(t[sel], uv(ga_d)[sel], color=C_DIFF, lw=1.6, ls="--", label=f"difference {a} − {b} (± SE across subjects)")
    decorate(ax)
    ax.text(np.mean(win) * 1000, ax.get_ylim()[0], f"{comp} window\n{win[0] * 1000:.0f}–{win[1] * 1000:.0f} ms",
            ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set(title=f"Grand average of {n} subjects at {ch}: {a} − {b} = {effect(ga_d):+.1f} µV",
           xlabel=f"Time after the {'picture' if task == 'oddball' else 'feedback'} (ms)", ylabel=f"µV at {ch}")
    ax.legend(frameon=False, fontsize=9, loc="upper left" if task == "oddball" else "lower left")

    ax = fig.add_subplot(gs[1, 9:])
    m = ga_d.copy().crop(*win).data.mean(1) * 1e6
    lim = np.abs(m).max()
    im, _ = mne.viz.plot_topomap(m, ga_d.info, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim),
                                 contours=6, sensors=True, mask=np.array([c == ch for c in ga_d.ch_names]),
                                 mask_params=dict(marker="o", markerfacecolor="w", markeredgecolor="k", markersize=7))
    fig.colorbar(im, ax=ax, shrink=0.7).set_label("µV")
    ax.set_title(f"{a} − {b}, {win[0] * 1000:.0f}–{win[1] * 1000:.0f} ms\n(white dot = {ch})", fontsize=10)

    fig.suptitle(f"{title}: does the {comp} stand out? (filter 0.1–40 Hz + ICA)", fontsize=13, y=0.995)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return {s: effect(d) for s, d in zip(subs, diffs)}, effect(ga_d)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_root"); p.add_argument("out_dir"); p.add_argument("subjects", nargs="+")
    p.add_argument("--shift-ms", type=float, default=0.0, help="move events later (40 = screen delay)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    erps = {task: {} for task in TASKS}
    lines = []
    for sub in args.subjects:
        ep, n_rm, n_ic = preprocess(args.data_root, sub, args.shift_ms)
        counts = ", ".join(f"{k} {len(ep[k])}" for k in ep.event_id)
        lines.append(f"{sub}: ICA removed {n_rm} of {n_ic} components; trials kept: {counts}")
        print(lines[-1], flush=True)
        for task, (_, a, b, *_rest) in TASKS.items():
            if a in ep.event_id and b in ep.event_id:
                erps[task][sub] = {c: ep[c].average() for c in (a, b)}     # step 5

    for i, task in enumerate(TASKS, 1):
        comp, ch = TASKS[task][5], TASKS[task][3]
        fname = f"{i}_{task}_{'P3' if task == 'oddball' else 'RewP'}_{ch}.png"
        per_sub, ga = figure(task, erps[task], os.path.join(args.out_dir, fname))
        lines.append(f"{task} {comp} at {ch}: " + ", ".join(f"{s} {v:+.1f}" for s, v in per_sub.items())
                     + f"; grand average {ga:+.1f} µV")
        print(lines[-1])
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
