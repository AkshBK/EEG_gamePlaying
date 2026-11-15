"""Our FILTERING step only (no ICA), shown with and without a surface Laplacian.

Dataset : OpenNeuro ds003517 (Cavanagh & Castellanos 2016), run-01 =
          oddball (standard / rare pictures) + gambling feedback (win / loss).
Pipeline (nothing else is applied - NO ICA, no bad-channel repair, no autoreject):
    1. load the BrainVision/EEGLAB file, put back the online reference CPz
    2. drop the 4 low edge channels FT9, FT10, TP9, TP10
    3. band-pass 0.1-40 Hz on the CONTINUOUS recording
       (MNE default: zero-phase FIR, Hamming window -> no time shift of the ERPs)
    4. average reference
    5a. "without Laplacian" = the voltage after step 4 (µV)
    5b. "with Laplacian"    = surface Laplacian / current source density of
                              step 4 (spherical splines, Perrin et al. 1989; µV/m²)
    6. epochs -0.5..1.0 s, baseline -200..0 ms; because there is no ICA, an epoch is
       dropped if any EEG channel exceeds 200 µV peak-to-peak (mostly blinks).
       The SAME trials are used for voltage and Laplacian.

Figures written to <out_dir>:
    1_filter_effect.png             what the 0.1-40 Hz filter does (response, spectrum, trace)
    2_continuous_<sub>.png          10 s of continuous EEG: voltage vs Laplacian + scalp maps
    3_erp_voltage_vs_laplacian.png  P3 (rare - standard) and RewP (win - loss), grand average

Usage:
    python filter_laplacian.py <data_root> <out_dir> sub-001 sub-002 sub-003 sub-004 \
        [--window-sub sub-004] [--shift-ms 0]
<data_root> is the folder that contains sub-001/eeg/... from OpenNeuro ds003517.
--shift-ms 40 moves every event 40 ms later (the screen draws the picture 40 ms after
the trigger; the authors correct this in their code). Default 0 = times as in the files.
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("error")

EVENTS = {"S 22": 1, "S 25": 2, "S 51": 3, "S 52": 4}            # trigger codes in run-01
NAMES = {"standard": 1, "rare": 2, "loss": 3, "win": 4}
VENTRAL = ["FT9", "FT10", "TP9", "TP10"]                          # low edge sites, often noisy
L_FREQ, H_FREQ = 0.1, 40.0
TMIN, TMAX, BASE = -0.5, 1.0, (-0.2, 0.0)
REJECT_PTP = 200e-6
CONTRASTS = [("P3: rare − standard", "rare", "standard", "Pz", (0.30, 0.50)),
             ("RewP: win − loss", "win", "loss", "FCz", (0.22, 0.32))]
TRACE_CHANS = ["Fp1", "AFz", "Fp2", "F3", "Fz", "F4", "FCz", "C3", "Cz", "C4",
               "CP3", "CP4", "P3", "Pz", "P4", "PO7", "POz", "PO8", "O1", "Oz", "O2"]
INK, MUTED, GREY, BLUE, ORANGE, RED, BAND = ("#1d2230", "#5d6270", "#9a9a9a", "#2a78d6",
                                              "#eb6834", "#c0392b", "#e4e2d9")


# ----------------------------------------------------------------------------- steps
def load(root, sub):
    """Step 1-2: raw run-01, CPz restored, standard 10-05 positions, edge sites dropped."""
    f = glob.glob(os.path.join(root, sub, "eeg", f"{sub}_*run-01_eeg.set"))
    if not f:
        raise FileNotFoundError(f"no run-01 .set file for {sub} under {root}")
    raw = mne.io.read_raw_eeglab(f[0], preload=True)
    raw.set_channel_types({"VEOG": "eog", "HEOG": "eog"})
    raw = mne.add_reference_channels(raw, "CPz")          # online reference -> a real channel
    raw.set_montage("standard_1005", on_missing="warn")   # positions needed for the Laplacian
    return raw.drop_channels(VENTRAL)


def filter_ours(raw):
    """Step 3-4: continuous 0.1-40 Hz zero-phase FIR, then average reference."""
    raw = raw.copy().filter(L_FREQ, H_FREQ, picks="eeg")
    return raw.set_eeg_reference("average")


def laplacian(raw):
    """Step 5b: surface Laplacian (current source density)."""
    return mne.preprocessing.compute_current_source_density(raw)


def events(raw, shift_ms):
    ev, _ = mne.events_from_annotations(raw, event_id=EVENTS)
    ev[:, 0] += int(round(shift_ms / 1000 * raw.info["sfreq"]))
    return ev, {k: v for k, v in NAMES.items() if v in ev[:, 2]}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ----------------------------------------------------------------------------- figure 1
def fig_filter_effect(raw, filt, out):
    sf = raw.info["sfreq"]
    h = mne.filter.create_filter(np.zeros(int(sf * 200)), sf, L_FREQ, H_FREQ)
    H = np.abs(np.fft.rfft(h, 2 ** 20)); f = np.fft.rfftfreq(2 ** 20, 1 / sf)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), dpi=140)
    ax = axes[0]
    ax.semilogx(f[1:], 20 * np.log10(np.maximum(H[1:], 1e-6)), color=BLUE, lw=1.8)
    for x in (L_FREQ, H_FREQ):
        ax.axvline(x, color=MUTED, lw=0.8, ls="--")
    ax.text(0.1, -75, " 0.1 Hz", fontsize=8, color=MUTED); ax.text(40, -75, " 40 Hz", fontsize=8, color=MUTED)
    ax.set(title=f"A  The filter: what it keeps (0 dB) and removes\nFIR, zero-phase, {len(h) / sf:.0f}-s kernel",
           xlabel="Frequency (Hz, log)", ylabel="Gain (dB)", xlim=(0.01, 250), ylim=(-80, 5))

    raw_avg = raw.copy().set_eeg_reference("average")      # same reference as `filt`: only the filter differs
    ax = axes[1]
    for r, col, lab in ((raw_avg, GREY, "before (raw, average ref)"), (filt, BLUE, "after 0.1–40 Hz")):
        p = r.compute_psd(method="welch", picks=["Pz"], fmin=0.05, fmax=120, n_fft=int(sf * 20))
        ax.semilogx(p.freqs, 10 * np.log10(p.get_data()[0] * 1e12), color=col, lw=1.3, label=lab)
    ax.set(title="B  Power spectrum at Pz, whole run\n(drift below 0.1 Hz and 60 Hz mains are gone)",
           xlabel="Frequency (Hz, log)", ylabel="Power (dB re 1 µV²/Hz)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    a, b = int(60 * sf), int(80 * sf)
    t = np.arange(a, b) / sf
    for r, col, lab in ((raw_avg, GREY, "before (raw, average ref)"), (filt, BLUE, "after 0.1–40 Hz")):
        y = r.get_data(picks=["Pz"], start=a, stop=b)[0] * 1e6
        ax.plot(t, y - y[:int(sf)].mean(), color=col, lw=0.7, label=lab)
    ax.set(title="C  20 s of Pz: slow drift and fast noise removed,\nbrain activity kept",
           xlabel="Time in recording (s)", ylabel="µV")
    ax.legend(frameon=False, fontsize=9)
    for a_ in axes:
        clean(a_)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- figure 2
def fig_continuous(volt, csd, ev, ids, sub, out, win=10.0):
    sf = volt.info["sfreq"]
    fp = volt.get_data(picks=["Fp1"])[0]
    best = None                    # a rare picture with a clear blink in the 10 s around it
    for rt in ev[ev[:, 2] == ids["rare"], 0] / sf:
        t0 = rt - 3.0
        a, b = int(t0 * sf), int((t0 + win) * sf)
        if a < 0 or b > len(fp):
            continue
        if np.ptp(volt.get_data(picks=TRACE_CHANS, start=a, stop=b), axis=1).max() > 250e-6:
            continue
        blink = (fp[a:b] - np.median(fp[a:b])).max()
        if best is None or blink > best[1]:
            best = (t0, blink, rt)
    t0, _, rt = best
    a, b = int(t0 * sf), int((t0 + win) * sf)
    t = np.arange(a, b) / sf - t0
    blink_t = (a + np.argmax(fp[a:b] - np.median(fp[a:b]))) / sf - t0
    p3_t = rt - t0 + 0.40

    fig = plt.figure(figsize=(17, 14), dpi=140)
    gs = fig.add_gridspec(2, 4, height_ratios=[3.2, 1], hspace=0.18, wspace=0.25)
    for col, (inst, title, unit, bar) in enumerate(((volt, "WITHOUT Laplacian: voltage (0.1–40 Hz, average ref)", "µV", 50),
                                                    (csd, "WITH Laplacian: CSD of the same data", "µV/m²", 50000))):
        ax = fig.add_subplot(gs[0, col * 2:col * 2 + 2])
        d = inst.get_data(picks=TRACE_CHANS, start=a, stop=b) * 1e6
        d -= d.mean(axis=1, keepdims=True)
        step = np.percentile(np.abs(d), 99) * 1.1
        n = len(TRACE_CHANS)
        for i in range(n):
            ax.plot(t, d[i] - i * step, color=INK, lw=0.6)
        ax.set_yticks(-np.arange(n) * step, TRACE_CHANS)
        for e in ev[(ev[:, 0] >= a) & (ev[:, 0] < b)]:
            x, rare = e[0] / sf - t0, e[2] == ids["rare"]
            ax.axvline(x, color=RED if rare else MUTED, lw=1.2 if rare else 0.6, ls="-" if rare else "--")
            ax.text(x, step * 1.2, "rare" if rare else "std", color=RED if rare else MUTED, fontsize=8, ha="center")
        for x, lab, c in ((blink_t, "A blink", BLUE), (p3_t, "B P3", ORANGE)):
            ax.axvspan(x - 0.02, x + 0.02, color=c, alpha=0.25)
            ax.text(x, -n * step + step * 0.1, lab, color=c, fontsize=9, ha="center", fontweight="bold")
        y0 = -n * step + step * 0.6
        ax.plot([win + 0.15] * 2, [y0, y0 + bar], color=INK, lw=2, clip_on=False)
        ax.text(win + 0.25, y0, f"{bar:g} {unit}", fontsize=8, va="bottom", rotation=90)
        ax.set(title=title, xlabel=f"Time (s) in a {win:.0f}-s window of {sub}'s run-01 (starts at {t0:.0f} s)",
               xlim=(0, win), ylim=(-n * step, step * 1.6))
        clean(ax)
    for j, (x, lab, c) in enumerate(((blink_t, "A blink", BLUE), (p3_t, "B P3 after rare", ORANGE))):
        i = a + int(x * sf)
        for k, (inst, ch_type, name, unit) in enumerate(((volt, "eeg", "voltage", "µV"), (csd, "csd", "Laplacian", "µV/m²"))):
            ax = fig.add_subplot(gs[1, j * 2 + k])
            v = inst.get_data(picks=ch_type, start=i - 10, stop=i + 10).mean(1) * 1e6
            lim = np.abs(v).max()
            im, _ = mne.viz.plot_topomap(v, inst.copy().pick(ch_type).info, axes=ax, show=False,
                                         cmap="RdBu_r", vlim=(-lim, lim), contours=6, sensors=True)
            ax.set_title(f"{lab}: {name}", color=c, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.75).set_label(unit)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- figure 3
def fig_erps(diffs, n_sub, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), dpi=140, gridspec_kw=dict(width_ratios=[1, 1, 1.6]))
    for row, (name, a, b, ch, win) in enumerate(CONTRASTS):
        if not diffs[(name, "volt")]:
            for ax in axes[row]:
                ax.set_axis_off()
            continue
        ga = {v: mne.grand_average(diffs[(name, v)]) for v in ("volt", "csd")}
        for col, (v, unit, lab) in enumerate((("volt", "µV", "WITHOUT Laplacian (voltage)"),
                                              ("csd", "µV/m²", "WITH Laplacian (CSD)"))):
            m = ga[v].copy().crop(*win).data.mean(1) * 1e6
            lim = np.abs(m).max()
            im, _ = mne.viz.plot_topomap(m, ga[v].info, axes=axes[row, col], show=False, cmap="RdBu_r",
                                         vlim=(-lim, lim), contours=6, sensors=True)
            axes[row, col].set_title(f"{lab}\n{name}, {win[0] * 1000:.0f}–{win[1] * 1000:.0f} ms", fontsize=10)
            fig.colorbar(im, ax=axes[row, col], shrink=0.7).set_label(unit)
        ax = axes[row, 2]
        t = ga["volt"].times
        for v, col, lab in (("volt", BLUE, "voltage"), ("csd", ORANGE, "Laplacian")):
            y = ga[v].copy().pick([ch]).data[0]
            ax.plot(t * 1000, y / np.abs(y).max(), color=col, lw=1.8, label=f"{lab} at {ch}")
        ax.axvspan(win[0] * 1000, win[1] * 1000, color=BAND, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6, ls="--")
        ax.set(title=f"{name} at {ch}, N = {len(diffs[(name, 'volt')])} (each curve scaled to its own peak)",
               xlabel="Time (ms)", ylabel="relative amplitude", xlim=(-200, 1000))
        ax.legend(frameon=False, fontsize=9)
        clean(ax)
    fig.suptitle(f"Filtering only (0.1–40 Hz, average ref), no ICA; epochs > 200 µV p-p dropped", fontsize=11)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_root"); p.add_argument("out_dir"); p.add_argument("subjects", nargs="+")
    p.add_argument("--window-sub", default=None, help="subject for figures 1-2 (default: last one)")
    p.add_argument("--shift-ms", type=float, default=0.0, help="move events later by this much (40 = screen delay)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    win_sub = args.window_sub or args.subjects[-1]

    diffs = {(c[0], v): [] for c in CONTRASTS for v in ("volt", "csd")}
    print(f"{'subject':9s}{'epochs':>8s}{'kept':>6s}  per condition (kept)")
    for sub in args.subjects:
        raw = load(args.data_root, sub)
        volt = filter_ours(raw)
        csd = laplacian(volt)
        ev, ids = events(volt, args.shift_ms)

        if sub == win_sub:
            fig_filter_effect(raw, volt, os.path.join(args.out_dir, "1_filter_effect.png"))
            fig_continuous(volt, csd, ev, ids, sub, os.path.join(args.out_dir, f"2_continuous_{sub}.png"))

        kw = dict(tmin=TMIN, tmax=TMAX, baseline=BASE, preload=True, reject_by_annotation=False)
        ep_v = mne.Epochs(volt, ev, ids, picks="eeg", **kw)
        ep_c = mne.Epochs(csd, ev, ids, picks="csd", **kw)
        assert len(ep_v) == len(ep_c)
        keep = np.ptp(ep_v.get_data(), axis=2).max(1) <= REJECT_PTP        # judged on voltage
        ep_v, ep_c = ep_v[np.where(keep)[0]], ep_c[np.where(keep)[0]]
        print(f"{sub:9s}{len(keep):8d}{keep.sum():6d}  " + ", ".join(f"{k} {len(ep_v[k])}" for k in ids))
        for name, a, b, ch, _ in CONTRASTS:
            if a in ids and b in ids:
                for v, e in (("volt", ep_v), ("csd", ep_c)):
                    diffs[(name, v)].append(mne.combine_evoked([e[a].average(), e[b].average()], [1, -1]))
    fig_erps(diffs, len(args.subjects), os.path.join(args.out_dir, "3_erp_voltage_vs_laplacian.png"))
    print("figures saved in", args.out_dir)


if __name__ == "__main__":
    main()
