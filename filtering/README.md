# Filtering step, with and without a surface Laplacian (no ICA)

Data: OpenNeuro **ds003517** (Cavanagh & Castellanos, 2016, *NeuroImage*), run-01:
oddball pictures (standard / rare) and gambling feedback (win / loss).
Subjects used here: sub-001 to sub-004.

This folder shows **only the filtering part of our pipeline**. Nothing else is applied:
no ICA, no bad-channel repair and no autoreject. The goal is to see what the filter does,
and what a surface Laplacian changes on top of it.

## What the script does

1. Load the raw file and add back the online reference channel CPz.
2. Drop the four low edge channels FT9, FT10, TP9 and TP10, which are often noisy.
3. Band-pass **0.1–40 Hz on the continuous recording**. This is MNE's default FIR filter:
   zero-phase with a Hamming window, so the ERPs are not shifted in time.
4. Average reference.
5. Two versions of the same data:
   - **Without Laplacian:** the voltage, in µV.
   - **With Laplacian:** the surface Laplacian, also called current source density (CSD),
     from `mne.preprocessing.compute_current_source_density`, in µV/m².
6. Epochs from −0.5 to 1.0 s with a −200 to 0 ms baseline. Because there is no ICA, an
   epoch is dropped when any channel exceeds 200 µV peak-to-peak, which mostly catches blinks.
   Voltage and Laplacian use **the same trials**.

## Figures (in `figures/`)

| File | What it shows |
|---|---|
| `1_filter_effect.png` | A: the filter's frequency response. B: the power spectrum at Pz before and after filtering. C: 20 s of Pz before and after. |
| `2_continuous_sub-004.png` | 10 s of continuous EEG, voltage beside Laplacian, with scalp maps at a blink and 400 ms after a rare picture. |
| `3_erp_voltage_vs_laplacian.png` | Grand-average P3 (rare − standard at Pz) and reward positivity (win − loss at FCz), voltage beside Laplacian. |

## What to look at

- **The filter (figure 1).** Between 0.1 and 40 Hz the before and after spectra lie on top of each
  other, so brain activity is untouched. Drift below 0.1 Hz, the 60 Hz US mains line and fast muscle
  noise are removed.
- **Blinks (figure 2).** In the voltage, a blink spreads to every channel. The dip at the back of the
  head comes from the average reference, not from the brain. In the Laplacian the blink stays at the
  front. The Laplacian does **not** remove blinks, so ICA is still needed for that.
- **P3 (figure 3, top).** The P3 is broad, and the Laplacian breaks it into patches, with the strongest
  values at edge electrodes. For the P3, the voltage at Pz is the better measure.
- **Reward positivity (figure 3, bottom).** Local frontal-central effects like this are where a
  Laplacian usually helps. Here it is only 2 subjects with few trials and no ICA, so it is too noisy
  to judge.
- **Units.** A Laplacian is voltage divided by distance squared, so its numbers (µV/m²) are huge and
  cannot be compared with µV.

## Caveats

- No ICA, so the simple 200 µV rule throws away many blink trials:

  | Subject | Epochs | Kept | Per condition (kept) |
  |---|---|---|---|
  | sub-001 | 235 | 191 | standard 128, rare 29, loss 16, win 18 |
  | sub-002 | 200 | 86 | standard 73, rare 13 |
  | sub-003 | 200 | 90 | standard 62, rare 28 |
  | sub-004 | 280 | 157 | standard 101, rare 30, loss 11, win 15 |

- Only sub-001 and sub-004 have gambling events, so the reward positivity uses N = 2.
- Event times are as stored in the files. The screen shows each picture **40 ms after** its trigger,
  and the authors correct for this in their code. Add `--shift-ms 40` to apply the same correction.

## How to run

```bash
pip install -r requirements.txt
python filter_laplacian.py <data_root> figures sub-001 sub-002 sub-003 sub-004 --window-sub sub-004
```

`<data_root>` is the folder that contains `sub-001/eeg/...`, downloaded from
https://openneuro.org/datasets/ds003517. The data are not stored in this repository.
Running all four subjects takes a few minutes.

Tested with MNE 1.13.2, NumPy 2.5.4, SciPy 1.18.1 and Matplotlib 3.11.2.
