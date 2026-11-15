# Filtering and ICA, each with and without a surface Laplacian

Data: OpenNeuro **ds003517** (Cavanagh & Castellanos, 2016, *NeuroImage*), run-01:
oddball pictures (standard / rare) and gambling feedback (win / loss).
Subjects used here: sub-001 to sub-004.

This folder shows **two cleaning steps of our pipeline**: the filter, and ICA.
Every result exists in four versions:

|  | without Laplacian (µV) | with Laplacian (µV/m²) |
|---|---|---|
| **filter only, no ICA** | figures 2, 3 | figures 2, 3 |
| **filter + ICA** | figures 5, 6 | figures 5, 6 |

Not included: bad-channel repair, autoreject, and the game run.

## What the script does

1. Load the raw file and add back the online reference channel CPz.
2. Drop the four low edge channels FT9, FT10, TP9 and TP10, which are often noisy.
3. Band-pass **0.1–40 Hz on the continuous recording**. This is MNE's default FIR filter:
   zero-phase with a Hamming window, so the ERPs are not shifted in time.
4. Average reference.
5. **ICA version only:**
   - Make a separate copy of the recording filtered 1–100 Hz, with notches at 60 and 120 Hz.
     ICA learns better without slow drift, and ICLabel was trained on data like this.
   - Cut the copy into 1-s pieces and leave out only pieces above 500 µV. Blinks stay in,
     because ICA has to see blinks to learn them.
   - Fit ICA with Picard (extended infomax), keeping the components that explain 99% of the variance.
   - ICLabel names every component. Components labelled eye (probability > 0.8),
     heart (> 0.8) or muscle (> 0.95) are removed from the 0.1–40 Hz data of step 3.
6. Two versions of the data from step 4 (or step 5):
   - **Without Laplacian:** the voltage, in µV.
   - **With Laplacian:** the surface Laplacian, also called current source density (CSD),
     from `mne.preprocessing.compute_current_source_density`, in µV/m². It always comes **after** ICA.
7. Epochs from −0.5 to 1.0 s with a −200 to 0 ms baseline. An epoch is dropped when any channel
   exceeds 200 µV peak-to-peak. Voltage and Laplacian use **the same trials**.

## Figures (in `figures/`)

| File | What it shows |
|---|---|
| `1_filter_effect.png` | A: the filter's frequency response. B: power spectrum at Pz before and after filtering. C: 20 s of Pz before and after. |
| `2_continuous_sub-004.png` | No ICA: 10 s of continuous EEG, voltage beside Laplacian, with scalp maps at a blink and 400 ms after a rare picture. |
| `3_erp_voltage_vs_laplacian.png` | No ICA: grand-average P3 (rare − standard at Pz) and reward positivity (win − loss at FCz), voltage beside Laplacian. |
| `4_ica_removed_sub-004.png` | The scalp maps of the components ICA removed for sub-004, with ICLabel's label and probability. |
| `5_continuous_ica_sub-004.png` | The same 10 s as figure 2: before ICA, after ICA, and after ICA with the Laplacian. |
| `6_erp_ica_voltage_vs_laplacian.png` | Figure 3 after ICA. |
| `7_erp_no_ica_vs_ica.png` | P3 and reward positivity in µV, without and with ICA, on one axis. |
| `summary.txt` | Per subject: components removed and trials kept, with and without ICA. |

## What to look at

- **The filter (figure 1).** Between 0.1 and 40 Hz the before and after spectra lie on top of each
  other, so brain activity is untouched. Drift below 0.1 Hz, the 60 Hz US mains line and fast muscle
  noise are removed.
- **Blinks without ICA (figure 2).** In the voltage, a blink spreads to every channel. The dip at the back
  of the head comes from the average reference, not from the brain. In the Laplacian the blink stays at
  the front, but it is not removed.
- **ICA (figures 4 and 5).** The removed components should look like eyes: strongest at the very front
  of the head. After ICA the blinks are gone from every channel, and the rest of the signal is unchanged.
- **P3 (figures 3 and 6, top).** The P3 is broad, and the Laplacian breaks it into patches, with the
  strongest values at edge electrodes. For the P3, the voltage at Pz is the better measure.
- **Reward positivity (figures 3 and 6, bottom).** Local frontal-central effects like this are where a
  Laplacian usually helps. With only 2 subjects it is too noisy to judge.
- **Without vs with ICA (figure 7).** With ICA, the P3 at Pz is smaller (about 4–5 µV instead of
  about 6–7 µV at its peak). We have not checked yet whether eye activity inflated the P3 without
  ICA, or whether ICA removed some brain activity. Looking at single subjects would tell.
- **Units.** A Laplacian is voltage divided by distance squared, so its numbers (µV/m²) are huge and
  cannot be compared with µV.

## Trials kept

An epoch is dropped when any channel exceeds 200 µV peak-to-peak. Without ICA, that rule
mostly throws away trials with blinks.

| Subject | Without ICA: kept | With ICA: kept | With ICA, per condition |
|---|---|---|---|
| sub-001 | 191 / 235 | 219 / 235 | standard 148, rare 36, loss 17, win 18 |
| sub-002 | 86 / 200 | 195 / 200 | standard 157, rare 38 |
| sub-003 | 90 / 200 | 200 / 200 | standard 159, rare 41 |
| sub-004 | 157 / 280 | 252 / 280 | standard 148, rare 36, loss 31, win 37 |

Components removed by ICA (details in `figures/summary.txt`):

| Subject | Removed | Eye | Muscle |
|---|---|---|---|
| sub-001 | 13 of 35 | 2 | 11 |
| sub-002 | 4 of 24 | 3 | 1 |
| sub-003 | 7 of 30 | 2 | 5 |
| sub-004 | 4 of 26 | 2 | 2 |

No heart components passed the threshold in any subject.

## Caveats

- Only sub-001 and sub-004 have gambling events, so the reward positivity uses N = 2.
- sub-001 has 11 muscle components removed. That is a lot, and it is worth looking at its
  components before trusting the threshold for that subject.
- ICA here is fitted on run-01 alone. Our full pipeline fits one ICA on both runs together.
- Event times are as stored in the files. The screen shows each picture **40 ms after** its trigger,
  and the authors correct for this in their code. Add `--shift-ms 40` to apply the same correction.

## How to run

```bash
pip install -r requirements.txt
python filter_laplacian.py <data_root> figures sub-001 sub-002 sub-003 sub-004 --window-sub sub-004
```

`<data_root>` is the folder that contains `sub-001/eeg/...`, downloaded from
https://openneuro.org/datasets/ds003517. The data are not stored in this repository.
Running all four subjects takes several minutes, mostly for ICA.

Tested with MNE 1.13.2, mne-icalabel 0.10.0, python-picard 0.8.2, onnxruntime 1.31.1,
NumPy 2.5.4, SciPy 1.18.1 and Matplotlib 3.11.2.
