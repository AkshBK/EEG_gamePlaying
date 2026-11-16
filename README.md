# ERPs in the oddball and gambling tasks (ds003517, all 17 subjects)

Data: OpenNeuro **ds003517** (Cavanagh & Castellanos, 2016, *NeuroImage*), run-01.
Question: after cleaning with ICA, does the **P3** stand out for rare pictures, and does the
**reward positivity** stand out for wins?

## Results

![Oddball](erp/figures/1_oddball_P3_Pz.png)

**Oddball, P3 at Pz (17 subjects).** Rare pictures give a larger positive wave than standard
pictures between 300 and 500 ms. The difference is strongest over the back-centre of the head,
as expected for a P3.

| | effect at Pz, 300–500 ms |
|---|---|
| Grand average, rare − standard | **+5.1 µV** (± 0.7 SE across subjects) |
| Subjects with a positive effect | **17 of 17** |
| One-sample t-test across subjects | t(16) = 7.2, p < 0.001 |

![Gambling](erp/figures/2_gambling_RewP_FCz.png)

**Gambling feedback, reward positivity at FCz (14 subjects).** Wins give a more positive wave
than losses around 220–320 ms, strongest over the front-centre of the head.

| | effect at FCz, 220–320 ms |
|---|---|
| Grand average, win − loss | **+2.1 µV** (± 0.2 SE across subjects) |
| Subjects with a positive effect | **14 of 14** |
| One-sample t-test across subjects | t(13) = 8.9, p < 0.001 |

sub-002, sub-003 and sub-005 have no gambling trials in run-01, so the gambling figure has 14 subjects.

### Each subject's effect (µV)

| | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 | 011 | 012 | 013 | 014 | 015 | 016 | 017 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P3, rare − standard | +0.3 | +6.0 | +6.7 | +2.3 | +3.4 | +4.8 | +12.8 | +8.6 | +5.3 | +5.3 | +4.5 | +2.5 | +2.7 | +4.5 | +9.2 | +3.8 | +4.9 |
| RewP, win − loss | +1.8 | – | – | +1.4 | – | +2.2 | +2.6 | +1.6 | +1.0 | +2.1 | +2.2 | +4.5 | +1.6 | +2.9 | +1.3 | +1.7 | +2.9 |

## How the averages are made

1. **Clean each subject:**
   - Drop FT9, FT10, TP9 and TP10.
   - Filter 0.1–40 Hz on the continuous recording, then apply the average reference.
   - Run ICA and remove the components ICLabel calls eye, heart or muscle.
   - Cut epochs from −0.5 to 1.0 s with a −200 to 0 ms baseline, and drop any epoch above 200 µV peak-to-peak.
2. **Average the trials within each subject**, per condition. This gives one ERP per condition per subject (small panels).
3. **Average the subjects:** the grand average is the mean of the subjects' ERPs, and every subject counts equally (large panel).

Trials kept after cleaning and the number of ICA components removed:

| | standard | rare | loss | win | ICA removed |
|---|---|---|---|---|---|
| sub-001 | 148 | 36 | 17 | 18 | 13 of 35 |
| sub-002 | 157 | 38 | – | – | 4 of 24 |
| sub-003 | 159 | 41 | – | – | 7 of 30 |
| sub-004 | 148 | 36 | 31 | 37 | 4 of 26 |
| sub-005 | 155 | 39 | – | – | 7 of 18 |
| sub-006 | 156 | 41 | 120 | 120 | 11 of 31 |
| sub-007 | 65 | 19 | 40 | 38 | 3 of 29 |
| sub-008 | 102 | 24 | 28 | 31 | 1 of 31 |
| sub-009 | 158 | 39 | 40 | 40 | 5 of 21 |
| sub-010 | 159 | 40 | 40 | 40 | 4 of 26 |
| sub-011 | 160 | 40 | 40 | 40 | 3 of 24 |
| sub-012 | 160 | 40 | 40 | 40 | 3 of 29 |
| sub-013 | 148 | 39 | 40 | 40 | 4 of 28 |
| sub-014 | 148 | 40 | 40 | 40 | 4 of 32 |
| sub-015 | 155 | 40 | 35 | 38 | 2 of 44 |
| sub-016 | 160 | 40 | 36 | 37 | 3 of 36 |
| sub-017 | 150 | 35 | 40 | 38 | 7 of 29 |

## Caveats

- Times are from the trigger. The screen shows each picture **40 ms later**, so the true
  latencies are about 40 ms earlier. Add `--shift-ms 40` to correct this.
- sub-007 and sub-008 lost many oddball trials to the 200 µV rule (65 and 102 of about 160
  standard trials kept), so their averages are noisier.
- sub-001 had 13 of 35 ICA components removed, mostly labelled muscle, and has the weakest P3
  (+0.3 µV). Its components should be checked.
- sub-006 has 120 win and 120 loss trials in run-01, about three times as many as the others.
  Every subject still counts equally in the grand average.

## Run it

```bash
pip install -r erp/requirements.txt
python erp/erp_oddball_gambling.py <data_root> erp/figures sub-001 sub-002 ... sub-017
```

`<data_root>` contains `sub-001/eeg/...` from https://openneuro.org/datasets/ds003517.
The data are not stored in this repository. A full run of 17 subjects takes about 20 minutes, mostly ICA.
