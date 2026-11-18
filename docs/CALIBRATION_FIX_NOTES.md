# Calibration bug fix — AeroTwin V16 manuscript validation

## What was wrong
`aerotwin_manuscript_validation.py` computed ECE as:

    ece = np.mean(np.abs(fraction_pos - mean_conf))

This averages each populated confidence bin equally, regardless of how many
test samples fall in it. Standard ECE (Guo et al. 2017) weights each bin's
contribution by n_bin/N.

## Why it mattered here
Test set: n=1425, 21 errors. Bin populations were wildly uneven:

    [0.4,0.5)  n=2     conf=0.442  acc=0.000  gap=+0.442
    [0.5,0.6)  n=14    conf=0.546  acc=0.786  gap=-0.240
    [0.6,0.7)  n=25    conf=0.651  acc=0.680  gap=-0.029
    [0.7,0.8)  n=29    conf=0.752  acc=0.897  gap=-0.145
    [0.8,0.9)  n=53    conf=0.862  acc=0.943  gap=-0.082
    [0.9,1.0)  n=1302  conf=0.990  acc=0.998  gap=-0.008

The unweighted average lets the 2-sample bin (gap = +0.442) count exactly as
much as the 1302-sample bin (gap = -0.008), producing the reported
ECE = 0.1577 and the "moderate overconfidence" narrative.

## Corrected numbers
- Sample-weighted ECE (raw softmax): **0.0170** (not 0.1577)
- Sample-weighted ECE (after Temperature Scaling, T*=0.68): **0.0045** (not 0.1358)
- Aggregate: mean confidence 0.9695 vs. accuracy 0.9853 -> model is
  **underconfident** in aggregate, not overconfident.
- Every populated bin except the 2-sample bin shows underconfidence
  (conf < acc). This is consistent with T*=0.68 (<1, sharpening), which
  is the mathematically correct correction for underconfidence -- resolving
  Reviewer 1's stated contradiction.
- Reviewer 1's separate point about sparse bins is independently confirmed:
  4 of 6 populated bins have <30 samples; the lowest has only 2.

## Suggested revision language
Report ECE as ~0.017 (well-calibrated to mildly underconfident), state the
bin populations explicitly, and drop or heavily caveat the <0.5 confidence
bins as statistically unreliable rather than presenting them as evidence of
"moderate overconfidence." This directly answers Reviewer 1's calibration
contradiction with a code-level correction rather than a rewritten
justification.

## Reproducing this
This script (aerotwin_manuscript_validation.py) is the same one from your
Do228.zip workspace, with paths repointed to a local directory and the ECE
computation replaced by a `weighted_ece()` helper (see STEP 6/7 in the
script). Run directly:

    pip install tensorflow-cpu scikit-learn scipy matplotlib seaborn pandas numpy
    python aerotwin_manuscript_validation.py

with AeroTwin_V16_Model.h5, AEROTWIN_Do228_V16p1_Dataset.csv, scaler_seq.pkl,
and scaler_scalar.pkl in the same folder (BASE variable at top of script).
