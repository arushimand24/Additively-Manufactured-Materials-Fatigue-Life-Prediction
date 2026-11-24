# Final Research Direction and Contribution

*FatigueData-AM2022 generalisation study, version 3, 20 November 2026. This document reconciles four stages of work: the primary experiment (version 1), the scientific revision (version 2), the technical audit and its closure (version 3, phase 1), and two supplementary investigations (version 3, phases 2 and 3). The primary experiment remains the main result. Every number below is computed by the notebooks and quoted from `reports/results_report.md` or the files it cites.*

## 1. The question and the main result

**Question.** How much does point-level random validation overstate the performance of machine-learning fatigue-life models, compared with curve-held-out and source-article-held-out validation, and can a Basquin-anchored residual model reduce that degradation?

**Main result.** This is the primary experiment, unchanged since version 1 and reproduced from a clean checkout. The setting is FatigueData-AM2022, with 12,666 failure points from 440 source articles in a nominal room-temperature cohort, eight inputs known before testing, and fixed model settings.

| Model | RMSE random | RMSE curve held out | RMSE article held out | Inflation C/A (95% CI) |
|---|---|---|---|---|
| Mean predictor | 0.993 | 0.994 | 0.994 | — |
| Basquin, per alloy | 0.938 | 0.954 | 0.992 | 1.06 (1.03–1.09) |
| Ridge | 0.897 | 0.951 | 0.970 | 1.08 (1.05–1.12) |
| Random Forest | 0.647 | 0.797 | 0.927 | 1.43 (1.36–1.50) |
| LightGBM | 0.673 | 0.787 | 0.924 | 1.37 (1.31–1.44) |
| Basquin-anchored residual | 0.667 | 0.788 | 0.940 | 1.41 (1.34–1.48) |

RMSE is in log10 cycles. Intervals come from a cluster bootstrap over articles of the saved out-of-fold predictions; they are conditional on the fitted models.

For the flexible models, the error on unseen articles is about 40% larger than random validation shows. Curve-held-out validation reveals only about half of that gap. On unseen articles, every model is close to the mean predictor.

## 2. Hypotheses and claims

**Status key.** *Supported*: the evidence meets the stated criterion. *Not supported*: it does not. *Inconclusive*: the evidence is too weak or confounded to decide.

### Pre-stated hypotheses

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1.** For the best direct learner, article-held-out RMSE ≥ 1.5 × random-split RMSE | **Not supported** at the stated threshold. The effect itself is large and robust | Inflation 1.43 (Random Forest; interval reaches 1.50) and 1.37 (LightGBM). Stable across three seeds and with articles weighted equally (1.37, 1.33). It exceeds 1.5 with richer features or author clusters held out |
| **H2.** The Basquin-anchored residual model has a smaller inflation *and* a lower article-held-out RMSE than direct gradient boosting | **Not supported**, in both constructions tested | In-sample anchor: 1.41 and 0.940 against 1.37 and 0.924. Cross-fitted anchor (investigation A): 1.38 and 0.959; worse than LightGBM by 0.035 (0.008–0.063) |
| **H3** (later work). Author-cluster-held-out error ≥ article-held-out error | **Inconclusive** | LightGBM 1.008 against 0.924. The clusters come from name matching, one of them chains 165 articles, and the folds are unequal; the split confounds a stricter unit with less training data |

### Claims made along the way

| Claim | Status | Evidence |
|---|---|---|
| Random validation overstates accuracy for flexible models | **Supported** | Inflation intervals exclude 1 for all three tree-based models. Robust to seeds, weighting, run-outs (S5), temperature evidence (S6) and the removal of ultrasonic tests (investigation B) |
| A large part of the article-held-out error is a whole-article offset | **Supported**, with the size depending on weighting | 48% point-weighted, 47% after the noise correction, 36% with articles weighted equally. The decomposition is statistical; its causes are not identified |
| For new sources, long predicted lives are non-conservative | **Supported** (supplementary, version 3) | Calibration slope 0.64; above 10⁷ predicted cycles, the true life is on average 0.61 log10 units shorter (303 points, 37 articles) |
| The tier-1 gain comes from Kt | **Not supported** (withdrawn in version 3) | Kt alone: 0.000 (−0.007 to +0.006). Frequency alone: −0.059. Notched rows give 1–6% of the gain |
| Test-regime information (the VHCF tag) improves transfer without raising the inflation | **Supported**, exploratory | −0.077 (−0.109 to −0.048), inflation 1.36. Consistent across three seeds and for Ridge |
| UTS features help under random validation but not on unseen articles | **Supported for the tree models only** | LightGBM +0.028 (−0.010 to +0.066), Random Forest −0.003; Ridge improves (−0.028, −0.054 to −0.004) |
| A reported UTS acts as a fingerprint of its source | **Inconclusive** | Plausible (82% of distinct values occur in one article), but not demonstrated |
| Re-published curves make the article-held-out error optimistic | **Supported, as a lower bound** | +0.012 (LightGBM) and +0.017 (Random Forest) under C2, averaged over three seeds. The size depends on the matching tolerance |
| Default temperatures distort the findings | **Inconclusive** (no evidence of distortion; the test is weak) | Ambiguous 25 °C rows reproduce the headline (0.921, inflation 1.38). The rows with evidence come from only 15 articles |
| The hybrid fails because of its in-sample residual target | **Not supported** (investigation A) | Cross-fitting makes the target honest (0.71 → 1.05 for moderate alloys) but the hybrid worse |
| The fitted models give the expected direction to surface, HIP and mean stress | **Supported on average only** | Correct average signs; not row by row (HIP helps for 69% of rows in LightGBM). The tree models' mean-stress penalty has the wrong order between R = 0.1 and R = 0.5 |

## 3. How the four stages fit together

- **Version 1** established the primary result and the hypothesis verdicts. A code review fixed the near-duplicate search.
- **Version 2** corrected descriptions without changing any number. The cohort became a nominal room-temperature cohort, the SWT form was documented, run-out counts were reconciled, the bootstrap was described as conditional, error factors were turned into percentiles, the decomposition got robustness checks, and the direction claim was narrowed.
- **Version 3, phase 1** closed the technical audit. The pipeline now runs from a fresh checkout, and the provenance record is rebuilt when absent. Three interpretations were corrected (calibration, the Kt attribution, UTS per model), the near-duplicate rule was made symmetric, and 23 audit checks joined the test suite (42 tests). No primary number changed: predictions are bit-identical, apart from Random Forest at 10⁻¹⁴.
- **Version 3, phase 2 (investigation A)** removed one possible objection to the H2 verdict. The conclusion that a per-alloy Basquin anchor fitted to pooled literature data does not help transfer became stronger.
- **Version 3, phase 3 (investigation B)** identified the largest avoidable source of error: test regimes mixed without a label. It showed that the primary generalisation finding does not depend on them.

Where the stages disagree, the later stage is right and the earlier statement is withdrawn. Every such case is listed in `reports/CHANGELOG.md`.

## 4. Kinds of evidence

| Kind | Examples |
|---|---|
| **Independently verified** | Counts and funnel rebuilt from the raw workbook. Every model rebuilt from the documented settings reproduces the saved predictions exactly. Fold rule, group separation and fitting on training rows only, in all ten prediction files (tests and the audit). Metrics and bootstrap re-implemented. A clean-checkout run reproduces all outputs |
| **Newly calculated in version 3** | Calibration slopes and bins; article-weighted inflation; reasons for the pooled-line fallback; per-model S3 and S4 intervals; tier-1 gain by row group; near-duplicate tolerance sensitivity; investigation A and B results |
| **Interpretation** | Physical causes of article offsets (defects, roughness, machines). Why the cross-fitted anchor fails (unstable per-alloy lines). Why the VHCF tag transfers better than exact frequency (rig fingerprints). Physical reasons for different ultrasonic lives (strain rate, self-heating, stressed volume, internal initiation). All of these are labelled as untested |
| **Uncertain** | True test temperatures of the 25 °C rows. Completeness of the resonance tag. About 18 ultrasonic points without a recorded frequency. Digitisation accuracy. Behaviour on other machines and platforms. Uncertainty that includes refitting. External validity |

## 5. Figures and tables for a paper

Notebook 04 (section 14.1) writes a numbered set to `results/paper/`, with captions in `results/paper/README.md`. Items 1–5 are the primary results; items marked S, together with Figure 6 and Table 6, are supplementary.

| Item | Content |
|---|---|
| Figure 1 | Filter funnel |
| Figure 2 | RMSE by model and validation scheme, with the mean-predictor line |
| Figure 3 | Inflation factors with conditional 95% intervals |
| Figure 4 | Article mean errors under random and article-held-out validation |
| Figure 5 | Calibration of LightGBM, random against article-held-out validation |
| Figure 6 | Supplementary B: ablation and error by test regime |
| Figures S1–S3 | Investigation A by alloy coverage; predicted S-N curves; per-alloy RMSE |
| Table 1 | Filter funnel |
| Table 2 | Main results with intervals |
| Table 3 | Hypotheses and verdicts |
| Table 4 | Size of single errors |
| Table 5 | Sensitivity analyses |
| Table 6 | Ablation |
| Tables S1–S4 | Calibration bins, investigation A by coverage, test regimes, regime variants |

## 6. Limitations

- **One database, internal validity only.** No external dataset. Coupons are not components. Article-held-out is not laboratory-held-out.
- **Inputs.**
  - Digitised data.
  - Defaults posing as values: 25 °C for 98% of the cohort, Kt = 1, NHT, as-built.
  - No defect, porosity or roughness information.
  - Coarse keyword categories; alloy names used as written.
  - The generic bending label.
  - Load-control modes pooled.
- **Design choices.**
  - Run-outs excluded from the primary analysis.
  - Fixed, untuned hyperparameters.
  - Point-weighted headline metrics.
  - The primary feature set does not distinguish test regimes.
- **Uncertainty.**
  - Bootstrap intervals are conditional on the fitted models.
  - The supplementary analyses were defined after the primary result, and their intervals are not corrected for multiple comparisons.
- **Grouping.**
  - The near-duplicate count is a lower bound that depends on the tolerance.
  - The author clusters are crude.

## 7. Future work, in order of priority

1. **Pre-register a feature set that includes the VHCF tag**, a label known before testing, and repeat the three-scheme comparison, including calibration.
2. **Per-source offset modelling** (mixed effects or few-shot calibration): how many points from a new source are needed to recover the lost accuracy?
3. **Calibrated uncertainty for new sources:** conformal intervals calibrated per held-out article, and repeated grouped cross-validation with refitting.
4. **A censored-data model for run-outs**, under the same schemes.
5. **Nested, grouped hyperparameter tuning**, to bound the share of the error that is due to the fixed settings.
6. **Other physics anchors**: a pooled line, or a minimum number of articles per alloy line. Cross-fitting alone did not help.
7. **Manual verification from the original articles** of a sample of the 25 °C rows, the untagged resonance tests and the ultrasonic tests without a recorded frequency.
8. **An external check** on a second, independent fatigue database, if one becomes available.

## 8. Contribution statement

*The statement describes what this work does. It makes no claim of priority: the competing literature has not been rechecked since the planning documents (see section 9).*

> This study measures, on the open FatigueData-AM2022 database and with a leakage-controlled, fully reproducible pipeline, how much random point-level validation overstates the accuracy of machine-learning fatigue-life models for additively manufactured alloys. With identical data, features, folds and settings, the error on unseen source articles is about 40% larger than random validation shows for flexible learners (inflation 1.37–1.43). Curve-held-out validation reveals only about half of that gap. On unseen articles every model is close to a mean predictor, and between a third and a half of the remaining error is a whole-article offset.
>
> A Basquin-anchored residual model does not reduce the gap, whether its correction is trained on in-sample or on cross-fitted anchor residuals.
>
> Supplementary analyses show three further things. Predictions for new sources are miscalibrated, with non-conservative long-life predictions. Feature choices made under random validation can be wrong. A single label known before testing, which identifies ultrasonic very-high-cycle tests, removes the largest identifiable source of avoidable error without changing the generalisation gap.
>
> The practical message is methodological: models trained on literature compilations should be validated with whole sources held out, reported per source and checked for calibration, and they should be told how each test was run.

## 9. Before any submission

1. **Literature recheck (required).** Repeat the search for article- or source-held-out validation on FatigueData-AM2022 and similar compilations. The novelty statement in the planning documents is unverified, and the cited studies were not re-read.
2. **Reproduce the figures** from a clean checkout. The procedure is tested (`audit/clean_checkout_run_v3.sh`), and outputs are identical apart from Random Forest at 10⁻¹⁴.
3. **Decide whether the paper uses the primary feature set only**, with the VHCF result as a supplementary finding, or pre-registers a new feature set and reruns. The second requires a new primary experiment.
4. **Cite the dataset** (Zhang & Xu, Scientific Data 10:249, 2023; Figshare DOI 10.6084/m9.figshare.22337629, CC BY 4.0) and state that the derived data are redistributed under its licence.
