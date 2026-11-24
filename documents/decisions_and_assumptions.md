# Decisions, Assumptions and Deviations

This file records how the project specification was interpreted, where its documents disagreed, what was decided and why, what was assumed, and what was not done. It exists so that every choice can be questioned.

Build date: 20 November 2026; revised the same day as version 2 after an independent review (section 9, and `reports/CHANGELOG_v2.md`), and as version 3 after a technical audit (section 10, and `reports/technical_closure_v3.md`). Specification: the eleven Markdown files in `FatigueAM_context_files/context/` (the two Markdown files in the project root and the Word dossier are verbatim copies of them, checked by text comparison).

## 1. Decisions taken by the project owner

Asked once, before implementation, and answered by the project owner:

| Question | Decision |
|---|---|
| Three or four notebooks? | Four, as in the implementation specification |
| Test temperatures in the primary dataset? | Room temperature only (15 to 35 °C). Implemented as a *nominal* room-temperature cohort, because 25 °C is also the database default (section 9) |
| Scope beyond the mandatory part? | Mandatory scope, all P1 items, and key sensitivity checks |
| Version control? | No git repository |

## 2. Conflicts between documents and how they were resolved

| # | Conflict | Resolution | Reason |
|---|---|---|---|
| 1 | The execution brief prefers three notebooks and names notebook 3 `03_Model_Development_and_Validation`; the implementation specification has four, with `03_Model_Development_and_Training` and `04_Model_Evaluation_and_Research_Findings` | Four notebooks with the specification's names | Owner's decision; the brief allows a fourth notebook for evaluation and findings |
| 2 | The original notebook brief asked notebook 04 to "load trained models" and also to "implement" the validations | Notebook 03 runs all cross-validated training and saves out-of-fold predictions; notebook 04 only evaluates them | Cross-validation trains a model per fold, so there is no single trained model to load. This is the resolution proposed in specification 09, section B.3 |
| 3 | Location of the context documents: `docs/context/` (specification), `context/` (brief), `FatigueAM_context_files/context/` (actual) | Left where they were | The brief forbids deleting, renaming or overwriting the specifications |
| 4 | The handoff prompt says "stop and ask" at three gates; the brief says to ask only when blocked | Gates 1, 2 and 3 are evaluated by code and recorded in the notebooks. All three were met, so no question was needed | Nothing was ambiguous: the file downloaded, every key column was identified, 440 articles remain |
| 5 | The specification keeps raw data "out of version control"; the owner chose no git | A `.gitignore` excluding `data/raw/` and `.venv/` is provided, and no repository was initialised | Satisfies both |
| 6 | "Four-hour sprint" in the specification against an overnight build | The specification's priorities were kept (P0, then P1, then selected P2) | Owner's scope decision |
| 7 | Feature list: "fatigue test type (axial or bending)" | Three classes: axial, rotating bending, plane bending | Rotating bending is always fully reversed and loads the specimen differently from plane bending; three classes follow directly from the ten strings in the file |
| 8 | The specification lists seven automated tests | Version 3 has two test files: the seven required tests and further leakage and correctness tests (`test_leakage.py`), and the checks of the technical audit (`test_audit_checks.py`); the current count is in `results/metrics/test_report.txt` | More coverage at no cost to clarity |

## 3. Differences between the documented assumptions and the actual data

| Assumed in the planning documents | Found in the file |
|---|---|
| Column headers unknown | Headers discovered and mapped (`results/audit/column_name_mapping.csv`); the `parameter` sheet has two header rows |
| The article identifier might exist only in the JSON file | Title, authors and DOI are in the workbook for every dataset; the JSON is not needed |
| 15,146 points, 1,610 curves, 459 articles (reported) | Reproduced exactly |
| "As-built" and "NHT" are filled in as defaults | Only partly: 277 curves say "NHT" and 361 have no heat-treatment step at all; 422 say "as-built" and 73 have no surface step |
| Test standard reported by 40% of articles | 26% of S-N articles have a test standard |
| L-PBF holds 83% of S-N data | 80.8% of curves, 79.9% of points |
| Run-out count unknown | 1,851 points in the whole S-N sheet (12.2%); 1,748 of them in the nominal room-temperature cohort, the other 103 on curves removed at the R step (30) and the temperature step (73); coding 1 = run-out confirmed from the data |
| Test temperature reported for every test | The column is never empty; 94% of all datasets and 98% of the cohort are at 25 °C, the dataset authors' default for an unreported temperature. A reported and a default 25 °C cannot be told apart, in the workbook or in the JSON file |
| Exact duplicates across articles are possible (review papers) | None exist. Instead, near-duplicate curves were found: the same test series published in two or more articles by the same groups. 85 pairs in the whole S-N sheet; 78 pairs in the nominal room-temperature cohort including its run-outs, the population the search uses (119 curves, 66 articles; 8.4% of the modelling points). On the failures alone the same rule finds 80 pairs |

## 4. Methodological decisions made during implementation

| Topic | Decision | Reason |
|---|---|---|
| Load ratio cells with two values ("0, 0.1") | Set to missing and counted (3 curves); the curve leaves the dataset at the R filter | Two values are bounds, not a measurement; the specification forbids silent averaging |
| Heat-treatment class | `HIP` if any HIP step, else `HT` if any HT step, else `NHT_or_unreported` | Small, documented set; "NHT" cannot be told apart from "not reported" |
| Surface class | Keyword mapping of each surface step, then the highest-priority class among peened, polished, machined, blasted, other, as-built | Explicit rule shown in notebook 02 with the full mapping table; 16 of 80 distinct steps fall in `other` |
| Exact duplicates inside a curve | Kept and logged (2 rows in the cohort) | The specification removes only cross-article copies and says no other row is removed |
| Near-duplicate curves across articles | Kept in the main dataset; a sensitivity split (C2, three seeds) holds linked articles out together | The specification says near-duplicates are reported, not removed. The sensitivity split measures their effect without deleting data |
| Notched specimens, strain- or displacement-controlled tests, tests in corrosive media | Kept | The funnel was fixed in advance and names no such filter. Kt enters only the tier-1 sensitivity run |
| Alloy names | Used as written (93 alloys); rare alloys are grouped by the one-hot encoder inside each training fold (`min_frequency = 20`) | No merging rule could be justified without metallurgical judgement on each name |
| Alloy family | A keyword rule, used for descriptive tables only | Never a model input |
| Basquin fit | Per alloy with at least 30 training points from at least 3 curves and a negative slope; otherwise the pooled line | As specified; curve identifiers are passed only to count curves per alloy |
| LightGBM row subsampling | `subsample_freq = 1` added | LightGBM ignores `subsample = 0.8` unless the frequency is positive; this makes the specified setting effective |
| "Best direct learner" in H1 | Reported for both readings: lowest random-split RMSE (Random Forest) and the main learner of the methodology (LightGBM) | The specification does not define "best"; neither reading reaches 1.5 at the point estimate |
| Mean predictor | Added as a reference row in notebook 04 | Makes clear what "no skill" looks like under each scheme |
| Monotone model (sensitivity) | LightGBM with a decreasing constraint on log stress, without `log_swt` | `log_swt` is a function of stress and R; a constraint "with everything else fixed" would be meaningless with a second stress column |
| Author clusters (sensitivity) | Articles sharing any author name string, chained | Simple and reproducible; known to be crude (one cluster of 165 articles). Reported as indicative only |
| Bootstrap | 1,000 resamples of whole articles, the same resamples for all models and schemes, applied to the saved out-of-fold predictions without refitting | Points within an article are not independent; pairing allows model differences to be judged. The intervals are conditional on the fitted models, the folds and the fixed settings |
| Rows recorded at 25 °C | Kept in the primary cohort and labelled by their temperature evidence (`temperature_evidence`, an audit column on the forbidden list) | The owner chose room temperature only; the default cannot be separated from a reported value, and excluding every 25 °C row would leave 4 articles |
| `log_swt` | Kept under its name and documented as the stress-based SWT form, √(σmax·σa) with R = σmin/σmax, from nominal stresses | It is the conventional stress-life form of the SWT parameter (the strain-based parameter under linear-elastic behaviour), not an ad hoc proxy; renaming would change file schemas without changing any value |
| Ridge and extreme load ratios | The 13 points with R of −3 or −10 are kept; their effect is shown separately in notebook 04 | Removing data after seeing its effect on a result is not acceptable |

## 5. Deviations from the specification

1. **Goodman equivalent amplitude was not implemented.** Tier 2 uses σa/UTS and log10(UTS). The Goodman amplitude σa/(1 − σm/UTS) is undefined or negative when the mean stress reaches the tensile strength, which happens for a number of points in this database (the maximum stress exceeds the reported UTS for 210 points of the modelling dataset, 222 of the cohort with run-outs), so it would have required an extra exclusion rule.
2. **Seaborn is not used**, although the brief lists it as preferred. Matplotlib alone was sufficient.
3. **The JSON file is downloaded in addition to the workbook.** Since version 2, notebook 02 reads it for one cross-check of the temperature field. Since version 3, notebook 01 downloads it, with an MD5 check, if it is absent, and the provenance record covers it.
4. **`run_all.py`** was added to the project root so that the pipeline can be rerun with one command.
5. **The `S-N` scatter "coloured by article"** highlights the three largest articles of each alloy against all others in grey, instead of giving each of up to 154 articles its own colour, which would be unreadable.

## 6. Assumptions

- The dataset authors' statements about digitisation accuracy are taken from the planning documents and were not verified independently. Their statements on default values, on the definition R = σmin/σmax and on the conversion of maxima to amplitudes (σa = (1 − R)/2 · σmax) were checked against the article text (PubMed Central copy, PMC10151339) in version 2.
- A DOI identifies one source article. DOI and title correspond one-to-one in the file.
- A recorded 25 °C is either a reported 25 °C or "room temperature", or the authors' default for an unreported temperature; the file cannot separate these. The filter keeps 20, 23 and 25 °C, the only values between 15 and 35 °C among the S-N curves. *Engineering judgement, not verified:* tests at an unreported elevated temperature should be rare, because an elevated-temperature fatigue test is normally the subject of its article, which then reports the temperature.
- Curves of the same alloy in different articles are treated as the same test data published twice when at least 4 points and at least 80% of the smaller curve coincide within about 2% in stress and 5% in life. The search runs on the cohort including run-outs; when both curves have the same number of points, matches are counted in both directions and the larger count is used (version 3). The two articles share an author cluster in every one of the 78 pairs, which supports the rule, but the tolerance is a judgement and the count is a lower bound (8 more pairs appear when alloy names may differ).
- The literature entries cited in the report are those collected in the planning documents. They were not re-read or re-verified during implementation.

## 7. What was not done

Deferred by the specification (P2) and not attempted:

- Censored-data (accelerated-failure-time) models for run-outs.
- Temporal split (train on earlier years, test on later years).
- Nested, grouped hyperparameter tuning.
- Conformal prediction intervals.
- SHAP values, permutation importance and partial-dependence analysis on held-out articles.
- "Default treated as unknown" sensitivity run.
- Reproduction of the models of earlier studies on this database.

Done from the deferred list, as sensitivity analyses: run-outs treated as failures, a monotone-constrained model, an author-cluster-held-out split. Added in version 2, as supplementary checks: a direction-of-effect analysis on the final models (in-sample, not the deferred partial dependence on held-out articles), a monotonicity check over every curve, and a subgroup check of the test temperature (S6).

Not verified:

- The competing-literature search has not been repeated; the novelty statement in the planning documents ("no published study reporting article-held-out validation on this database as of 20 November 2026") is unchanged and unconfirmed by this build.

## 8. Independent review and what it changed

After the first complete run, the code was reviewed independently for leakage and correctness (a separate read-only pass over `src/`, the tests, the notebooks' code and the saved outputs).

Confirmed as correct by the review: the join and the funnel (rebuilt independently, same 12,666 rows); the feature lists; the nesting of curves, articles and groups; fold assignment and group separation in all prediction files; fitting inside folds; the metric formulas, inflation table, article bootstrap (re-implemented independently, same intervals) and hypothesis rules; reproducibility of the predictions.

Defects found and fixed before the final run:

| Finding | Fix |
|---|---|
| The near-duplicate search compared grid cells, so two close points on either side of a cell edge did not match. It found 40 pairs in the cohort where a proper tolerance comparison finds 78 | `find_near_duplicate_curves` now checks the true distance against the stated tolerance (own and neighbouring cells). `article_group_id`, the C2 sensitivity run and all related figures were regenerated. C2 is now also run for three seeds, because regrouping reshuffles every fold |
| The run-out sensitivity run (S5) was described as using the same folds as the primary run; that is only true for the article-held-out scheme | Wording corrected in notebook 04 |
| One Basquin test could not fail | Replaced by a test that alters the lives of a held-out fold and checks that its predictions do not change, for all five models |
| The table of rare alloys compared slightly different row sets across schemes | The row set is now defined once, under experiment C |
| Without LightGBM, the monotone sensitivity model would have been silently unconstrained | The fallback learner now receives the constraint |
| Short surface keywords could match inside other words ("roll" in "controlled") | Word boundaries added; the mapping of all 80 surface steps in the data is unchanged |

Disclosed, not changed: alloy names are used as written, so spelling variants of one alloy are separate categories; eight pairs of near-identical curves inside single articles are kept; the monotone sensitivity run changes two things at once (constraint, and no `log_swt`).

None of these changed the primary results for experiments A, B and C, which were reproduced to within floating-point rounding.

## 9. Second review (version 2): what was checked and what changed

After version 1, an independent reviewer read the three reports and raised eight points. Each was checked against the code, the raw files and the saved outputs. Version 1 is kept unchanged in `archive/v1_2026-11-20/` (reports, notebooks, source, tests, metrics, audit tables, figures, processed data, primary predictions and the hashes of all prediction files). The concise change log is `reports/CHANGELOG_v2.md`.

| # | Review point | What the check found | Status | Change |
|---|---|---|---|---|
| 1 | Temperature defaults | The temperature column is never empty (workbook and JSON); 25 °C for 94% of all datasets, 98% of the cohort. The paper states the 25 °C ambient assumption. Only 129 points (4 articles) carry another room temperature; 131 points (11 articles) are at 25 °C in articles that report other temperatures | Confirmed; distinction not recoverable | Cohort described as nominal room temperature everywhere; audit table and `temperature_evidence` column added; S6 subgroup check: the ambiguous rows reproduce the headline (RMSE 0.921, inflation 1.38) and the evidence rows are not predicted better (1.04, 15 articles). Exclusion run not feasible (4 articles); primary cohort unchanged |
| 2 | SWT definition | `log_swt` = log10(√(σmax·σa)), σmax = 2σa/(1 − R), R = σmin/σmax, MPa, nominal stresses; matches the paper's conventions | Confirmed as the conventional stress-based form | Not renamed; formula, conventions, units and limits documented in the code, notebook 02, README and report; a test checks the formula |
| 3 | Run-out counts | 1,851 = whole sheet; 1,748 = nominal room-temperature cohort; 30 + 73 removed with their curves at steps 5 and 6 | Confirmed; wording corrected | Reconciliation table in notebook 02 (`results/audit/runout_reconciliation.csv`); every count now names its population |
| 4 | Bootstrap | Resamples fixed out-of-fold predictions; no refitting | Confirmed; no implementation error | Intervals described as conditional on the fitted models, folds and settings, in the code, notebook 04 and report |
| 5 | Basquin generalisation | Pooled line for 12.5% / 13.9% / 16.8% of held-out rows (A/B/C); 1,042 rows of 60 alloys absent from the training fold under C. For the five well-represented alloys the residual model equals LightGBM (+0.001); the deficit comes from 14 moderately represented alloys with their own line (+0.18, 102% of the gap) | New finding | Supplementary section 9.2 of notebook 04 (report 6.1); primary result unchanged |
| 6 | Error decomposition and engineering claims | The decomposition is an exact point-weighted partition; robust to a noise correction (47%) but 36% with articles weighted equally. The "direction of effects" claim had not been tested; the new check supports it only on average | Partly corrected | Method and assumptions documented; "about half" became "between a third and a half"; causal wording removed; direction claim narrowed (section 9.3 of notebook 04) |
| 7 | Error factors and monotonicity | 10^RMSE had been called a "typical error"; the median single error is a factor of 3.3 (article held out), not 8.4. The monotonicity check covered one condition | Corrected | Percentile table added; factor wording corrected; monotonicity check extended to all 1,455 curves with rise sizes, and its scope stated |
| 8 | Reporting | Negative results (H1, H2) kept. One double-rounded interval bound found (LightGBM inflation upper limit 1.4446 quoted as 1.45) | Corrected | Quoted as 1.44 |

**Numbers that did not change.** All primary and sensitivity metrics of version 1. The regenerated predictions are identical for Basquin, Ridge, LightGBM and the residual model and differ by at most 4.4 × 10⁻¹⁵ for Random Forest; every metric file agrees with version 1 to within 2 × 10⁻¹⁵. The processed datasets have the same rows and the same values in every original column.

**Unresolved.** Whether the rows recorded at 25 °C were tested at room temperature (needs the original articles); intervals that include refitting variability (needs repeated grouped cross-validation).

## 10. Technical audit (version 3): what was decided

The technical audit (`reports/independent_technical_audit.md`) found no leakage and no metric error, but listed issues ranked by severity (`reports/critical_issues_and_recommendations.md`). Version 3 resolves them as follows; the evidence for each is in `reports/technical_closure_v3.md`.

| Decision | Reason |
|---|---|
| Notebook 01 downloads both raw files if absent, and rebuilds the provenance record from the public Figshare API record if it is absent; an existing record is never overwritten | A fresh checkout must run without hand-made inputs, while the record of the original download must stay the reference |
| Notebook 02 opens the JSON through `ensure_raw_data` | The file is not in version control |
| The near-duplicate population (cohort with run-outs) is kept and documented; ties are now judged symmetrically | A run-out is a published point of the curve. The tie rule removes an order dependence and leaves the 78 pairs unchanged |
| The life-dependence reading of version 2 is replaced by a calibration analysis | Binning errors by the true life produces the reported pattern for any imperfect predictor |
| The tier-1 gain is attributed by row group, not to Kt | Notched rows give 1–6% of the gain, tests at 1 kHz or above 50–64% |
| UTS conclusions are stated per model with paired intervals | The effect differs in sign between Ridge and the boosted models |
| The pooled-line group is split by the reason for pooling | Rare, absent and "slope not negative" alloys behave differently |
| Test-method strings (`test_type_raw`, `load_control`, `frequency_text`, `test_machine`) are saved as audit columns; `test_machine` is forbidden as an input | Needed for the row-group checks and supplementary investigation B; a machine name can identify a laboratory |
| The dense one-hot encoding is recorded in `model_config.json` | It changes Random Forest predictions at the level of its own randomness |
| The audit's checks are part of the test suite | So that the next change cannot silently break them |

The primary cohort, features, folds, models and primary results are unchanged.

**Supplementary investigations (version 3).**

| Decision | Reason |
|---|---|
| Investigation A: inner folds over the outer training articles use the documented fold rule; the test-time anchor is the Basquin model fitted on all outer training rows; alloys without enough data use the pooled line of each fit | As specified for the investigation; no held-out article ever informs an anchor |
| Investigation B: "ultrasonic" means a recorded frequency of at least 15 kHz; the regime categories come from the test-type strings and are checked against frequency and machine names | No category was assumed: every recorded value of 1 kHz or above lies at 19–20.5 kHz, and the machine names confirm ultrasonic systems |
| Load control is not used as an input | For VHCF tests it is the database's default ("displacement"), not independent information |
| The resonance tag is used only in one ablation, with its incompleteness stated | 678 untagged points ran on resonance machines |
| Ablation models: LightGBM (main learner) and Ridge (linear contrast); seeds 43 and 44 for LightGBM under C | The question is attribution, not an algorithm competition |
| The primary feature set is not changed, although the VHCF tag helps | Changing it after seeing the results would turn a supplementary finding into a post-hoc primary result; a pre-registered rerun is listed as future work |

## 11. Environment

- Windows 11, Python 3.12.4, CPU only. The system Python had no scientific packages; a virtual environment `.venv` was created in the project root and the packages in `requirements.txt` were installed with pip.
- LightGBM 4.7.0 installed without problems, so the `HistGradientBoostingRegressor` fallback was not needed.
- The four notebooks run from start to finish in about eight minutes (version 2: 01 in 24 s, 02 in 13 s, 03 in 427 s, 04 in 25 s).
