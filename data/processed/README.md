# data/processed

This folder is empty at first. The notebooks fill it:

- notebook 02 writes `modelling_dataset.csv` (the cleaned table the models learn from, 12,666 rows),
  `sn_cohort_with_runouts.csv` (the same cohort with the run-out tests kept) and `feature_config.json`;
- notebook 03 writes `folds.csv` (which cross-validation fold every row belongs to, for each validation scheme).

You can open the CSV files in Excel to look at them. If you change and save one, run notebook 02 (and 03) again
to restore it.
