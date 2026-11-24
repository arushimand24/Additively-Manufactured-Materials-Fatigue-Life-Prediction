# Machine Learning-Based Fatigue Life Prediction of Additively Manufactured Metallic Materials

A research project investigating **fatigue-life prediction for
additively manufactured (AM) metals** and, more importantly, how
reliably machine-learning models generalise to experimental data from
**previously unseen research articles**.

The project uses the literature-derived **FatigueData-AM2022** database
and compares random point-level, curve-held-out, and article-held-out
validation.

------------------------------------------------------------------------

## Overview

Additive manufacturing (also known as metal 3D printing) makes it
possible to manufacture complex components layer by layer. It is
increasingly relevant to aerospace, automotive, biomedical, and energy
applications. However, AM components can contain pores, lack-of-fusion
defects, rough surfaces, and residual stresses. These factors can
initiate cracks and make fatigue behaviour difficult to predict.

**Fatigue** is the progressive damage and eventual failure of a material
subjected to repeated or cyclic loading. In this project,
machine-learning models are developed to estimate the fatigue life of AM
metallic specimens from material, manufacturing, and loading
information.

The central focus is not only to predict fatigue life, but to
investigate whether model performance measured using conventional random
data splitting gives an overly optimistic estimate of performance on
data from entirely unseen research articles.

## Research Problem

The FatigueData-AM2022 database combines experimental observations
extracted from multiple published studies. Individual observations are
not fully independent:

-   Multiple observations can belong to the same stress-life (S--N)
    curve.
-   Multiple curves can originate from the same research article.
-   Measurements from the same source may share material batches,
    machines, specimen geometries, and testing procedures.

If observations from the same curve or article appear in both training
and testing sets, a model may benefit from source-specific similarities.
Consequently, random point-level validation may not represent the
difficulty of predicting observations from a new article.

### Primary research question

**How much does point-level random validation overstate the performance
of machine-learning models compared with curve-held-out and
article-held-out validation, and does a Basquin-based residual model
improve generalisation to unseen articles?**

## Research Objectives

1.  Prepare and analyse literature-derived fatigue data for additively
    manufactured metallic materials.
2.  Develop engineering-based and machine-learning models to predict
    logarithmic fatigue life.
3.  Compare random point-level, curve-held-out, and article-held-out
    validation.
4.  Quantify the change in prediction error as increasingly larger
    groups of related observations are held out.
5.  Investigate whether combining the Basquin fatigue relationship with
    machine learning improves prediction on unseen articles.
6.  Examine model limitations and the implications of grouped validation
    for engineering applications.

## Dataset

This project uses **FatigueData-AM2022**, compiled by Zhang and Xu
(2023).

-   **Dataset:** Fatigue Database of Additively Manufactured Alloys
-   **Source:** Figshare
-   **DOI:**
    [10.6084/m9.figshare.22337629](https://doi.org/10.6084/m9.figshare.22337629)
-   **Dataset paper:** [Zhang & Xu, Scientific Data
    (2023)](https://www.nature.com/articles/s41597-023-02150-x)
-   **Licence:** CC BY 4.0 (as stated in the project report; consult the
    dataset landing page for the applicable terms)

The database contains fatigue observations extracted from published
research articles. The project primarily uses:

-   **S--N sheet:** stress-life observations, including stress amplitude
    and cycles to failure/run-out information.
-   **Parameter sheet:** material, additive-manufacturing, processing,
    and testing descriptors.

The S--N observations are joined with their corresponding experimental
metadata using the dataset/curve identifier. The source article is
identified using article metadata such as DOI.

### Dataset summary reported in the project report

  Description                                    Count
  -------------------------------------- -------------
  Raw S--N observations                         15,146
  Observations after project filtering          12,666
  S--N curves                                    1,533
  Source research articles                         440
  Distinct alloys                                   93
  Prediction target                        `log10(Nf)`

These are the counts reported in the project report for the analysed
dataset. Refer to the audit outputs in the repository when reproducing
or updating the analysis.

### Prediction target

The target is the base-10 logarithm of fatigue life:

\[ y = `\log`{=tex}\_{10}(N_f) \]

where (N_f) is the number of cycles to failure.

A logarithmic target is appropriate because fatigue life can span
several orders of magnitude.

## Methodology

The project follows a sequence of data understanding, preparation, model
development, and evaluation.

1.  **Dataset audit and exploratory data analysis:** inspect the
    available variables, distributions, missing values, alloy
    representation, and source/article grouping.
2.  **Data preparation:** combine S--N observations with the
    corresponding parameter metadata and apply documented quality
    filters.
3.  **Feature engineering:** derive stress-related variables and prepare
    numerical and categorical model inputs.
4.  **Model development:** train engineering-based and machine-learning
    regression models.
5.  **Grouped validation:** evaluate the models using three different
    splitting strategies.
6.  **Performance analysis:** compare prediction errors, investigate the
    effect of source grouping, and interpret the findings.

### Data cleaning and preprocessing

The project report describes filtering unsuitable observations,
including run-outs for the primary failure-only regression, non-positive
measurements, and records with unsuitable or missing stress-ratio
information.

Numerical missing values are handled using training-data-based
imputation. Categorical variables are encoded for model use.
Preprocessing is fitted within the training folds rather than on the
complete dataset.

Run-outs are censored observations: the specimen survived to a stopping
limit, so its exact failure life is unknown. They are excluded from the
primary failure-only regression rather than treated as exact failures.

## Models

The project investigates the following approaches:

  -----------------------------------------------------------------------
  Model                   Description             Purpose
  ----------------------- ----------------------- -----------------------
  Basquin model           Traditional stress-life Engineering baseline
                          relationship, fitted    
                          per alloy where         
                          appropriate             

  Ridge Regression        Linear regression with  Linear ML baseline
                          L2 regularisation       

  Random Forest           Ensemble of decision    Nonlinear tabular-data
                          trees                   model

  LightGBM                Gradient-boosted        Main direct
                          decision trees          machine-learning model

  Basquin-based residual  Basquin prediction      Tests whether physics
  hybrid                  combined with a         anchoring helps
                          machine-learning        generalisation
                          correction              
  -----------------------------------------------------------------------

### Basquin relationship

The Basquin equation describes the relationship between stress amplitude
and fatigue life under high-cycle fatigue conditions. In logarithmic
form, it can be represented as:

\[ `\log`{=tex}*{10}(N_f) = A + B`\log`{=tex}*{10}(`\sigma`{=tex}\_a) \]

where (`\sigma`{=tex}\_a) is stress amplitude and (A) and (B) are fitted
coefficients.

### Basquin-based residual model

The hybrid approach first obtains a Basquin-based prediction and then
uses a machine-learning model to estimate the remaining prediction error
(residual). The predicted residual is combined with the Basquin
prediction.

This allows the model to use a conventional engineering relationship as
a starting point while learning additional patterns from the dataset.

## Validation Strategy

The most important part of this research is the comparison of three
validation strategies.

  -----------------------------------------------------------------------
  Validation strategy     What is held out?       What it evaluates
  ----------------------- ----------------------- -----------------------
  Random point-level      Individual observations Prediction among
                                                  observations that may
                                                  share curves and
                                                  articles with training
                                                  data

  Curve-held-out          Entire S--N curves      Prediction for a curve
                                                  not used during
                                                  training

  Article-held-out        Entire research         Generalisation to
                          articles                observations from an
                                                  unseen source article
  -----------------------------------------------------------------------

### Why article-held-out validation matters

Article-held-out validation ensures that observations from a test
article are not used to train the model. It provides a more demanding
assessment of whether a model can generalise across published sources.

It does **not** prove generalisation to an entirely unseen laboratory. A
research group or laboratory may contribute multiple articles.

## Feature Engineering

The model uses information that is available before the fatigue test,
subject to the feature set and data coverage used in each experiment.

Examples of features include:

-   Logarithm of stress amplitude.
-   Stress ratio (`R`).
-   Alloy/material category.
-   Additive manufacturing process.
-   Surface condition.
-   Heat-treatment or HIP condition.
-   Fatigue test type.
-   Additional variables in extended feature tiers, where sufficiently
    available.

A stress-based Smith-Watson-Topper (SWT) feature is also considered:

\[ `\sigma`{=tex}\_{`\mathrm{SWT}`{=tex}} =
`\sqrt{\sigma_{\max}\sigma_a}`{=tex} \]

The exact feature list used for a particular experiment should be
checked in the corresponding notebook and saved configuration.

Identifiers such as curve ID, article DOI, article title, and author
information are retained for grouping and auditing, **not as predictive
model inputs**.

## Results

The project report records the following LightGBM RMSE values:

  Validation strategy     RMSE (log10 fatigue-life units)
  --------------------- ---------------------------------
  Random point-level                                0.673
  Curve-held-out                                    0.787
  Article-held-out                                  0.924

The article-held-out RMSE is approximately **37.3% higher** than the
random point-level RMSE.

This indicates that the estimated prediction error increases when entire
source articles are excluded from training. The article-held-out result
was also reported to be only modestly better than a simple
mean-prediction baseline (approximately 0.994 RMSE).

The result highlights why validation design matters when working with
literature-derived experimental datasets.

**Interpretation note:** RMSE is measured in `log10(Nf)` units, not in
raw cycles or as a percentage error in fatigue life. The values above
are transcribed from the project report; consult the saved predictions
and evaluation outputs for reproduction and any subsequent updates.

## Repository Structure

The repository is organised around four main notebooks and reusable
Python modules. The structure below is the intended project layout;
adjust filenames if the current repository uses different names.

``` text
FatigueData-AM2022-Research/
│
├── data/
│   ├── raw/                         # Original dataset (not committed)
│   └── processed/                   # Cleaned modelling data and fold assignments
│
├── notebooks/
│   ├── 01_Dataset_Audit_and_EDA.ipynb
│   ├── 02_Data_Preparation_and_Feature_Engineering.ipynb
│   ├── 03_Model_Development_and_Training.ipynb
│   └── 04_Model_Evaluation_and_Research_Findings.ipynb
│
├── src/
│   ├── data_utils.py                # Loading, auditing, and data-join utilities
│   ├── feature_utils.py             # Feature creation and preprocessing helpers
│   ├── model_utils.py               # Basquin and ML model utilities
│   └── evaluation_utils.py          # Metrics and evaluation utilities
│
├── tests/
│   └── test_leakage.py              # Validation and leakage checks
│
├── results/
│   ├── audit/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
│
├── reports/
│   └── results_report.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Notebook workflow

  -----------------------------------------------------------------------------------------
  Notebook                                              Main responsibility
  ----------------------------------------------------- -----------------------------------
  `01_Dataset_Audit_and_EDA.ipynb`                      Inspect the raw dataset, audit its
                                                        structure, and perform exploratory
                                                        analysis

  `02_Data_Preparation_and_Feature_Engineering.ipynb`   Join, clean, filter, and prepare
                                                        the modelling dataset

  `03_Model_Development_and_Training.ipynb`             Build models, run validation
                                                        experiments, and save predictions

  `04_Model_Evaluation_and_Research_Findings.ipynb`     Calculate metrics, compare
                                                        validation strategies, and
                                                        interpret findings
  -----------------------------------------------------------------------------------------

## Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd FatigueData-AM2022-Research
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it:

**Windows**

``` bash
.venv\Scripts\activate
```

**macOS/Linux**

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

If the repository does not yet contain a `requirements.txt`, install the
dependencies specified by the notebooks and project environment before
running them. The project uses the Python scientific-computing and
machine-learning ecosystem, including tools such as pandas, NumPy,
Matplotlib, and scikit-learn; LightGBM is required for the LightGBM
experiments.

### 4. Add the dataset

Download the FatigueData-AM2022 dataset from
[Figshare](https://doi.org/10.6084/m9.figshare.22337629) and place the
required workbook in:

``` text
data/raw/
```

The raw dataset is not included in this repository. Follow the dataset's
licence and attribution requirements.

### 5. Run the notebooks in order

Run the notebooks sequentially:

1.  `01_Dataset_Audit_and_EDA.ipynb`
2.  `02_Data_Preparation_and_Feature_Engineering.ipynb`
3.  `03_Model_Development_and_Training.ipynb`
4.  `04_Model_Evaluation_and_Research_Findings.ipynb`

The later notebooks depend on data, fold assignments, predictions, and
other outputs generated by earlier stages.

## Reproducibility and Leakage Prevention

The project uses the following principles to support fair evaluation:

-   Keep curve and article identifiers for grouping, not as model
    features.
-   Do not use the target (`Nf`) or run-out flag as an input feature.
-   Fit imputation, encoding, scaling, and model steps using training
    data only.
-   Keep all observations from a held-out curve or article out of the
    corresponding training fold.
-   Use consistent rows, features, model settings, and fold definitions
    when comparing validation strategies.
-   Record exclusions and data-cleaning decisions rather than silently
    dropping observations.
-   Preserve fold assignments and out-of-fold predictions so that
    evaluation can be inspected and reproduced.

## Limitations

-   The dataset is derived from published literature rather than a
    single controlled experimental campaign.
-   Experimental procedures, machines, specimen geometries, and
    reporting practices vary between sources.
-   Some material and process descriptors contain missing or
    default-filled values.
-   Important fatigue drivers such as detailed defect populations,
    porosity, surface roughness, and residual stress may not be
    available consistently.
-   Run-outs are excluded from the primary failure-only regression, so
    the long-life region may be affected by this choice.
-   Article-held-out validation evaluates unseen articles, not
    necessarily unseen laboratories.
-   The models are intended for research and preliminary screening; they
    do not replace physical fatigue testing or establish certified
    design allowables.
-   Predictions outside the represented alloy, process, stress, and
    testing ranges should be treated cautiously.

## Future Work

Potential extensions include:

-   Censored-data and accelerated-failure-time approaches to incorporate
    run-outs.
-   Author- or research-group-held-out validation.
-   Temporal validation using earlier articles for training and later
    articles for testing.
-   Sensitivity analysis for default-filled and missing metadata.
-   Uncertainty estimation and conformal prediction.
-   Further model interpretation using SHAP and partial-dependence
    analysis.
-   Validation on independent experimental datasets.

## References

1.  Zhang, Z. and Xu, Z. (2023). **Fatigue database of additively
    manufactured alloys.** *Scientific Data*, 10, 249.
    <https://doi.org/10.1038/s41597-023-02150-x>
2.  **FatigueData-AM2022 dataset.** Figshare.
    <https://doi.org/10.6084/m9.figshare.22337629>
3.  Kapoor, S. and Narayanan, A. (2023). **Leakage and the
    reproducibility crisis in machine-learning-based science.**
    *Patterns*, 4(9), 100804.
4.  Di Maggio, et al. (2025). Research on machine-learning-based fatigue
    prediction using AM fatigue data. *Engineering with Computers*.
    <https://doi.org/10.1007/s00366-025-02139-7>
5.  Ahn, et al. (2026). Research on machine-learning-based fatigue-life
    prediction. *Applied Sciences*, 16(13), 6493.
    <https://doi.org/10.3390/app16136493>

## Authors

**Arushi Mandal**\
**Hiti Panghal**

B.Tech\
Indira Gandhi Delhi Technical University for Women (IGDTUW)

------------------------------------------------------------------------
