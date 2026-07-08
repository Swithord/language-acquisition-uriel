# Language Similarity Measures and Human Language Acquisition

This repository studies what language similarity measures capture when evaluated against human language acquisition outcomes.

The central distinction is between:

* **language similarity as a construct**, meaning a latent notion of how similar or distant two languages are;
* **language similarity measures as operationalisations**, meaning concrete functions that return numeric distance scores.

The project asks whether commonly used distance measures can support claims about the underlying construct, or whether they should be interpreted more narrowly as task-specific predictive features.

## Objective

Language distance measures are widely used in multilingual NLP and language acquisition research. They are often used for downstream prediction, model selection, transfer-language selection, or interpretation of empirical results.

However, researchers often move from statements about a specific distance measure to statements about language similarity itself. For example, a paper may use one typological distance measure, find that it predicts transfer performance, and then conclude that structural language similarity determines transfer success.

This project evaluates whether such claims are justified.

We use human language acquisition outcomes as an external criterion. The goal is to test whether distance measures agree with each other, whether they relate to speaking outcomes, whether they predict held-out language performance, and whether their apparent effects survive adjustment for learner and country-level confounding.

## Main research questions

The analysis is organised around four questions.

### 1. Do operationalisations of the same modality agree?

For each distance family, we ask whether measures that claim to represent the same modality behave similarly.

Examples:

* Do typological measures agree with other typological measures?
* Do genetic measures agree with other genetic measures?
* Does speaker-distribution geographic distance behave like URIEL or lang2vec geographic distance?

This is tested using within-family rank correlations and principal-component summaries.

### 2. Are language distances related to acquisition outcomes?

We test whether each distance measure is associated with speaking score.

The expected direction is:

larger distance → lower speaking score.

This is evaluated using standardised regression models, separately and jointly across distance families.

The core analysis is run in an apples-to-apples way across TOEFL and STEX using only the shared base columns.

### 3. Are language distances useful for held-out prediction?

Language distances are often used in cross-lingual transfer as predictive features. The predictive analysis mirrors that use case.

We evaluate whether distance measures improve prediction of held-out language-pair performance using grouped cross-validation.

The main comparison is between:

* a mean baseline;
* the best single distance measure selected inside each training fold;
* modality-level distance summaries;
* all distance measures in a regularised regression model.

### 4. Are distance effects robust to confounding?

For STEX only, we use extended learner and country-level covariates to test whether distance effects persist after adjustment.

This analysis asks whether language distance contributes information beyond:

* learner background;
* country-level opportunity variables;
* auxiliary-language background;
* native-auxiliary and auxiliary-target distances.

If distance effects attenuate strongly after adjustment, then the measure may be predictive mainly because it proxies for exposure, geography, educational opportunity, or auxiliary-language knowledge.

## Datasets

The repository assumes three input files.

### `data/toefl_base.csv`

TOEFL speaking-score data with one row per observation.

Main columns:

* `dataset`
* `observation`
* `native_language`
* `native_code`
* `target_language`
* `target_code`
* `speaking_score`
* shared language distance columns

The TOEFL target language is English.

### `data/stex_base.csv`

STEX speaking-score data with the same core columns as TOEFL.

Main columns:

* `dataset`
* `observation`
* `native_language`
* `native_code`
* `target_language`
* `target_code`
* `speaking_score`
* shared language distance columns

STEX may contain repeated individual observations for the same native-target language pair. The core analysis aggregates STEX to the language-pair level before fitting language-distance models.

### `data/stex_extended.csv`

STEX-only extended covariates.

This file is merged to `stex_base.csv` by `observation`.

It contains learner-level, country-level, and auxiliary-language variables, including:

* `age_at_arrival`
* `residence_length`
* `target_education_days`
* `sex`
* `country`
* `country_code`
* `hdi`
* `gdp_per_capita`
* `adult_literacy`
* `internet_use`
* `education_expenditure`
* `tertiary_enrolment`
* `auxiliary_language`
* `auxiliary_target_distance_*`
* `native_auxiliary_distance_*`

These columns are used only in the STEX extended analysis.

## Distance families

The core distance columns are grouped into pre-specified modality families.

### Lexical

* `asjp_lexical`
* `concepts_lexical`

### Genetic

* `glottolog_genetic`
* `l2v_genetic`
* `urielplus_genetic`
* `modality_hyperbolic`

`modality_hyperbolic` is treated as a genetic distance.

### Geographic

* `l2v_geographic`
* `urielplus_geographic`
* `modality_speaker`

`modality_speaker` is treated as a geographic distance.

### Typological / structural

* `grambank_typological`
* `qwals_typological`
* `l2v_featural`
* `l2v_syntactic`
* `urielplus_featural`
* `urielplus_syntactic`
* `urielplus_morphological`
* `modality_island`

`modality_island` is treated as a typological or structural distance.

### Phonological / inventory

* `l2v_inventory`
* `l2v_phonological`
* `phoible_phonological`
* `urielplus_inventory`
* `urielplus_phonological`

### Script

* `scripts_script`
* `urielplus_script`

## Repository structure

```text
.
├── main.py
├── extended.py
├── requirements.txt
├── slurm/
│   ├── run_main.sbatch
│   └── run_extended.sbatch
└── src/
    └── langsim/
        ├── __init__.py
        ├── config.py
        ├── dataloader/
        │   ├── __init__.py
        │   ├── core.py
        │   └── extended.py
        ├── models/
        │   ├── __init__.py
        │   ├── base.py
        │   └── transformers.py
        ├── fitting/
        │   ├── __init__.py
        │   ├── agreement.py
        │   ├── inference.py
        │   ├── prediction.py
        │   └── stex_extended.py
        ├── evaluate/
        │   ├── __init__.py
        │   ├── metrics.py
        │   └── plotting.py
        └── utils/
            ├── __init__.py
            ├── io.py
            ├── preprocess.py
            └── stats.py
```

## Core analysis

The core analysis is run by `main.py`.

It uses only the shared base columns in `toefl_base.csv` and `stex_base.csv`. This allows an apples-to-apples comparison between TOEFL and STEX.

The core analysis performs:

1. coverage checks for each distance measure;
2. within-family agreement analysis;
3. single-distance inferential models;
4. joint within-family inferential models;
5. modality-level inferential models;
6. grouped predictive evaluation.

By default, repeated native-target observations are aggregated to the language-pair level. This is important because language distance values are pair-level quantities. Treating repeated STEX rows as independent would inflate precision.

### Core outputs

The core analysis writes:

```text
artifacts/core/
├── analysis_data.csv
├── metadata.json
├── coverage.csv
├── distance_agreement.csv
├── single_distance_effects.csv
├── family_joint_effects.csv
├── modality_effects.csv
├── predictive_per_fold.csv
├── predictive_summary.csv
└── plots/
    ├── distance_spearman_heatmap.png
    ├── single_effects_forest.png
    └── predictive_summary.png
```

The most important files are:

* `distance_agreement.csv`: whether measures within the same modality agree;
* `single_distance_effects.csv`: measure-level associations with speaking score;
* `family_joint_effects.csv`: whether distance families have joint effects;
* `modality_effects.csv`: which modality-level summaries matter;
* `predictive_summary.csv`: held-out predictive value of the distance measures.

## Extended STEX analysis

The extended analysis is run by `extended.py`.

It is STEX-specific and uses both:

* `data/stex_base.csv`
* `data/stex_extended.csv`

The purpose is to test whether language-distance effects remain after adjustment for learner, country, and auxiliary-language covariates.

The extended analysis performs:

1. merge of base and extended STEX data by `observation`;
2. construction of learner, country, and auxiliary-language adjustment blocks;
3. unadjusted versus adjusted distance-effect comparison;
4. grouped predictive evaluation with and without confounders.

### Extended outputs

The extended analysis writes:

```text
artifacts/stex_extended/
├── analysis_data.csv
├── metadata.json
├── distance_effect_attenuation.csv
├── predictive_per_fold.csv
└── predictive_summary.csv
```

The most important files are:

* `distance_effect_attenuation.csv`: how much distance effects change after confounder adjustment;
* `predictive_summary.csv`: whether distances improve prediction beyond confounders.

## Installation

Create and activate a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set the Python path.

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Running locally

### Run the core analysis

```bash
python main.py \
  --csv data/toefl_base.csv data/stex_base.csv \
  --outdir artifacts/core \
  --make_plots
```

### Run the STEX extended analysis

```bash
python extended.py \
  --stex_base data/stex_base.csv \
  --stex_extended data/stex_extended.csv \
  --outdir artifacts/stex_extended
```

## Running on a cluster with Slurm

### Core analysis

```bash
sbatch slurm/run_main.sbatch
```

Optional environment overrides:

```bash
OUT_DIR=artifacts/core_v1 \
N_SPLITS=5 \
RANDOM_STATE=42 \
sbatch slurm/run_main.sbatch
```

To run the core analysis on a different set of core-format CSVs:

```bash
CSVS="data/toefl_base.csv data/stex_base.csv data/other_base.csv" \
OUT_DIR=artifacts/core_with_other \
sbatch slurm/run_main.sbatch
```

### STEX extended analysis

```bash
sbatch slurm/run_extended.sbatch
```

Optional environment overrides:

```bash
STEX_BASE=data/stex_base.csv \
STEX_EXTENDED=data/stex_extended.csv \
OUT_DIR=artifacts/stex_extended_v1 \
N_SPLITS=5 \
RANDOM_STATE=42 \
sbatch slurm/run_extended.sbatch
```

## Interpretation guide

The analysis is designed to distinguish three possible conclusions.

### Strong construct support

This occurs if:

* measures within the same modality agree;
* their effects on speaking score are in the expected direction;
* the effects replicate across TOEFL and STEX;
* distance measures improve held-out prediction;
* STEX effects remain after adjustment for confounders.

In this case, the results provide evidence that the operationalisations capture something about the underlying language-distance construct.

### Operationalisation-specific support

This occurs if:

* some individual measures predict acquisition outcomes;
* other measures in the same modality do not;
* within-family agreement is weak or mixed.

In this case, claims should be made about specific measures, not about the modality as a whole.

### Weak construct support

This occurs if:

* measures within a modality disagree;
* effects are weak, unstable, or directionally incoherent;
* predictive gains are small;
* effects attenuate strongly after confounder adjustment.

In this case, language distance measures may still be useful as task-specific features, but they should not be treated as direct evidence about inherent language similarity.

## Notes on design choices

The core analysis is intentionally simple. The goal is not to maximise predictive performance with a complex model. The goal is to evaluate whether language distance measures have coherent, replicable, and interpretable relationships with acquisition outcomes.

The extended analysis is STEX-only because TOEFL does not have the same learner-level and country-level covariates. This separation keeps the TOEFL/STEX comparison clean while still allowing a richer confounding analysis where the data permit it.

Grouped cross-validation is used because language-distance values are attached to language pairs. Random row-level splits would leak the same language-pair distance profile across training and test sets, especially in STEX.

## Expected input format for new core datasets

A new dataset can be included in the core analysis if it has the same essential structure:

```text
dataset
observation
native_language
native_code
target_language
target_code
speaking_score
<distance columns>
```

The dataset does not need to contain every configured distance column. Missing distance columns are skipped automatically.

The outcome column can be changed with:

```bash
python main.py \
  --csv data/new_base.csv \
  --outcome_col speaking_score \
  --outdir artifacts/new_core
```
