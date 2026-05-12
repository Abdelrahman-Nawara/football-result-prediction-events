# Football Match Result Prediction

Machine learning project for predicting the final result of a football match (`home_win`, `draw`, or `away_win`) from in-game events and pre-match context.

The main workflow is now contained in one standalone notebook:

```text
football_result_prediction.ipynb
```

The notebook does not import `football_result_prediction.py`. It includes the data loading, cleaning, feature engineering, EDA, cutoff comparison, model training, model evaluation, and feature-importance workflow directly in the notebook cells.

## What The Notebook Uses

- Wyscout-style event files across England, Spain, Italy, Germany, France, the World Cup, and the European Championship.
- Match metadata from `Data set/matches/`.
- Team names from `teams.json`.
- Player profile features from `players.json`.
- Prior historical PlayerRank features from `playerank.json`.
- Main-referee IDs and prior referee-history features built from match/event history.
- Event-derived live-match features up to cutoff minutes `45`, `55`, `65`, `75`, and `80`.

## Latest Results

The executed notebook compares all requested cutoff times and includes XGBoost. The best holdout result was:

| Cutoff | Best model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---:|---|---:|---:|---:|---:|
| 80 | XGBoost | 0.773 | 0.742 | 0.741 | 0.774 |

Best model by cutoff:

| Cutoff | Best model | Accuracy | Macro F1 |
|---:|---|---:|---:|
| 45 | XGBoost | 0.634 | 0.556 |
| 55 | XGBoost | 0.657 | 0.598 |
| 65 | XGBoost | 0.706 | 0.664 |
| 75 | XGBoost | 0.750 | 0.714 |
| 80 | XGBoost | 0.773 | 0.741 |

Full tables are written to:

```text
artifacts/tables/cutoff_model_comparison.csv
artifacts/tables/best_model_by_cutoff.csv
artifacts/tables/model_comparison.csv
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

If cloning this repository, install Git LFS before pulling the raw event files:

```bash
git lfs install
git lfs pull
```

## Run

Open and run:

```text
football_result_prediction.ipynb
```

The notebook will:

1. Load the dataset from `Data set/`.
2. Build event features for every requested cutoff in one pass over the event files.
3. Add team-form, player-profile, prior PlayerRank, and referee-history features.
4. Run EDA and save figures/tables.
5. Train Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, Dummy baseline, and XGBoost.
6. Select the best cutoff/model and save the final model artifact.

## Key Artifacts

```text
artifacts/models/best_football_result_model.joblib
artifacts/figures/cutoff_model_macro_f1.png
artifacts/figures/best_model_confusion_matrix.png
artifacts/figures/permutation_feature_importance.png
artifacts/tables/engineered_match_features.csv
artifacts/tables/engineered_match_features_all_cutoffs.csv
artifacts/tables/notebook_run_summary.json
```

## Data Source

Dataset source: [Soccer match event dataset on Figshare](https://figshare.com/collections/Soccer_match_event_dataset/4415000/3)

Recommended citation:

Pappalardo, L., Cintia, P., Rossi, A. et al. *A public data set of spatio-temporal match events in soccer competitions*. Scientific Data 6, 236 (2019).

## Notes

- `referees.json` in this repo appears truncated, so the notebook safely continues without parsing that file and instead uses referee IDs from match metadata plus prior referee-history features computed from matches/events.
- The old Python script is no longer required by the notebook; it can remain as a reference, but the main deliverable is the standalone executed notebook.
- Large event JSON files are tracked through Git LFS.
