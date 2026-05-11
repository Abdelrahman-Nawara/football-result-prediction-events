# Football Match Result Prediction from In-game Events

Machine learning project for **Project 5: Predicting Football Games Result using In-game Events**. The goal is to predict the final result of a football match (`home_win`, `draw`, or `away_win`) using in-game event data observed before a selected match minute.

## Project Highlights

- Uses Wyscout-style event and match JSON data across England, Spain, Italy, Germany, France, the World Cup, and the European Championship.
- Builds event-based features up to a configurable cutoff minute, such as goals at cutoff, shot pressure, shot-quality proxy, pass accuracy, progressive passes, box entries, duels, cards, set pieces, and final-third activity.
- Includes data cleaning, EDA, feature engineering, model training, evaluation, and feature-importance analysis.
- Provides both a professional Python pipeline and an executed notebook for presentation/report use.

## Main Results

Using events up to minute `75` and excluding team IDs to keep the model focused on in-game behavior:

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Softmax Logistic Regression | 0.693 | 0.666 | 0.666 | 0.706 |
| Softmax Logistic Regression, lower L2 | 0.688 | 0.662 | 0.661 | 0.701 |
| Dummy Most Frequent Baseline | 0.461 | 0.333 | 0.210 | 0.291 |

The strongest signals were related to current goals, conversion indicators, assists, duel outcomes, attacking progression, final-third entries, penalty-area pressure, and set pieces.

## Repository Structure

```text
.
├── football_result_prediction.ipynb      # Executed notebook with connected project chunks
├── football_result_prediction.py         # Reusable CLI pipeline and helper functions
├── requirements.txt                      # Python dependencies
├── artifacts/                            # Report-ready generated figures, tables, and model
├── Data set/                             # Dataset files
│   ├── events/                           # Large raw events JSON files tracked with Git LFS
│   ├── matches/                          # Match metadata JSON files
│   └── *.json / *.csv                    # Team, player, tag, competition metadata
├── Project guidelines.pdf
└── data_paper_soccer_nsd.pdf
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If cloning this repository, install Git LFS before pulling the raw event files:

```bash
git lfs install
git lfs pull
```

## Run the Pipeline

Default run, using all competitions and events up to minute 75:

```bash
python3 football_result_prediction.py --cutoff-minute 75
```

Useful alternatives:

```bash
# Half-time prediction
python3 football_result_prediction.py --cutoff-minute 45

# Faster run on one competition
python3 football_result_prediction.py --competitions England --cutoff-minute 75

# Include team IDs as categorical strength/context features
python3 football_result_prediction.py --include-team-ids
```

Generated outputs are written to `artifacts/`.

## Notebook

Open:

```text
football_result_prediction.ipynb
```

The notebook is organized into connected chunks:

1. Setup and configuration
2. Match metadata loading
3. Event cleaning and feature engineering
4. EDA
5. Model training
6. Evaluation
7. Feature importance
8. Report-ready summary

## Data Source

Dataset source: [Soccer match event dataset on Figshare](https://figshare.com/collections/Soccer_match_event_dataset/4415000/3)

Recommended citation:

Pappalardo, L., Cintia, P., Rossi, A. et al. *A public data set of spatio-temporal match events in soccer competitions*. Scientific Data 6, 236 (2019).

## Notes

- The local Anaconda environment used during development had a `scikit-learn` / NumPy binary mismatch. The code automatically falls back to a NumPy softmax logistic regression if `scikit-learn` cannot import.
- In a clean environment with working `scikit-learn`, the script also trains Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting models.
- Large event JSON files exceed GitHub's normal 100 MB file limit, so they are intended to be tracked with Git LFS.

