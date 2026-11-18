# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap reconstructs realistic vehicle trajectories by matching noisy GPS
observations to a road network. This repository implements the complete pipeline:
data processing, feature engineering, candidate generation, multi-model
classification, Bayesian hyperparameter optimisation, HMM/Viterbi route
reconstruction, a FastAPI backend and an evaluator-facing web UI.

## Repository layout

```
paths snap/
├── pathsnap/            # core library
│   ├── io.py            # raw trip file loading
│   ├── geo.py           # projection / distance / bearing helpers
│   ├── candidates.py    # candidate road-segment generation (spatial index)
│   ├── features.py      # feature engineering
│   ├── labeling.py      # reference-route alignment (ground-truth labels)
│   ├── dataset.py       # dataset assembly + caching
│   ├── models.py        # model catalogue (6 models, 4 families)
│   ├── evaluate.py      # metrics (Top-1, Macro F1, Log Loss, continuity)
│   ├── hmm.py           # HMM + Viterbi route reconstruction
│   └── pipeline.py      # end-to-end inference
├── scripts/
│   ├── eda.py           # exploratory data analysis + Folium maps
│   ├── build_data.py    # build + cache train/student-test datasets
│   ├── train_models.py  # train/compare 6 models (group-aware CV)
│   ├── bayes_opt.py     # Bayesian hyperparameter optimisation (Optuna)
│   ├── evaluate_hmm.py  # classifier-only vs classifier+HMM comparison
│   └── make_report.py   # figures + report generation
├── api/main.py          # FastAPI backend (Laptop 1)
├── ui/index.html        # evaluator web UI (Laptop 2)
├── experiments/         # cached datasets, logs, results
├── artifacts/models/    # saved model weights
└── report/              # technical report + figures + maps
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. exploratory data analysis
PYTHONPATH=. python3 scripts/eda.py

# 2. build cached datasets (80 train + 10 student-test trips)
PYTHONPATH=. python3 scripts/build_data.py

# 3. train & compare models
PYTHONPATH=. python3 scripts/train_models.py

# 4. Bayesian hyperparameter optimisation (on the selected model)
PYTHONPATH=. python3 scripts/bayes_opt.py --model LightGBM --trials 20

# 5. HMM/Viterbi evaluation
PYTHONPATH=. python3 scripts/evaluate_hmm.py --model LightGBM_tuned
```

## Deployment (two laptops)

**Laptop 1 — API** (hosts the model, HMM and road-network data):

```bash
PATHSNAP_DATA_DIR=/path/to/evaluator/trips \
PATHSNAP_MODEL=artifacts/models/LightGBM_tuned.joblib \
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Laptop 2 — UI**: open `ui/index.html` (or serve it with `python3 -m http.server`),
select an evaluation trip and click *Predict route*. The UI sends the trip id to
the API, which runs the full pipeline and returns the predicted route, reference
route and evaluation results.

## Data

Each trip directory contains four files:

| file    | contents |
|---------|----------|
| `.track`| GPS observations: `longitude latitude timestamp(s)` (~1 Hz) |
| `.nodes`| road-network nodes: `longitude latitude` (row index = node id) |
| `.arcs` | road segments: `from_node to_node` (row index = arc id) |
| `.route`| reference route: chronological sequence of arc ids (ground truth) |

The dataset spans 80 training trips and 10 student-test trips across cities
world-wide. The 10 evaluator-testing trips are reserved for final evaluation only
and are never used for training or model selection.

## Results

Six models across four families were trained with 5-fold group-aware CV and
evaluated on the 10 student-test trips (observation-level Top-1 accuracy is the
primary metric):

| Model | Family | CV Macro F1 | Top-1 | Macro F1 | Log Loss |
|-------|--------|-------------|-------|----------|----------|
| Logistic Regression | Linear / probabilistic | 0.797 | 0.814 | 0.791 | 0.264 |
| Random Forest | Ensemble (bagging) | 0.854 | 0.823 | 0.861 | 0.176 |
| HistGradientBoosting | Boosting | 0.846 | 0.822 | 0.852 | 0.187 |
| XGBoost | Boosting | 0.849 | 0.824 | 0.853 | 0.184 |
| **LightGBM** | Boosting | 0.847 | **0.824** | 0.852 | 0.187 |
| MLP | Neural network | 0.866 | 0.820 | 0.878 | 0.123 |

**Final system** = LightGBM classifier + HMM/Viterbi decoding:

| Stage | Top-1 | Route continuity |
|-------|-------|------------------|
| Classifier only | 0.842 | 0.941 |
| + HMM/Viterbi | **0.920** | **0.973** |

Bayesian hyperparameter optimisation (Optuna TPE, 20 trials, CV Top-1 objective)
improved the CV objective (0.769 → 0.776) but did not transfer to student-test
Top-1 (a train/test distribution gap from a minority of noisy training trips), so
the simpler, faster untuned LightGBM is deployed — the decision is documented in
the technical report.
