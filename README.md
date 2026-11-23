# Sri Lankan Household Electricity Consumption Prediction Using Ensemble Learning

Predicting a Sri Lankan household's **next monthly electricity bill (kWh)** with **Bagging**, **Boosting** and
**Stacking** ensembles, on the **RECON-SL** dataset (LIRNEasia & Lanka Electricity Company), and comparing them
honestly against simple baselines. Includes a Streamlit dashboard.

> University assignment (Ensemble Learning). AI assistance is disclosed in [`AI_DECLARATION.md`](AI_DECLARATION.md).

---

## 1. Research objective

**Primary question:** How effectively can Bagging, Boosting and Stacking predict monthly residential electricity
consumption from historical consumption and household characteristics?

Secondary questions: which technique has the lowest error, whether survey (household) information improves
predictions, which features matter most, how ensembles compare with baselines, and how performance changes when
predicting future months.

**Task:** supervised regression. For household *h* and billing month *t*, predict the bill (kWh) using only information
available at the start of month *t* (bills up to *t − 1*). This is one-month-ahead prediction for existing customers.

## 2. Dataset — RECON-SL

| | |
|---|---|
| Name | Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL) |
| Publisher | LIRNEasia, in collaboration with Lanka Electricity Company (LECO) |
| Access | Zenodo record [15023048](https://zenodo.org/records/15023048) (also IEEE DataPort, Kaggle); DOI [10.21227/n1dk-q860](https://doi.org/10.21227/n1dk-q860) |
| Licence | **CC-BY-4.0** (attribution required) |
| Paper | Algama et al., *Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL)*, arXiv:2609.28783 |
| Used here | Monthly bills of 4,063 households, Oct 2022 – Oct 2024 (25 billing months); Wave 1 survey; 15-minute smart-meter data (analysis only) |
| Unit | **kWh**. The data dictionary says "kilowatts (KW)", but the dataset paper states kilowatt-hours, and bills match smart-meter kWh registers (median ratio 0.993). |
| Scope | Households in **LECO service areas along Sri Lanka's western/south-western coast**. Not representative of the whole country. |

The raw data (≈1.2 GB zipped) are **not** stored in this repository. See *Reproduce the results*.

## 3. Method

**Data cleaning** (`src/data_loader.py`, `src/data_processing.py`, notebook 02)
- Values such as `"1,159"` stored with thousands separators are parsed correctly. A naive conversion would silently
  delete the largest consumers.
- Missing bills stay missing and are never treated as zero.
- 15 households billed 0 kWh in every month are removed as **invalid records**. In the survey they were occupied
  homes with a refrigerator, and their linked smart meters also read 0.
- Zero months and extreme values are kept. 18 strictly isolated one-month spikes are only flagged.

**Features** (`src/feature_engineering.py`, notebook 03). All features are built from bills up to *t − 1*:
- lags 1, 2, 3, 6 and 12;
- 3- and 6-month rolling mean, a 3-month rolling median and a 6-month rolling standard deviation;
- the long-run mean and maximum;
- the most recent change;
- counts of zero and missing bills;
- the calendar month (cyclic).

Optional Wave 1 survey features (appliances, household size, house type, socio-economic class, …) are used **only
after each household's interview date**. Waves 2 and 3 were collected after the consumption data end and are never
used.

**Evaluation design** (`src/validation.py`)
- Training: billing months Jan 2023 – May 2024. **Untouched test period: Jun – Oct 2024**, predicted one month ahead.
- Model selection and tuning: 5 expanding-window folds inside the training period. Each fold trains on earlier
  months and validates on the next month (Jan … May 2024).
- **Selection rule, fixed in advance:** lowest mean validation MAE. The test set is never used for choices.
- Extra experiments: history-only vs history + survey; unseen households (GroupKFold); robustness without the 34
  low-activity/invalid households and without isolated spikes.
- **Leakage checks:** 13 checks on the real data (notebook 03) plus automated tests. For example, changing future
  bills never changes earlier features, and survey data are hidden before the interview date.

**Models** (`src/models.py`, `src/ensembles.py`)

| Family | Models |
|---|---|
| Baselines | Naive last month, naive 3-month mean, DummyRegressor, Linear Regression, Ridge |
| Bagging | Random Forest, Extra Trees |
| Boosting | Gradient Boosting, XGBoost (HistGradientBoosting tried untuned) |
| Combinations | VotingRegressor; **time-ordered Stacking** (custom: sklearn's StackingRegressor would let base models learn from future months) with a Ridge meta-learner |

Hyper-parameters, including the **loss function** (squared / absolute / Huber error) and an optional log target,
were tuned with `RandomizedSearchCV` on the time-ordered folds (`scripts/tune_models.py`).

## 4. Results

All errors are **kWh per household per month**. "Val" is the mean of the 5 validation folds; "Test" is Jun–Oct 2024
(20,094 household-months).

| Rank | Model | Family | Val MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| 1 | **Voting Regressor (tuned)** | Combination | 18.78 | 19.67 | 44.99 | 0.882 |
| 2 | Gradient Boosting (tuned) | Boosting | 18.91 | 20.02 | 45.49 | 0.880 |
| 3 | Stacking (time-ordered, Ridge meta-learner) (tuned) | Combination | 18.98 | 19.63 | 43.83 | 0.888 |
| 4 | XGBoost (tuned) | Boosting | 19.05 | 20.37 | 50.09 | 0.854 |
| 5 | Random Forest (tuned) | Bagging | 19.23 | 19.60 | 43.68 | 0.889 |
| 6 | Extra Trees (tuned) | Bagging | 19.24 | 20.06 | 44.09 | 0.887 |
| 7 | Naive: last month's bill | Baseline | 19.69 | 19.49 | 48.84 | 0.861 |
| 8 | Ridge Regression | Baseline | 20.81 | 19.89 | 43.41 | 0.890 |
| 9 | Linear Regression | Baseline | 20.91 | 20.62 | 44.17 | 0.886 |
| 10 | Naive: mean of last 3 months | Baseline | 21.73 | 20.25 | 46.21 | 0.876 |
| 11 | DummyRegressor (training mean) | Baseline | 73.47 | 74.19 | 131.37 | -0.004 |

**Selected model (rule fixed in advance: lowest validation MAE): Voting** (tuned Gradient Boosting + XGBoost + Random
Forest), saved as `models/best_model.joblib`.

**Key findings**
- **Ensembles vs the naive forecast.** On validation the ensembles beat "same as last month" (18.78 vs 19.69 kWh MAE).
  On the test period **none is significantly better on MAE**: a household-bootstrap 95% interval for Voting minus
  naive is −0.20 to +0.58 kWh. The ensembles do reduce large errors (RMSE 43.7–45.0 vs 48.8 kWh).
- **Survey information does not help** once bill history is available. Voting scores 19.57 kWh test MAE with
  Wave 1 features vs 19.67 without; Ridge gets worse.
- **Unseen households:** Voting scores 19.79 kWh on households never seen in training, vs 19.67 on known ones.
- **Robustness:** without the 34 low-activity/invalid households, Voting scores 19.76 vs naive 19.58.
  Without isolated spikes in training it scores 19.65.
- **Most important feature:** last month's bill. Shuffling it raises test MAE by about 37 kWh.
- **Post-hoc finding (not used for selection):** the tree models under-predicted October 2024 by about 16–19 kWh,
  because the training data contain only one October (2023), which was unusually low. Without calendar features,
  Gradient Boosting reaches test MAE 18.16 kWh, but its validation MAE is worse
  (19.49). With about two years of data, seasonal patterns cannot be learned reliably.

Full analysis: `notebooks/07_model_evaluation.ipynb`; interactive results: the dashboard.

## 5. Repository structure

```
app.py                     Streamlit dashboard entry point
dashboard/                 dashboard pages (overview, exploration, comparison, performance, prediction)
src/                       data loading, cleaning, features, preprocessing, models, stacking, evaluation, prediction
scripts/                   reproducible pipeline: audit, processed data, tuning, experiments, final model
notebooks/01–07            dataset inspection → EDA → features & leakage audit → baselines → ensembles → tuning → evaluation
tests/                     pytest suite (parsing, cleaning, leakage, preprocessing, prediction, dashboard)
data/processed/            small cleaned datasets used by the dashboard (raw data not included)
models/                    best_model.joblib (final pipeline + metadata), ridge_baseline.joblib
artifacts/                 saved results: model_results.csv, cross_validation_results.csv, test_predictions.csv,
                           feature_importance.csv, experiment_results.csv, model_metadata.json, tuning/, experiments/
AI_DECLARATION.md          AI usage declaration
```

## 6. Installation

Python 3.13 was used.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt     # dashboard + notebooks + tests
```

## 7. Run the dashboard

```bash
streamlit run app.py
```

The dashboard only loads saved results and `models/best_model.joblib`. It never trains a model.

## 8. Reproduce the results

1. Download `data.zip` from [Zenodo record 15023048](https://zenodo.org/records/15023048) (≈1.2 GB; MD5
   `dcdac5809b4da6b6362519d33112a5f9`). Extract its `data/` folder so that you get
   `data/raw/consumption_data/...` and `data/raw/survey_data/...`.
2. Run, from the project root:

```bash
python scripts/audit_smart_meter.py            # audit of smart-meter files (~10 min)
python scripts/build_processed_data.py         # cleaned panel + Wave 1 household features
python scripts/build_smart_meter_summaries.py  # 15-minute summaries (~10 min)
python scripts/tune_models.py                  # hyper-parameter searches (~40 min)
python scripts/run_experiments.py              # tuned models, voting, stacking, extra experiments (~2 h)
python scripts/run_experiments.py diagnostic   # post-hoc diagnostic (see section 4)
python scripts/save_final_model.py             # saves models/ and artifacts/
pytest -q                                      # test suite
```

3. Notebooks: open `notebooks/` in VS Code or Jupyter and run them in order (01 → 07). They read the saved
   artifacts; set `RERUN = True` in notebook 05 to retrain from scratch.

All randomness uses `random_state=42`.

## 9. Deploy to Streamlit Community Cloud

1. Push this repository to GitHub, including `models/`, `artifacts/` and `data/processed/`. The raw data are
   excluded by `.gitignore`. The largest file is `models/best_model.joblib` (≈21 MB), below GitHub's 100 MB limit,
   so Git LFS is not needed.
2. Go to <https://share.streamlit.io> → **Create app** → choose the repository and branch, with main file `app.py`.
3. Under **Advanced settings**, choose **Python 3.13**. This must match the versions pinned in `requirements.txt`,
   because the saved model has to be loaded with the same scikit-learn/xgboost versions.
4. Deploy.

## 10. Limitations

- The households come from LECO service areas only (western/south-western coast) and are not nationally
  representative.
- Only 25 billing months exist, so seasonality cannot be estimated reliably (see the October 2024 finding).
- The model needs a household's recent bills and is not designed for new households without history.
- The billing month mostly reflects the previous calendar month's usage.
- Survey information is only available for a small part of the training period when used without leakage.
- Smart-meter data cover about a third of the panel and have a gap in early 2024. They are not used as model inputs.

## 11. AI assistance

Developed with the help of **Claude Code (Anthropic)**. How AI was used is stated in
[`AI_DECLARATION.md`](AI_DECLARATION.md).

## 12. Citation

If you use the data, cite: LIRNEasia, *Residential Electricity Consumption: Dataset Combining Multi-round Longitudinal
Surveys and Energy Provider Data*, Zenodo, 2025, DOI 10.21227/n1dk-q860 (CC-BY-4.0).
