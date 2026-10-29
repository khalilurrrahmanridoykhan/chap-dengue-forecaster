# Climate-Informed Dengue Forecaster for DHIS2 CHAP

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-DHIS2%20CHAP-orange.svg)](https://chap.dhis2.org)

An open-source, climate-informed Dengue outbreak prediction model developed for the **Climate Health Analytics Platform (CHAP)** coordinated by the **HISP Centre, University of Oslo**.

---

## 1. Overview & Epidemiological Context

Dengue virus transmission is heavily regulated by climate dynamics:
- **Rainfall Lags (1–3 Months)**: High rainfall expands surface water pooling and artificial container breeding habitats for *Aedes aegypti* and *Aedes albopictus* mosquitoes.
- **Thermal Acceleration**: Ambient temperatures above 26°C shorten both the mosquito gonotrophic cycle and the extrinsic incubation period (EIP) of the virus within the vector.
- **Disease Persistence**: Local viral reservoirs and human mobility sustain localized transmission chains.

This model captures these non-linear temporal dynamics using **lagged meteorological covariates**, **seasonal harmonic encoding**, and an **L2-regularized log-linear regressor** that guarantees strictly non-negative disease case predictions across heterogeneous administrative health regions (`orgUnit`).

---

## 2. Model Architecture & Features

The model constructs an engineered feature vector for each administrative location:

| Feature Category | Variables | Description |
| :--- | :--- | :--- |
| **Direct Meteorological** | `rainfall`, `mean_temperature` | Concurrent monthly rainfall (mm) and mean temperature (°C). |
| **Precipitation Lags** | `rainfall_lag_1`, `rainfall_lag_2` | 1-month and 2-month prior rainfall to capture larval maturation cycles. |
| **Temperature Lags** | `temp_lag_1` | 1-month prior temperature governing adult vector density. |
| **Autoregressive State** | `cases_lag_1` | Prior month case count (dynamically updated in forward forecasting). |
| **Seasonal Harmonics** | `sin_month`, `cos_month` | Cyclical Fourier encoding ($\sin(2\pi m / 12), \cos(2\pi m / 12)$). |

Predictions are computed in log-space ($y = \log(cases + 1)$) and inverted via $\max(0, \exp(\hat{y}) - 1)$ to enforce realistic epidemiological counts without negative case artifacts.

---

## 3. CHAP Compatibility Contract (`MLproject`)

This repository conforms strictly to the [chap-core](https://github.com/dhis2-chap/chap-core) execution contract:

* **Training (`train.py`)**:
  ```bash
  python train.py <path_to_train_csv> <model_output_path>
  ```
* **Inference (`predict.py`)**:
  ```bash
  python predict.py <model_path> <historic_data_csv> <future_climate_csv> <output_csv>
  ```

---

## 4. Quickstart: Local Evaluation with `chap-core`

To evaluate this model against standard benchmark datasets using the official CHAP evaluator:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run local evaluation on benchmark dataset
chap eval --model-name . --dataset-csv example_data/laos_subset.csv
```

---

## 5. Repository Layout

```text
chap-dengue-forecaster/
├── MLproject            # Standard MLflow definition for CHAP engine
├── pyproject.toml       # Modern packaging configuration
├── requirements.txt     # Python dependencies
├── LICENSE              # Apache License 2.0
├── CITATION.cff         # Academic and Zenodo citation metadata
├── README.md            # Documentation and scientific methodology
├── train.py             # CHAP-compatible training entry point
├── predict.py           # CHAP-compatible inference entry point
├── src/
│   ├── __init__.py
│   ├── features.py      # Weather lag and harmonic cyclical feature generators
│   └── model.py         # Log-transformed regularized epidemiological model
└── example_data/
    └── laos_subset.csv  # Official multi-region benchmark dataset
```

---

## 6. Citation

If you use or evaluate this model, please cite it using the metadata in [CITATION.cff](CITATION.cff):

```bibtex
@software{rahman2026chapdengue,
  author = {Rahman, Khalilur},
  title = {chap-dengue-forecaster: Climate-Informed Dengue Outbreak Forecasting Model for DHIS2 CHAP},
  year = {2026},
  url = {https://github.com/khalilurrrahmanridoykhan/chap-dengue-forecaster},
  license = {Apache-2.0}
}
```

---

## 7. License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
