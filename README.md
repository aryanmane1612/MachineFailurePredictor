# Machine Failure Predictor using Machine Learning

An AI-based predictive maintenance project that uses machine sensor data to classify machine conditions as **Healthy** or **Damaged**.

## Project Overview

Unexpected machine failures can cause downtime, maintenance costs, and reduced productivity. This project explores how machine-learning techniques can help identify potential bearing damage using vibration and phase-current measurements.

A Support Vector Machine (SVM) classifier is trained on features extracted from the Paderborn bearing dataset.

## Objectives

- Analyze sensor data associated with machine-bearing conditions.
- Extract statistical and signal-processing features from sensor measurements.
- Train a machine-learning model to classify bearing conditions.
- Evaluate the model using grouped cross-validation.
- Prepare a prediction module for future integration with ESP32-based hardware.

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Joblib
- Jupyter Notebook
- ESP32 (planned hardware integration)

## Machine Learning Model

The project uses a Support Vector Machine (SVM) with an RBF kernel.

The trained model uses **48 numerical features** extracted from three signals:

- Vibration
- Phase current 1
- Phase current 2

The model predicts one of two classes:

| Label | Predicted condition |
|---|---|
| 0 | Healthy |
| 1 | Damaged |

## Dataset

The project uses the Paderborn bearing dataset.

The processed dataset contains 796 rows after deduplication. Raw recordings are excluded from this repository to keep it lightweight.

Dataset source: [Paderborn University Bearing DataCenter](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter)

Please review the dataset's terms of use before redistributing its data.

## Project Structure

```text
MachineFailurePredictor/
├── data/
│   └── processed/
├── models/
│   ├── feature_columns.json
│   ├── machine_failure_svm.joblib
│   └── model_metadata.json
├── notebooks/
├── reports/
│   ├── evaluation_metrics.json
│   └── nested_cv_predictions.csv
├── src/
│   ├── feature_extraction.py
│   ├── predictor.py
│   ├── feature_extraction.ipynb
│   └── inspect_data.ipynb
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aryanmane1612/MachineFailurePredictor.git
   cd MachineFailurePredictor
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```powershell
   .venv\Scripts\activate
   ```

3. Install the required libraries:

   ```bash
   pip install numpy pandas scipy scikit-learn joblib
   ```

## Model Files

The `models/` directory contains:

- `machine_failure_svm.joblib` — trained model pipeline.
- `feature_columns.json` — expected feature names and order.
- `model_metadata.json` — model-related metadata.

The prediction module is located at `src/predictor.py`.

**Note:** The prediction function currently expects a dictionary containing all 48 features. The raw-signal-to-feature pipeline must be completed and verified before making predictions directly from new sensor recordings.

## Evaluation

The grouped nested cross-validation experiment achieved approximately **99.75% accuracy** on the evaluated splits.

This result is specific to the evaluation performed on this dataset. It does not guarantee equivalent performance on unseen machines or real-world hardware.

## Current Status

- [x] Dataset preprocessing and deduplication
- [x] Feature engineering
- [x] SVM model training
- [x] Model evaluation
- [x] Prediction module
- [ ] Complete raw-signal prediction pipeline
- [ ] ESP32 integration and hardware testing

## Future Scope

- Integrate the model with an ESP32 development board.
- Collect sensor measurements from physical machinery.
- Develop real-time machine-condition monitoring.
- Evaluate performance on independent machines and real operating conditions.

## Author

**Aryan Rahul Mane**

GitHub: [@aryanmane1612](https://github.com/aryanmane1612)

LinkedIn: [Aryan Mane](https://linkedin.com/in/aryan-mane-4414b7397)

---

*This project is an educational predictive-maintenance prototype. It is not a certified industrial safety system.*
