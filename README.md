# Indian Consumer Credit Behavior Classification Model

## 📊 Project Overview
This data science project constructs a binary logistic regression classifier using `scikit-learn` to predict consumer demographic profiles based on credit card transaction traits. After proving via exploratory data diagnostics that raw transaction amounts possessed zero linear correlation with consumer features ($R^2$: -0.0001), the pipeline was refactored from a regression baseline into a classification framework to isolate subtle behavioral spending signals across 26,052 real-world retail transactions.

---

## 🛠️ Key Technical Implementations

*   **Pipeline Refactoring**: Successfully pivoted a dead-end regression framework into a binary logistic classification system, shifting the target variable from continuous pricing rows to demographic indicators.
*   **Chronological Feature Engineering**: Parsed raw calendar text strings to extract day-of-the-week indexes, engineering a custom behavioral metric (`is_weekend`) to isolate shifting consumer weekend expenditure patterns.
*   **Categorical Matrix Encoding**: Implemented multi-column One-Hot Encoding via Pandas across dimensional transaction factors (`Card Type` and `Expense Categories`), applying explicit integer type casting (`.astype(int)`) to eliminate boolean noise and guarantee matrix shape alignment.
*   **Data Leakage Prevention**: Explicitly dropped unique system constants, text strings, tracking identifiers, and regional city inputs from the final feature grid ($X$), maintaining strict predictive validation integrity.

---

## 📈 Model Performance & Statistical Evaluation

The logistic classifier was fitted to an 80% training data partition and validated against an unseen 20% testing matrix containing 5,211 independent transaction records.

*   **Overall Classification Accuracy**: `53.60%` — The model successfully beats a random 50/50 blind coin-flip threshold, proving the presence of a mild behavioral spending signal within category selection.

### 📊 Confusion Matrix Diagnostics
Using automated `ConfusionMatrixDisplay` pipelines, predictions were bucketed into a strict 2x2 grid to diagnose underlying model behavior and check classification errors:
*   **True Negatives (Correctly Identified Males)**: `733`
*   **True Positives (Correctly Identified Females)**: `2,060`
*   **False Negatives (Missed Females)**: `694`
*   **False Positives (Missed Males)**: `1,724`

**Analytical Takeaway**: The heavy concentration of True Positives and False Positives highlights a distinct classification bias toward the majority class in the training dataset. This visual diagnosis provides a direct avenue for production optimization via synthetic oversampling (SMOTE) or the integration of continuous regional parameters.

---

## 🧰 Tech Stack Used
*   **Data Preprocessing**: Python, Pandas, NumPy
*   **Machine Learning Math**: Scikit-Learn (`LogisticRegression`, `train_test_split`)
*   **Visual Validation**: Matplotlib, `ConfusionMatrixDisplay`
*   **Execution Environment**: Google Colab Notebook
