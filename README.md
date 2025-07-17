# codealpha_-Disease-Prediction-from-Medical-Data-
 Disease Prediction from Medical Data 
# 🩺 Disease Prediction from Medical Data

This Machine Learning project predicts the likelihood of certain diseases based on structured medical data such as age, symptoms, test results, and health indicators. This project was developed as part of my internship at [CodeAlpha](https://www.linkedin.com/company/codealpha/).

---

## 🎯 Objective

To build an accurate classification model that can predict the presence or absence of diseases like **Heart Disease**, **Diabetes**, and **Breast Cancer** using patient data.

---

## 📊 Datasets Used

Publicly available datasets from the **UCI Machine Learning Repository**:

- Heart Disease Dataset
- Diabetes Dataset (Pima Indians)
- Breast Cancer Wisconsin Dataset

Each dataset contains labeled patient records including features such as:
- Age, Gender
- Blood Pressure, Glucose Levels
- BMI, Insulin, Skin Thickness
- Cholesterol, Chest Pain Type
- Tumor size, Texture, etc.

---

## 🧠 Machine Learning Algorithms

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier (optional)

---

## 📦 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn (for visualization)
- Jupyter Notebook

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling

2. **Exploratory Data Analysis (EDA)**
   - Visualizing distributions
   - Correlation heatmaps
   - Feature importance

3. **Model Building**
   - Training and testing split
   - Hyperparameter tuning (GridSearchCV)
   - Cross-validation

4. **Model Evaluation**
   - Accuracy
   - Precision, Recall
   - F1-Score
   - Confusion Matrix
   - ROC-AUC Curve

5. **Model Comparison and Conclusion**

---

## 🧪 Results

| Dataset       | Best Model         | Accuracy | F1-Score | ROC-AUC |
|---------------|--------------------|----------|----------|---------|
| Heart Disease | Random Forest      | XX%      | XX%      | XX      |
| Diabetes      | SVM                | XX%      | XX%      | XX      |
| Breast Cancer | Logistic Regression| XX%      | XX%      | XX      |

> Replace `XX` with actual values after model training.

---

## 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/disease-prediction-ml.git
cd disease-prediction-ml
