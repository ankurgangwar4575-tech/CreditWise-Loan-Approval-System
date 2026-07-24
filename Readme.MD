# 💼 CreditWise Loan Approval System

CreditWise is a Streamlit web app that predicts whether a loan application should be approved or rejected. It uses applicant, loan, and property details, then runs the input through pre-trained machine learning models to return a prediction and confidence score

Live app: https://loan-classifier.streamlit.app/

## 🔍 What This Project Does

The app helps simulate a fast loan pre-screening workflow for banks and NBFCs. A user enters financial and profile information such as income, credit score, debt-to-income ratio, employment status, loan purpose, and property area. The app then predicts the approval outcome with one of four trained models

## ✨ Main Features

- 📝 Interactive Streamlit form for loan-application inputs.
- 🧠 Model selector for Decision Tree, Logistic Regression, Gaussian Naive Bayes, and K-Nearest Neighbors
- ♻️ Reuses saved encoders, scaler, and feature order from the training pipeline
- 📊 Shows the predicted class and confidence score.
- 📈 Expands to display approval and rejection probabilities

## 📁 Project Structure

- `app.py` - Streamlit application entrypoint.
- `notebooks/credit_wise.ipynb` - data cleaning, EDA, feature engineering, training, and evaluation
- `data/loan_approval_data.csv` - training dataset
- `models/` - saved models and preprocessing artifacts
- `images/` - project images and supporting visuals

## 🗃️ Dataset Overview

The dataset contains applicant-level records with demographic, financial, employment, and loan-related attributes. The target variable is loan approval status, stored as `Yes` or `No`.

The training pipeline handled missing values with standard imputation strategies and encoded categorical variables using label encoding for binary or ordinal fields and one-hot encoding for nominal fields.

## 🛠️ Feature Engineering

The model pipeline uses engineered features to improve prediction quality:

- `Applicant_Income_log` using `np.log1p`.
- `Credit_Score_sq` to capture non-linear credit score effects.
- `DTI_Ratio_sq` to capture non-linear debt burden effects.

The final input vector is reordered to match the saved training feature list before prediction.

## 🤖 Models

The notebook trains and evaluates four classification models on the same feature set:

- Decision Tree
- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors

Decision Tree is the strongest performer in the current pipeline and is a good default choice for the app.

## ⚙️ Tech Stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, Joblib

## 🚀 How To Run Locally

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal.

## 🧩 How It Works

When you submit the form, the app:

1. Encodes categorical fields using the saved encoders.
2. Recomputes engineered numeric features from raw inputs.
3. Aligns the final dataframe to the training feature order.
4. Scales the input where required.
5. Sends the data to the selected model and displays the result.

## 📝 Notes

- The app uses saved artifacts from the `models/` directory instead of retraining on every run.
- If you update the training pipeline, regenerate the saved models and feature metadata so the app stays aligned.

## 👤 Author

Ankur Gangwar
- LinkedIn: https://www.linkedin.com/in/ankur-gangwar/
- GitHub: https://github.com/ankurgangwar4575-tech
- Project repo: https://github.com/ankurgangwar4575-tech/Mini-ML-Project
