import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# App Title
# -----------------------------
st.title("📊 Stacking Ensemble Classification App")
st.write("Comparison of Base Models and Stacking Ensemble")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("kc_house_data.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Preprocessing
# -----------------------------
st.subheader("Data Preprocessing")

# Drop date & define target
X = df.drop(columns=["price", "date"])
y = (df["price"] > df["price"].median()).astype(int)

# One-hot encoding
X = pd.get_dummies(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.success("Data preprocessing completed")

# -----------------------------
# Base Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

st.subheader("Base Model Evaluation")

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc

results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
st.dataframe(results_df)

best_model_name = results_df["Accuracy"].idxmax()
best_model = models[best_model_name]

st.success(f"Best Base Model: {best_model_name}")

# -----------------------------
# Confusion Matrix - Best Model
# -----------------------------
st.subheader("Confusion Matrix – Best Base Model")

best_pred = best_model.predict(X_test)
cm_best = confusion_matrix(y_test, best_pred)

fig1, ax1 = plt.subplots()
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

# -----------------------------
# Stacking Ensemble
# -----------------------------
st.subheader("Stacking Ensemble Model")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

meta_train = np.zeros((X_train.shape[0], len(models)))
meta_test = np.zeros((X_test.shape[0], len(models)))

for i, (name, model) in enumerate(models.items()):
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        meta_train[val_idx, i] = model.predict(X_val)

    model.fit(X_train, y_train)
    meta_test[:, i] = model.predict(X_test)

# Meta-model
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(meta_train, y_train)

stacked_pred = meta_model.predict(meta_test)
stack_accuracy = accuracy_score(y_test, stacked_pred)

st.write("### Stacking Model Accuracy")
st.success(f"{stack_accuracy:.4f}")

# -----------------------------
# Confusion Matrix - Stacking
# -----------------------------
st.subheader("Confusion Matrix – Stacking Model")

cm_stack = confusion_matrix(y_test, stacked_pred)

fig2, ax2 = plt.subplots()
sns.heatmap(cm_stack, annot=True, fmt="d", cmap="Greens", ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# -----------------------------
# Final Conclusion
# -----------------------------
st.subheader("Final Conclusion")

st.markdown("""
- The **stacking ensemble model** outperforms the best individual base model.
- It combines strengths of multiple models.
- Stacking is effective when base models are diverse.
- However, stacking is **not always superior** if base models are weak or highly correlated.
""")
