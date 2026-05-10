import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# -------------------------------
# Create required folders
# -------------------------------
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/processed_dataset.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Models
# -------------------------------
models = {
    "Random_Forest": RandomForestClassifier(random_state=42),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(random_state=42)
}

results = {}

# Track best model
best_model = None
best_score = 0

# -------------------------------
# Confusion Matrix Function
# -------------------------------
def plot_cm(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{name} Confusion Matrix")
    
    plt.savefig(f"outputs/plots/{name}_cm.png")
    plt.close()

# -------------------------------
# Train all models
# -------------------------------
for name, model in models.items():
    print(f"\n🔹 Training {name}...\n")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(classification_report(y_test, y_pred))
    
    plot_cm(y_test, y_pred, name)

    # Track best model
    if acc > best_score:
        best_score = acc
        best_model = model

# -------------------------------
# Save best model
# -------------------------------
joblib.dump(best_model, "models/model.pkl")
print("\n✅ Best model saved as models/model.pkl")

# -------------------------------
# Model Comparison
# -------------------------------
print("\n📊 Model Comparison:\n")
for model_name, acc in results.items():
    print(f"{model_name}: {acc*100:.2f}%")

# Convert to table
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])

print("\n📋 Comparison Table:\n")
print(results_df)

# Save comparison
results_df.to_csv("outputs/model_comparison.csv", index=False)

# -------------------------------
# Plot Model Comparison
# -------------------------------
plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)

plt.savefig("outputs/plots/model_comparison.png")
plt.show()

# -------------------------------
# Feature Importance (Random Forest)
# -------------------------------
rf_model = models["Random_Forest"]

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure()
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Top 10 Important Features")

plt.savefig("outputs/plots/feature_importance.png")
plt.show()

print("\n✅ Feature importance plot saved")