import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load Data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# 2. Train Model (We start with 10 estimators)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluate
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.4f}")

# 4. Save Metrics to a Markdown report
with open("report.md", "w", encoding="utf-8") as f:
    f.write("# Model Evaluation Report 📊\n\n")
    f.write(f"- **Accuracy:** {acc:.4f}\n\n")
    f.write("## Confusion Matrix\n")
    f.write("![Confusion Matrix](./plot.png)\n")

# 5. Save Plot
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, display_labels=data.target_names, cmap=plt.cm.Blues
)
plt.savefig("plot.png")
