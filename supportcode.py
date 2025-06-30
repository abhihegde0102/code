import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, roc_curve, auc
from random import randint
import random
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

true_labels = np.random.randint(0, 5, size=50)  
predicted_labels = np.random.randint(0, 5, size=50)
class_names = ["Cellulitis Impetigo", "Eczema", "Psoriasis", "Rosacea", "Scabies Lyme"]
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plt.subplots(figsize=(8, 8))
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
cmd.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
plt.show()

np.random.seed(42) 
true_labels_binary = np.random.choice([0, 1], size=(50, 5), p=[0.7, 0.3])  
predicted_labels_binary = np.random.choice([0, 1], size=(50, 5), p=[0.7, 0.3])
y_true_binarized = true_labels_binary
y_pred_binarized = predicted_labels_binary
if not (1 in y_true_binarized[:, 0] and 0 in y_true_binarized[:, 0]):
    y_true_binarized[:, 0] = np.random.choice([0, 1], size=50, p=[0.7, 0.3])
precision, recall, _ = precision_recall_curve(y_true_binarized[:, 0], y_pred_binarized[:, 0])
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot(name="Precision-Recall")
plt.title("Precision-Recall Curve")
plt.show()

fpr, tpr, _ = roc_curve(y_true_binarized[:, 0], y_pred_binarized[:, 0])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
X, y = make_classification(n_samples=500, n_features=20, n_informative=15, n_classes=2, random_state=42)
model = LogisticRegression()

# Adjusted Learning Curve and Validation Curve to avoid high memory usage
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Training Accuracy", color="blue")
plt.plot(train_sizes, test_mean, label="Validation Accuracy", color="green")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.show()

# Validation Curve with limited parallel jobs
param_range = np.logspace(-3, 3, 7)
train_scores, test_scores = validation_curve(model, X, y, param_name="C", param_range=param_range, cv=5, scoring="accuracy", n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(param_range, train_mean, label="Training Accuracy", color="blue")
plt.plot(param_range, test_mean, label="Validation Accuracy", color="green")
plt.xscale("log")
plt.xlabel("Parameter C")
plt.ylabel("Accuracy")
plt.title("Validation Curve")
plt.legend(loc="best")
plt.show()

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
plt.figure()
plt.plot(range(1, 6), cv_scores, marker='o', label="CV Accuracy", color="purple")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Scores")
plt.legend(loc="best")
plt.grid()
plt.show()
