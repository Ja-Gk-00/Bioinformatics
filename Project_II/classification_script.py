import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Directory must exist
os.makedirs("results", exist_ok=True)

k_values = [3, 4, 5, 6]
test_size = 0.2
random_state = 42
n_splits_cv = 10

classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    "SVM": SVC(probability=True, random_state=random_state),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state)
}

def run_classification(k, neg_type):
    Data_path = 'counted_split_data/'

    positive_file = Data_path + f"positive_{k}_mer.csv"
    if neg_type == "vista":
        negative_file = Data_path + f"negative_vista{k}_mer.csv"
        neg_label = "vista"
    else:
        negative_file = Data_path + f"negative_random{k}_mer.csv"
        neg_label = "random"

    pos_df = pd.read_csv(positive_file)
    neg_df = pd.read_csv(negative_file)

    df = pd.concat([pos_df, neg_df], ignore_index=True)
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)

    print(f"\n=== Processing k={k}, Negative={neg_label} ===")
    print("Cross-validation AUC on training set:")
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"{name}: Mean AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    print("---------------------------------------------")

    results = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n{name} (k={k}, Neg={neg_label}) Test Set Performance:")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")

        results.append({
            "Classifier": name,
            "k": k,
            "Negative_Set": neg_label,
            "AUC": auc,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg','Pos'])
        plt.title(f"{name} (k={k}, Neg={neg_label}) - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_filename = f"results/cm_{name}_k{k}_{neg_label}.png"
        plt.savefig(cm_filename)
        plt.close()

        # ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.figure(figsize=(4,3))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
        plt.plot([0,1], [0,1], 'k--')
        plt.title(f"{name} (k={k}, Neg={neg_label}) - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc='lower right')
        plt.tight_layout()
        roc_filename = f"results/roc_{name}_k{k}_{neg_label}.png"
        plt.savefig(roc_filename)
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/metrics_k{k}_{neg_label}.csv", index=False)
    print(f"Saved metrics and plots for k={k} with {neg_label} negatives to 'results/' directory.")

# Run classification
for k in k_values:
    run_classification(k, "vista")
    run_classification(k, "random")
