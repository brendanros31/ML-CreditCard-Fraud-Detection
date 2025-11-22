import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics


# Prediction Stats
def evaluate(model, y_test, X_test):
    print(f'\n{model.__class__.__name__}')

    y_prob = model.predict_proba(X_test)[:,1]
    print(f'AUC: {metrics.roc_auc_score(y_test, y_prob)}')

    y_pred = model.predict(X_test)   # Predictions
    print(f'F1 Score: {metrics.f1_score(y_test, y_pred)}')
    print(f'Precision: {metrics.precision_score(y_test, y_pred)}')
    print(f'Recall: {np.sqrt(metrics.recall_score(y_test, y_pred))}')
    print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
        # Accuracy is not very important for this dataset.


# Plot Actual, Predicted values
class plots:
    # Confusion Matrix
    def ConfusionMatrix(model, X_test, y_test):
        y_pred = model.predict(X_test)   # Predictions

        metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

        model_name = model.__class__.__name__
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()

    # Area Under Curve
    def auc(model, X_test, y_test):
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
        
        # Compute AUC
        auc_score = metrics.roc_auc_score(y_test, y_prob)
        
        # Plot ROC curve
        plt.figure(figsize=(8,6))

        model_name = model.__class__.__name__
        plt.plot(fpr, tpr, label=f'{model_name or model.__class__.__name__} (AUC = {auc_score:.4f})')
        plt.plot([0,1], [0,1], 'k--', label='Random Guess')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title(f'ROC Curve - {model_name or model.__class__.__name__}')

        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()