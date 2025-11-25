import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def train_and_explain_model(X_features, y_labels, feature_names, n_estimators, random_state):
    """
    Train model with cross-validation, then train final model on all data.
    Returns feature importance and saves the final model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Cross-validation for evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    importances = []
    accuracies = []

    print("\n" + "="*50)
    print("Cross-Validation Results:")
    print("="*50)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_labels), 1):
        X_train = X_scaled[train_idx]
        y_train = y_labels[train_idx]
        X_test = X_scaled[test_idx]
        y_test = y_labels[test_idx]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)
        importances.append(model.feature_importances_)
        
        print(f"Fold {fold}: Accuracy = {accuracy:.4f}")

    print("="*50)
    print(f"Mean CV Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print("="*50)

    # Calculate mean feature importance
    mean_importance = np.mean(importances, axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_importance
    }).sort_values("Importance", ascending=False)

    # Train FINAL model on ALL data
    print("\nTraining final model on all data...")
    final_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced"
    )
    final_model.fit(X_scaled, y_labels)
    
    # Save model and scaler
    print("Saving model to 'trained_model.pkl'...")
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump({
            'model': final_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'classes': final_model.classes_
        }, f)
    
    print("✅ Model saved successfully!")
    print(f"Classes: {final_model.classes_}")
    
    return importance_df, np.mean(accuracies)