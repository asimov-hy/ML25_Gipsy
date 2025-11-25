import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def train_and_explain_model(X_features, y_labels, feature_names, n_estimators, random_state):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    importances = []

    for train_idx, _ in skf.split(X_scaled, y_labels):
        X_train = X_scaled[train_idx]
        y_train = y_labels[train_idx]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        importances.append(model.feature_importances_)

    mean_importance = np.mean(importances, axis=0)

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_importance
    }).sort_values("Importance", ascending=False)

    return df
