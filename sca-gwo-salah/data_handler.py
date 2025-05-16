import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def load_and_prepare_data(csv_path, target_column_index=9):
    """Charge les données, identifie les NaN et prépare les indices."""
    data_full = pd.read_csv(csv_path).values
    features = data_full[:, :target_column_index].copy()
    target = data_full[:, target_column_index]

    nan_indices = np.array([
        (row, col)
        for col in range(features.shape[1])
        for row in np.argwhere(np.isnan(features[:, col])).flatten()
    ])
    nan_n = len(nan_indices)

    if nan_n == 0:
        raise ValueError("Aucune valeur NaN trouvée dans le dataset.")

    # Mise à l'échelle globale
    scaler = StandardScaler()
    # Fitter sur les lignes sans NaN pour plus de robustesse
    scaler.fit(features[~np.isnan(features).any(axis=1)])
    features_scaled = scaler.transform(features) # Les NaN restent NaN

    # Recalculer les indices NaN sur les données mises à l'échelle
    nan_indices_scaled = np.array([
        (row, col)
        for col in range(features_scaled.shape[1])
        for row in np.argwhere(np.isnan(features_scaled[:, col])).flatten()
    ])
    nan_n_scaled = len(nan_indices_scaled)

    # Masque et indices pour la séparation
    nan_mask_scaled = np.isnan(features_scaled).any(axis=1)
    original_indices_nan = np.where(nan_mask_scaled)[0]
    original_indices_complete = np.where(~nan_mask_scaled)[0]

    features_complete_scaled = features_scaled[~nan_mask_scaled]
    target_complete = target[~nan_mask_scaled]

    # Séparer les données complètes pour l'entraînement/test du KNN
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        features_complete_scaled, target_complete, test_size=0.2, random_state=42
    )

    # Entraîner KNN (sera utilisé dans la fonction objectif)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_knn, y_train_knn)

    # Définir les bornes pour l'optimisation
    lb = np.min(features_complete_scaled) # Borne inférieure scalaire pour imputation
    ub = np.max(features_complete_scaled) # Borne supérieure scalaire pour imputation
    
    # Nombre de caractéristiques pour la sélection
    n_features = features_scaled.shape[1]

    problem_info = {
        "features_scaled": features_scaled,
        "target": target,
        "nan_indices_scaled": nan_indices_scaled,
        "nan_n_scaled": nan_n_scaled,
        "original_indices_nan": original_indices_nan,
        "X_train_knn": X_train_knn,
        "X_test_knn": X_test_knn,
        "y_train_knn": y_train_knn,
        "y_test_knn": y_test_knn,
        "knn_model": knn,
        "lower_bound": lb,
        "upper_bound": ub,
        "dim": nan_n_scaled, # Dimension de la solution (nombre de NaN)
        "n_features": n_features, # Nombre de caractéristiques pour sélection
        "scaler": scaler # Garder le scaler pour une éventuelle transformation inverse
    }
    return problem_info

def impute_data(solution, problem_info):
    """Impute les valeurs manquantes avec la solution."""
    features_imputed = problem_info["features_scaled"].copy()
    nan_indices = problem_info["nan_indices_scaled"]
    
    # Imputer les valeurs
    for k, (row, col) in enumerate(nan_indices):
        features_imputed[row, col] = solution[k]
        
    return features_imputed

def calculate_fitness(imputation_solution, feature_selection, problem_info):
    """Calcule la fitness avec imputation et sélection de caractéristiques."""
    # Imputer les données
    features_imputed = impute_data(imputation_solution, problem_info)
    
    # Convertir la solution de sélection de caractéristiques binaire 
    # (arrondir les valeurs continues entre 0 et 1)
    feature_mask = np.round(feature_selection).astype(bool)
    
    # S'assurer qu'au moins une caractéristique est sélectionnée
    if not np.any(feature_mask):
        feature_mask[0] = True  # Sélectionner la première caractéristique par défaut
    
    # Préparer les données de test combinées avec sélection de caractéristiques
    X_test_imputed = features_imputed[problem_info["original_indices_nan"], :][:, feature_mask]
    X_test_knn = problem_info["X_test_knn"][:, feature_mask]
    X_test_combined = np.concatenate((X_test_imputed, X_test_knn), axis=0)
    
    y_test_combined = np.concatenate(
        (problem_info["target"][problem_info["original_indices_nan"]], 
         problem_info["y_test_knn"]), axis=0
    )
    
    # Réentraîner le modèle KNN sur les caractéristiques sélectionnées
    X_train_selected = problem_info["X_train_knn"][:, feature_mask]
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_selected, problem_info["y_train_knn"])
    
    # Prédire et évaluer
    y_pred = knn.predict(X_test_combined)
    acc = accuracy_score(y_test_combined, y_pred)
    
    # Retourner l'accuracy directement (nous maximisons maintenant)
    return acc