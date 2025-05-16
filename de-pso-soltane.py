import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mealpy.math_based import SCA
from mealpy.swarm_based import GWO, PSO
from mealpy.evolutionary_based import DE
from mealpy.utils.space import FloatVar
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
import time
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(path):
    """Load data and prepare with minimal initial analysis"""
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    
    # Extract features and target
    features = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    
    feature_names = df.columns[:-1].tolist()
    
    return features, target, feature_names


def get_nan_info(features):
    """Get information about missing values"""
    nan_indices = [(r, c) for c in range(features.shape[1])
                   for r in np.argwhere(np.isnan(features[:, c])).flatten()]
    return nan_indices, len(nan_indices)


def calculate_statistical_params(features, feature_names):
    """Calculate statistical parameters for each feature to guide optimization bounds"""
    feature_stats = []
    
    for col in range(features.shape[1]):
        col_data = features[:, col]
        non_nan_data = col_data[~np.isnan(col_data)]
        
        # Calculate statistics for this column
        stats = {
            'name': feature_names[col],
            'min': np.min(non_nan_data),
            'max': np.max(non_nan_data),
            'mean': np.mean(non_nan_data),
            'median': np.median(non_nan_data),
            'std': np.std(non_nan_data),
            'q1': np.percentile(non_nan_data, 25),
            'q3': np.percentile(non_nan_data, 75)
        }
        feature_stats.append(stats)
    
    return feature_stats


def get_column_specific_bounds(feature_stats, nan_indices):
    """Define column-specific bounds for optimization based on statistics"""
    lb = []
    ub = []
    
    for row, col in nan_indices:
        stats = feature_stats[col]
        iqr = stats['q3'] - stats['q1']
        
        # Define bounds based on statistical properties with some margin
        lower_bound = max(stats['min'], stats['q1'] - 1.0 * iqr)
        upper_bound = min(stats['max'], stats['q3'] + 1.0 * iqr)
        
        lb.append(lower_bound)
        ub.append(upper_bound)
    
    return lb, ub


def objective_factory(features, target, nan_indices, nan_n, feature_names):
    """Create simplified objective function for faster execution"""
    # Create a clean dataset mask (rows without missing values)
    nan_mask = ~np.isnan(features).any(axis=1)
    features_clean = features[nan_mask]
    target_clean = target[nan_mask]
    
    # Use a single simple classifier for faster evaluation
    classifier = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_split=5, 
                                       random_state=42, n_jobs=-1)
    
    # Using fewer cross-validation folds
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def fitness(solution):
    # نسخة مؤقتة من البيانات الأصلية
        temp_features = features.copy()
        
        # عوّض القيم المفقودة بالحل الحالي
        for k, (row, col) in enumerate(nan_indices):
            temp_features[row, col] = solution[k]
        
        # تحديث النسخة النظيفة بناءً على النسخة المعوضة
        nan_mask_updated = ~np.isnan(temp_features).any(axis=1)
        features_clean_updated = temp_features[nan_mask_updated]
        target_clean_updated = target[nan_mask_updated]
        
        # تأكد من وجود بيانات قابلة للتدريب
        if len(features_clean_updated) < 10:
            return 1.0  # لا يمكن التدريب، نرجع أقل دقة ممكنة

        # استخدم أول تقسيم من cross-validation
        train_idx, test_idx = next(skf.split(features_clean_updated, target_clean_updated))
        X_train, y_train = features_clean_updated[train_idx], target_clean_updated[train_idx]
        X_test, y_test = features_clean_updated[test_idx], target_clean_updated[test_idx]
        
        # تدريب النموذج
        classifier.fit(X_train, y_train)
        
        # التنبؤ
        y_pred = classifier.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        return 1 - acc
    
    return fitness


def run_optimizers(features, target, nan_indices, nan_n, feature_names, lb, ub, epochs=20, pop_size=50):
    """Run fewer optimization algorithms with reduced parameters for faster execution"""
    # Create objective function
    fitness_func = objective_factory(features, target, nan_indices, nan_n, feature_names)
    
    # Define optimization problem
    problem = {
        "obj_func": fitness_func,
        "bounds": FloatVar(lb=lb, ub=ub, name="impute"),
        "minmax": "min"
    }
    
    # Reduced set of algorithms to run
    optimizers = [
        {"name": "PSO", "algorithm": PSO.OriginalPSO(epoch=epochs, pop_size=pop_size)},
        {"name": "DE", "algorithm": DE.OriginalDE(epoch=epochs, pop_size=pop_size)}
    ]
    
    results = []
    
    # Run each algorithm
    for opt in optimizers:
        print(f"\nRunning {opt['name']} algorithm...")
        start_time = time.time()
        
        # Define termination condition
        termination = {
            "max_epoch": epochs
        }
        
        # Run optimization
        algorithm = opt["algorithm"]
        result = algorithm.solve(problem, termination=termination)
        
        accuracy = 1 - result.target.fitness
        execution_time = time.time() - start_time
        
        print(f"{opt['name']} algorithm - Accuracy: {accuracy:.4f}, Execution time: {execution_time:.2f} seconds")
        
        # ✅ استخراج تطور الدقة من history في كائن الـ algorithm
        fitness_history = algorithm.history.list_global_best_fit
        accuracy_history = [1 - f for f in fitness_history]

        # رسم تطور الدقة
        plt.figure(figsize=(8, 4))
        plt.plot(accuracy_history, marker='o', label=opt['name'])
        plt.title(f"Accuracy evolution over epochs - {opt['name']}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        results.append({
            "name": opt["name"],
            "result": result,
            "accuracy": accuracy,
            "execution_time": execution_time
        })
    
    best_result = max(results, key=lambda x: x["accuracy"])
    
    print(f"\nBest algorithm: {best_result['name']}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")
    
    return best_result



def evaluate_solution(features, target, solution, nan_indices, feature_names):
    """Evaluate performance of proposed solution using a single model"""
    # Impute missing values
    imputed_features = features.copy()
    for k, (row, col) in enumerate(nan_indices):
        imputed_features[row, col] = solution[k]
        print(f"Imputing value at row {row}, column {feature_names[col]} with value {solution[k]:.4f}")
    
    # Use a single model for evaluation
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    # Simplified cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, imputed_features, target, cv=skf, scoring='accuracy')
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    
    print(f"RandomForest - Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Train model on full dataset
    model.fit(imputed_features, target)
    
    # Detailed evaluation 
    predictions = model.predict(imputed_features)
    
    print(f"\nFinal accuracy: {mean_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(target, predictions))
    
    return model, mean_acc, imputed_features


def feature_importance(features, target, feature_names):
    """Analyze feature importance to help improve the model"""
    print("\nFeature importance analysis:")
    
    # Using RandomForest with reduced parameters
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    # Remove rows with missing values for training
    mask = ~np.isnan(features).any(axis=1)
    X_train = features[mask]
    y_train = target[mask]
    
    rf.fit(X_train, y_train)
    
    # Extract and display feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature ranking by importance:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return importances, indices


def optimize_and_evaluate(path, epochs=20, pop_size=50):
    """Main function to run optimization and evaluate results"""
    print(f"Starting with reduced parameters: epochs={epochs}, population size={pop_size}")
    
    # Load and prepare data
    features, target, feature_names = load_and_prepare_data(path)
    
    # Get information about missing values
    nan_indices, nan_n = get_nan_info(features)
    if nan_n == 0:
        print("No missing values to impute!")
        return None, None, None
    
    print(f"\nFound {nan_n} missing values in the dataset")
    
    # Calculate statistical parameters for features
    feature_stats = calculate_statistical_params(features, feature_names)
    
    # Get column-specific bounds for optimization
    lb, ub = get_column_specific_bounds(feature_stats, nan_indices)
    
    # Normalize data using RobustScaler (more resistant to outliers)
    scaler = RobustScaler()
    features_scaled = features.copy()
    
    # Scale columns that don't contain missing values
    for col in range(features.shape[1]):
        col_data = features[:, col]
        non_nan_indices = ~np.isnan(col_data)
        if np.any(non_nan_indices):
            col_data_non_nan = col_data[non_nan_indices].reshape(-1, 1)
            col_data_scaled = scaler.fit_transform(col_data_non_nan).flatten()
            features_scaled[non_nan_indices, col] = col_data_scaled
    
    # Analyze feature importance
    importances, indices = feature_importance(features, target, feature_names)
    
    # Run optimization algorithms
    print("\nStarting optimization process...")
    best_result = run_optimizers(features_scaled, target, nan_indices, nan_n, feature_names, lb, ub, epochs, pop_size)
    
    # Evaluate solution
    print("\nEvaluating proposed solution...")
    best_model, best_acc, imputed_features = evaluate_solution(
        features, target, best_result["result"].solution, nan_indices, feature_names
    )
    
    # Save imputed data
    df_imputed = pd.DataFrame(imputed_features, columns=feature_names)
    df_imputed['target'] = target
    output_path = f"{path.split('.')[0]}_imputed.csv"
    df_imputed.to_csv(output_path, index=False)
    
    print(f"\nFinal results:")
    print(f"Best optimization algorithm: {best_result['name']}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Imputed dataset saved as: {output_path}")
    
    return best_acc, best_model, imputed_features


if __name__ == "__main__":
    # Configuration parameters - significantly reduced
    data_path = "water_potability.csv"
    epochs = 20  # Reduced from 50
    pop_size = 200  # Reduced from 50
    
    # Run optimization and evaluation
    best_acc, best_model, imputed_features = optimize_and_evaluate(data_path, epochs, pop_size)