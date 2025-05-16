import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Set random seed for reproducibility
np.random.seed(42)

# Define the Sine Cosine Algorithm (SCA) for imputation
class SCA_Imputer:
    def __init__(self, n_agents=7, max_iter=50, a=2, r_min=0, r_max=2*np.pi):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.a = a
        self.r_min = r_min
        self.r_max = r_max
        self.best_solution = None
        self.best_fitness = float('inf')
        self.solutions = []

    def initialize_population(self, data):
        population = []
        strategies = [
            ('mean', SimpleImputer(strategy='mean')),
            ('median', SimpleImputer(strategy='median')),
            ('most_frequent', SimpleImputer(strategy='most_frequent')),
            ('constant_0', SimpleImputer(strategy='constant', fill_value=0)),
        ]
        for name, imputer in strategies:
            imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            population.append(imputed_data)
        while len(population) < self.n_agents:
            random_data = data.copy()
            for col in data.columns:
                if data[col].isnull().sum() > 0:
                    mask = data[col].isnull()
                    random_data.loc[mask, col] = np.random.uniform(data[col].min(), data[col].max(), size=mask.sum())
            population.append(random_data)
        return population

    def fitness_function(self, imputed_data, original_data, X_cols, y_col):
        X = imputed_data[X_cols]
        y = imputed_data[y_col]
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            return 1 - acc
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return float('inf')

    def update_position(self, current_position, best_position, iteration):
        r1 = self.a - iteration * (self.a / self.max_iter)
        r2 = np.random.uniform(self.r_min, self.r_max)
        r3 = np.random.random()
        r4 = np.random.random()
        new_position = current_position.copy()
        for col in current_position.columns:
            if current_position[col].isnull().sum() > 0:
                missing_idx = current_position[col].isnull()
                if r4 < 0.5:
                    new_values = current_position[col] + r1 * np.sin(r2) * np.abs(r3 * best_position[col] - current_position[col])
                else:
                    new_values = current_position[col] + r1 * np.cos(r2) * np.abs(r3 * best_position[col] - current_position[col])
                new_position[col] = new_values
        return new_position

    def optimize(self, data, X_cols, y_col):
        population = self.initialize_population(data)
        fitness_values = [self.fitness_function(agent, data, X_cols, y_col) for agent in population]
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        for iteration in range(self.max_iter):
            for i in range(self.n_agents):
                new_position = self.update_position(population[i], self.best_solution, iteration)
                new_fitness = self.fitness_function(new_position, data, X_cols, y_col)
                if new_fitness < fitness_values[i]:
                    population[i] = new_position
                    fitness_values[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_position.copy()
                        self.best_fitness = new_fitness
            print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_fitness}")
        indices = np.argsort(fitness_values)[:5]
        self.solutions = [population[idx].copy() for idx in indices]
        return self.solutions

    def save_solutions(self, solutions, base_filename="imputed_solution"):
        filenames = []
        for i, solution in enumerate(solutions):
            filename = f"{base_filename}_{i+1}.csv"
            solution.to_csv(filename, index=False)
            filenames.append(filename)
            print(f"Saved solution to {filename}")
        return filenames

# Grey Wolf Optimizer for Feature Selection
class GWO_FeatureSelection:
    def __init__(self, n_wolves=10, max_iter=30, classifier=None):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.classifier = classifier
        self.alpha_pos = None
        self.alpha_score = float('inf')
        self.beta_pos = None
        self.beta_score = float('inf')
        self.delta_pos = None
        self.delta_score = float('inf')
        self.best_solution = None
        self.convergence_curve = np.zeros(max_iter)
        self.accuracy_curve = np.zeros(max_iter)  # Track accuracy per iteration

    def initialize_population(self, n_features):
        return np.random.randint(0, 2, size=(self.n_wolves, n_features))

    def fitness_function(self, wolf_position, X, y):
        selected = np.where(wolf_position == 1)[0]
        if len(selected) == 0:
            return float('inf')
        X_sel = X[:, selected]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        alpha = 0.99
        beta = 0.01
        fitness = alpha * (1 - acc) + beta * (len(selected) / X.shape[1])
        return fitness, acc, confusion_matrix(y_test, y_pred), selected

    def update_positions(self, positions, a):
        n_wolves, n_features = positions.shape
        new_positions = np.zeros((n_wolves, n_features))
        for i in range(n_wolves):
            for j in range(n_features):
                r1, r2 = np.random.random(), np.random.random()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * self.alpha_pos[j] - positions[i, j])
                X1 = self.alpha_pos[j] - A1 * D_alpha
                r1, r2 = np.random.random(), np.random.random()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * self.beta_pos[j] - positions[i, j])
                X2 = self.beta_pos[j] - A2 * D_beta
                r1, r2 = np.random.random(), np.random.random()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * self.delta_pos[j] - positions[i, j])
                X3 = self.delta_pos[j] - A3 * D_delta
                X_new = (X1 + X2 + X3) / 3
                sig = 1 / (1 + np.exp(-10 * (X_new - 0.5)))
                new_positions[i, j] = 1 if np.random.random() < sig else 0
        return new_positions

    def optimize(self, X, y):
        n_features = X.shape[1]
        positions = self.initialize_population(n_features)
        self.alpha_pos = np.zeros(n_features)
        self.beta_pos = np.zeros(n_features)
        self.delta_pos = np.zeros(n_features)

        for it in range(self.max_iter):
            a = 2 - it * (2 / self.max_iter)
            best_acc = 0
            for i in range(self.n_wolves):
                fitness, acc, cm, sel = self.fitness_function(positions[i], X, y)
                # Update alpha, beta, delta
                if fitness < self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                    self.alpha_score, self.alpha_pos = fitness, positions[i].copy()
                elif fitness < self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = fitness, positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score, self.delta_pos = fitness, positions[i].copy()
            positions = self.update_positions(positions, a)
            self.convergence_curve[it] = self.alpha_score
            # Evaluate accuracy on alpha_pos
            fit, acc, cm, sel = self.fitness_function(self.alpha_pos, X, y)
            self.accuracy_curve[it] = acc
            # Store best solution once
            if it == self.max_iter - 1 or acc > (self.best_solution['accuracy'] if self.best_solution else 0):
                self.best_solution = {
                    'accuracy': acc,
                    'selected_features': sel,
                    'num_features': len(sel),
                    'confusion_matrix': cm
                }
            print(f"GWO Iter {it+1}/{self.max_iter}, Fitness: {self.alpha_score:.4f}, Acc: {acc:.4f}, Features: {len(sel)}")
        return self.best_solution, self.convergence_curve, self.accuracy_curve

# Plotting functions
def plot_confusion_matrix(cm, accuracy, n_features, title='Confusion Matrix'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{title}\nAccuracy: {accuracy:.4f}, Features: {n_features}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

def plot_convergence(curve, title='Convergence Curve'):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(curve)+1), curve, marker='o')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.tight_layout()
    return plt

# New function: plot accuracy curve
```def plot_accuracy_curve(curve, title='Accuracy per Iteration'):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(curve)+1), curve, marker='o')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    return plt
```

def main():
    data = pd.read_csv('water_potability.csv')
    X_cols = [c for c in data.columns if c != 'Potability']
    y_col = 'Potability'
    sca = SCA_Imputer(n_agents=7, max_iter=20)
    print("Starting SCA optimization for imputation...")
    imputed_solutions = sca.optimize(data, X_cols, y_col)
    sol_files = sca.save_solutions(imputed_solutions)
    results = {}

    for i, f in enumerate(sol_files):
        print(f"\nProcessing imputed solution {i+1}: {f}")
        imputed = pd.read_csv(f)
        X = StandardScaler().fit_transform(imputed[X_cols].values)
        y = imputed[y_col].values
        mlp = MLPClassifier(hidden_layer_sizes=(10,7), activation='relu', solver='adam', max_iter=200, random_state=42)
        gwo = GWO_FeatureSelection(n_wolves=10, max_iter=20, classifier=mlp)
        print(f"Starting GWO optimization for feature selection on solution {i+1}...")
        best_sol, conv_curve, acc_curve = gwo.optimize(X, y)
        results[f] = {'best_solution': best_sol, 'convergence_curve': conv_curve, 'accuracy_curve': acc_curve}
        # Save plots
        plot_confusion_matrix(best_sol['confusion_matrix'], best_sol['accuracy'], best_sol['num_features'], title=f'Confusion Matrix - Solution {i+1}').savefig(f'confusion_matrix_solution_{i+1}.png')
        plot_convergence(conv_curve, title=f'GWO Convergence Curve - Solution {i+1}').savefig(f'convergence_curve_solution_{i+1}.png')
        plot_accuracy_curve(acc_curve, title=f'Accuracy Curve - Solution {i+1}').savefig(f'accuracy_curve_solution_{i+1}.png')
        print(f"Saved accuracy curve for solution {i+1} as accuracy_curve_solution_{i+1}.png")

    # Combined plots
    plt.figure(figsize=(12,6))
    for i, (f, res) in enumerate(results.items()):
        plt.plot(range(1, len(res['accuracy_curve'])+1), res['accuracy_curve'], marker='o', label=f'Sol {i+1}')
    plt.grid(True)
    plt.title('Combined Accuracy Curves')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('combined_accuracy_curves.png')
    print('Saved combined accuracy curves as combined_accuracy_curves.png')

    return results

if __name__ == "__main__":
    start = time.time()
    results = main()
    end = time.time()
    print(f"\nTotal execution time: {end - start:.2f} seconds")
