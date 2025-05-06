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

# Load the dataset
data = pd.read_csv('water_potability.csv')
print("Original dataset shape:", data.shape)
print("Missing values per column:\n", data.isnull().sum())

# Define the Sine Cosine Algorithm (SCA) for imputation
class SCA_Imputer:
    def __init__(self, n_agents=7, max_iter=50, a=2, r_min=0, r_max=2*np.pi):
        self.n_agents = n_agents  # Number of search agents
        self.max_iter = max_iter  # Maximum number of iterations
        self.a = a  # Constant parameter to determine the balance between exploration and exploitation
        self.r_min = r_min  # Minimum random value for r1
        self.r_max = r_max  # Maximum random value for r1
        self.best_solution = None
        self.best_fitness = float('inf')
        self.solutions = []  # To store all good solutions

    def initialize_population(self, data):
        """Initialize population with different imputation strategies"""
        population = []

        # Create different imputation strategies
        strategies = [
            ('mean', SimpleImputer(strategy='mean')),
            ('median', SimpleImputer(strategy='median')),
            ('most_frequent', SimpleImputer(strategy='most_frequent')),
            ('constant_0', SimpleImputer(strategy='constant', fill_value=0)),
        ]

        # Add the strategies
        for name, imputer in strategies:
            imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            population.append(imputed_data)

        # Add some random imputations to complete the population
        while len(population) < self.n_agents:
            random_data = data.copy()
            for col in data.columns:
                if data[col].isnull().sum() > 0:
                    mask = data[col].isnull()
                    random_values = np.random.uniform(
                        data[col].min(),
                        data[col].max(),
                        size=mask.sum()
                    )
                    random_data.loc[mask, col] = random_values
            population.append(random_data)

        return population

    def fitness_function(self, imputed_data, original_data, X_cols, y_col):
        """Evaluate fitness using KNN classifier accuracy"""
        # Split data
        X = imputed_data[X_cols]
        y = imputed_data[y_col]

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_scaled, y_train)

            # Predict and calculate accuracy
            y_pred = knn.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)

            return 1 - acc  # We want to minimize, so return 1-acc
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return float('inf')  # Return worst fitness if error occurs

    def update_position(self, current_position, best_position, iteration):
        """Update position using Sine Cosine Algorithm"""
        r1 = self.a - iteration * (self.a / self.max_iter)  # Decreasing from a to 0
        r2 = np.random.uniform(self.r_min, self.r_max)  # Random number in [0, 2π]
        r3 = np.random.random()  # Random number in [0, 1]
        r4 = np.random.random()  # Random number in [0, 1]

        # For each feature that had missing values, update
        new_position = current_position.copy()

        for col in current_position.columns:
            if current_position[col].isnull().sum() > 0 or best_position[col].isnull().sum() > 0:
                # Get indices where original data had missing values
                missing_idx = current_position[col].isnull()

                if r4 < 0.5:  # Update using sine
                    new_values = current_position[col] + r1 * np.sin(r2) * np.abs(r3 * best_position[col] - current_position[col])
                else:  # Update using cosine
                    new_values = current_position[col] + r1 * np.cos(r2) * np.abs(r3 * best_position[col] - current_position[col])

                # Only update where there were missing values
                new_position[col] = new_values

        return new_position

    def optimize(self, data, X_cols, y_col):
        """Run the SCA optimization process"""
        # Initialize population
        population = self.initialize_population(data)

        # Evaluate initial population
        fitness_values = [self.fitness_function(agent, data, X_cols, y_col) for agent in population]

        # Find the best solution
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]

        # Main loop
        for iteration in range(self.max_iter):
            # Update each agent's position
            for i in range(self.n_agents):
                new_position = self.update_position(population[i], self.best_solution, iteration)

                # Evaluate new position
                new_fitness = self.fitness_function(new_position, data, X_cols, y_col)

                # Update if better
                if new_fitness < fitness_values[i]:
                    population[i] = new_position
                    fitness_values[i] = new_fitness

                    # Update global best if needed
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_position.copy()
                        self.best_fitness = new_fitness

            print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_fitness}")

        # Store the top 5 solutions
        indices = np.argsort(fitness_values)[:5]  # Get indices of top 5 solutions
        self.solutions = [population[idx].copy() for idx in indices]

        return self.solutions

    def save_solutions(self, solutions, base_filename="imputed_solution"):
        """Save the solutions to CSV files"""
        filenames = []
        for i, solution in enumerate(solutions):
            filename = f"{base_filename}_{i+1}.csv"
            solution.to_csv(filename, index=False)
            filenames.append(filename)
            print(f"Saved solution to {filename}")
        return filenames

# Grey Wolf Optimizer for Feature Selection
class GWO_FeatureSelection:
    def __init__(self, n_wolves=10, max_iter=30, dataset=None, classifier=None):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dataset = dataset
        self.classifier = classifier
        self.alpha_pos = None
        self.alpha_score = float('inf')
        self.beta_pos = None
        self.beta_score = float('inf')
        self.delta_pos = None
        self.delta_score = float('inf')
        self.best_features = None
        self.best_accuracy = 0
        self.best_solution = None
        self.convergence_curve = np.zeros(max_iter)

    def initialize_population(self, n_features):
        """Initialize binary wolves population"""
        # Each wolf is a binary vector where 1 means feature is selected, 0 means not selected
        return np.random.randint(0, 2, size=(self.n_wolves, n_features))

    def fitness_function(self, wolf_position, X, y):
        """Calculate fitness using MLP classifier and weighted accuracy/features"""
        # Get selected features based on wolf position (1 means selected, 0 means not selected)
        selected_features = np.where(wolf_position == 1)[0]

        # If no features selected, return worst fitness
        if len(selected_features) == 0:
            return float('inf')

        # Select features from data
        X_selected = X[:, selected_features]

        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

            # Train classifier
            self.classifier.fit(X_train, y_train)

            # Predict and calculate accuracy
            y_pred = self.classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Calculate fitness with the given formula
            alpha = 0.99  # Weight for accuracy
            beta = 0.01   # Weight for feature count
            n_selected = len(selected_features)
            n_total = X.shape[1]

            fitness = alpha * (1 - acc) + beta * (n_selected / n_total)

            return fitness
        except Exception as e:
            print(f"Error in GWO fitness calculation: {e}")
            return float('inf')

    def update_positions(self, positions, a, X, y):
        """Update positions of wolves based on alpha, beta, and delta positions"""
        n_wolves, n_features = positions.shape
        new_positions = np.zeros((n_wolves, n_features))

        for i in range(n_wolves):
            for j in range(n_features):
                # Calculate position updates based on alpha, beta, and delta
                r1 = np.random.random()
                r2 = np.random.random()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * self.alpha_pos[j] - positions[i, j])
                X1 = self.alpha_pos[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * self.beta_pos[j] - positions[i, j])
                X2 = self.beta_pos[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * self.delta_pos[j] - positions[i, j])
                X3 = self.delta_pos[j] - A3 * D_delta

                # Average the three positions
                X_new = (X1 + X2 + X3) / 3

                # Convert to binary using sigmoid function
                sigmoid_val = 1 / (1 + np.exp(-10 * (X_new - 0.5)))

                if np.random.random() < sigmoid_val:
                    new_positions[i, j] = 1
                else:
                    new_positions[i, j] = 0

        return new_positions

    def optimize(self, X, y):
        """Run the GWO optimization process"""
        n_features = X.shape[1]

        # Initialize wolf population
        positions = self.initialize_population(n_features)

        # Initialize alpha, beta, and delta positions
        self.alpha_pos = np.zeros(n_features)
        self.beta_pos = np.zeros(n_features)
        self.delta_pos = np.zeros(n_features)

        # Main loop
        for iteration in range(self.max_iter):
            # Calculate a (linearly decreased from 2 to 0)
            a = 2 - iteration * (2 / self.max_iter)

            # Evaluate each wolf
            for i in range(self.n_wolves):
                fitness = self.fitness_function(positions[i], X, y)

                # Update alpha, beta, and delta
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()

                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()

                    self.alpha_score = fitness
                    self.alpha_pos = positions[i].copy()

                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()

                    self.beta_score = fitness
                    self.beta_pos = positions[i].copy()

                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = positions[i].copy()

            # Update positions
            positions = self.update_positions(positions, a, X, y)

            # Store convergence info
            self.convergence_curve[iteration] = self.alpha_score

            print(f"GWO Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.alpha_score}")

            # Calculate accuracy and selected features for best solution
            selected_features = np.where(self.alpha_pos == 1)[0]
            X_selected = X[:, selected_features]
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Update best solution if better
            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_features = selected_features

                # Store full solution info
                self.best_solution = {
                    'accuracy': acc,
                    'selected_features': selected_features,
                    'num_features': len(selected_features),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

            print(f"Selected {len(selected_features)}/{n_features} features, Accuracy: {acc:.4f}")

        return self.best_solution, self.convergence_curve

def plot_confusion_matrix(cm, accuracy, n_features, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{title}\nAccuracy: {accuracy:.4f}, Features: {n_features}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

def plot_convergence(curve, title='Convergence Curve'):
    """Plot convergence curve"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(curve)+1), curve, marker='o')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.tight_layout()
    return plt

def main():
    # Load the dataset
    data = pd.read_csv('water_potability.csv')

    # Separate features and target
    X_cols = [col for col in data.columns if col != 'Potability']
    y_col = 'Potability'

    # Initialize SCA imputer
    sca_imputer = SCA_Imputer(n_agents=7, max_iter=20)

    # Run SCA optimization to get multiple imputed datasets
    print("Starting SCA optimization for imputation...")
    imputed_solutions = sca_imputer.optimize(data, X_cols, y_col)

    # Save imputed solutions to files
    solution_files = sca_imputer.save_solutions(imputed_solutions)

    # Initialize results dictionary to store GWO results for each imputed solution
    results = {}

    # For each imputed solution, run GWO feature selection
    for i, solution_file in enumerate(solution_files):
        print(f"\nProcessing imputed solution {i+1}: {solution_file}")

        # Load the imputed data
        imputed_data = pd.read_csv(solution_file)

        # Separate features and target
        X = imputed_data[X_cols].values
        y = imputed_data[y_col].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize MLP classifier for GWO
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=200,
            random_state=42
        )

        # Initialize GWO
        gwo = GWO_FeatureSelection(n_wolves=10, max_iter=20, classifier=mlp)

        # Run GWO optimization
        print(f"Starting GWO optimization for feature selection on solution {i+1}...")
        best_solution, convergence_curve = gwo.optimize(X_scaled, y)

        # Store results
        results[solution_file] = {
            'best_solution': best_solution,
            'convergence_curve': convergence_curve
        }

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        cm = best_solution['confusion_matrix']
        cm_plt = plot_confusion_matrix(
            cm,
            best_solution['accuracy'],
            best_solution['num_features'],
            title=f'Confusion Matrix - Solution {i+1}'
        )
        cm_plt.savefig(f'confusion_matrix_solution_{i+1}.png')

        # Plot and save convergence curve
        conv_plt = plot_convergence(
            convergence_curve,
            title=f'GWO Convergence Curve - Solution {i+1}'
        )
        conv_plt.savefig(f'convergence_curve_solution_{i+1}.png')

        # Print summary
        print(f"\nSolution {i+1} Summary:")
        print(f"Accuracy: {best_solution['accuracy']:.4f}")
        print(f"Selected {best_solution['num_features']} features out of {len(X_cols)}")
        print(f"Selected feature indices: {best_solution['selected_features']}")
        print(f"Selected feature names: {[X_cols[idx] for idx in best_solution['selected_features']]}")
        print(f"Confusion Matrix:\n{best_solution['confusion_matrix']}")

    # Find the best overall solution
    best_solution_file = max(results.keys(), key=lambda x: results[x]['best_solution']['accuracy'])
    best_accuracy = results[best_solution_file]['best_solution']['accuracy']
    best_num_features = results[best_solution_file]['best_solution']['num_features']
    best_feature_indices = results[best_solution_file]['best_solution']['selected_features']
    best_feature_names = [X_cols[idx] for idx in best_feature_indices]

    print("\n" + "="*50)
    print("BEST OVERALL SOLUTION")
    print("="*50)
    print(f"Best solution file: {best_solution_file}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Number of selected features: {best_num_features} out of {len(X_cols)}")
    print(f"Selected feature names: {best_feature_names}")
    print("="*50)

    # Plot combined convergence curves
    plt.figure(figsize=(12, 6))
    for i, (solution_file, result) in enumerate(results.items()):
        plt.plot(
            range(1, len(result['convergence_curve'])+1),
            result['convergence_curve'],
            marker='o',
            label=f'Solution {i+1}'
        )

    plt.grid(True)
    plt.title('GWO Convergence Curves for All Solutions')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('combined_convergence_curves.png')

    return results, best_solution_file

if __name__ == "__main__":
    start_time = time.time()
    results, best_solution = main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")