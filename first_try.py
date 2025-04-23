import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import random
import math

# Load and prepare the dataset
def load_data(file_path):
    df = pd.read_csv('water_potability.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    return df

# Sine Cosine Algorithm (SCA) for missing value imputation
class SCAImputation:
    def __init__(self, n_agents=30, max_iter=50, a=2, r_min=0, r_max=2):
        self.n_agents = n_agents  # Number of search agents
        self.max_iter = max_iter  # Maximum number of iterations
        self.a = a  # Parameter to control exploration vs exploitation
        self.r_min = r_min  # Min random value
        self.r_max = r_max  # Max random value
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.convergence_curve = []
    
    def initialize_population(self, X, missing_indices):
        """
        Initialize population with random values for missing data
        X: Dataset with missing values
        missing_indices: Indices of missing values per column
        """
        population = []
        
        # Get the range of each feature to initialize values within bounds
        min_vals = X.min()
        max_vals = X.max()
        
        # Create n_agents copies of the dataset
        for _ in range(self.n_agents):
            X_copy = X.copy()
            
            # Fill missing values with random values within feature range
            for col in missing_indices:
                if len(missing_indices[col]) > 0:  # If column has missing values
                    random_values = np.random.uniform(
                        low=min_vals[col],
                        high=max_vals[col],
                        size=len(missing_indices[col])
                    )
                    X_copy.iloc[missing_indices[col], X_copy.columns.get_loc(col)] = random_values
            
            population.append(X_copy)
        
        return population
    
    def evaluate_fitness(self, population, X, y, missing_indices, test_size=0.2):
        """
        Evaluate fitness using KNN classifier accuracy
        """
        fitness_values = []
        
        for agent in population:
            try:
                # Split the dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    agent, y, test_size=test_size, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Train KNN classifier
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, y_train)
                
                # Evaluate
                y_pred = knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                fitness_values.append(accuracy)
                
                # Update best solution if current is better
                if accuracy > self.best_fitness:
                    self.best_fitness = accuracy
                    self.best_solution = agent.copy()
            except Exception as e:
                # In case of errors, assign low fitness
                print(f"Error in fitness evaluation: {e}")
                fitness_values.append(0.0)
        
        return fitness_values
    
    def update_position(self, population, fitness_values, current_iter):
        """
        Update positions using SCA equations
        """
        best_idx = np.argmax(fitness_values)
        best_agent = population[best_idx]
        
        r1_coef = self.a - current_iter * (self.a / self.max_iter)  # Decreasing gradually
        
        new_population = []
        
        for agent_idx, agent in enumerate(population):
            if agent_idx == best_idx:
                new_population.append(agent)
                continue
                
            new_agent = agent.copy()
            
            # Update each missing value
            for col in agent.columns:
                if col in self.missing_indices and len(self.missing_indices[col]) > 0:
                    for idx in self.missing_indices[col]:
                        # Calculate SCA parameters
                        r1 = r1_coef  # Defines balance between exploration and exploitation
                        r2 = np.random.uniform(self.r_min, self.r_max)  # Random number for destination weight
                        r3 = np.random.uniform(0, 2*np.pi)  # Random angle
                        r4 = np.random.random()  # Random number to decide sin or cos
                        
                        # Destination position (from best agent)
                        best_val = best_agent.iloc[idx, agent.columns.get_loc(col)]
                        current_val = agent.iloc[idx, agent.columns.get_loc(col)]
                        
                        # Update using SCA equation
                        if r4 < 0.5:  # Use sine
                            new_val = current_val + r1 * np.sin(r3) * np.abs(r2 * best_val - current_val)
                        else:  # Use cosine
                            new_val = current_val + r1 * np.cos(r3) * np.abs(r2 * best_val - current_val)
                        
                        # Ensure value is within feature bounds
                        min_val = agent[col].min()
                        max_val = agent[col].max()
                        new_val = max(min_val, min(new_val, max_val))
                        
                        new_agent.iloc[idx, new_agent.columns.get_loc(col)] = new_val
            
            new_population.append(new_agent)
        
        return new_population
    
    def fit(self, X, y):
        """
        Run SCA algorithm to impute missing values
        """
        # Find missing value indices for each column
        self.missing_indices = {}
        for col in X.columns:
            self.missing_indices[col] = X[X[col].isnull()].index.tolist()
        
        # Initialize population
        population = self.initialize_population(X, self.missing_indices)
        
        # Optimization loop
        for iteration in range(self.max_iter):
            # Evaluate fitness
            fitness_values = self.evaluate_fitness(population, X, y, self.missing_indices)
            
            # Store best fitness for convergence curve
            self.convergence_curve.append(np.max(fitness_values))
            
            # Update positions
            population = self.update_position(population, fitness_values, iteration)
            
            print(f"SCA Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")
        
        return self.best_solution
    
    def plot_convergence(self):
        """
        Plot the convergence curve
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.convergence_curve) + 1), self.convergence_curve)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (KNN Accuracy)')
        plt.title('SCA Convergence Curve for Imputation')
        plt.grid(True)
        plt.show()

# Grey Wolf Optimizer (GWO) for feature selection
class GWOFeatureSelection:
    def __init__(self, n_wolves=13, max_iter=15, a_init=2, a_final=0):
        self.n_wolves = n_wolves  # Number of wolves (search agents)
        self.max_iter = max_iter  # Maximum number of iterations
        self.a_init = a_init  # Initial value of parameter a
        self.a_final = a_final  # Final value of parameter a
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.convergence_curve = []
        
    def initialize_population(self, n_features):
        """
        Initialize binary wolf population for feature selection
        1 means feature is selected, 0 means feature is not selected
        """
        # Initialize population with random binary values
        population = np.random.randint(0, 2, size=(self.n_wolves, n_features))
        
        # Ensure at least one feature is selected for each wolf
        for i in range(self.n_wolves):
            if np.sum(population[i]) == 0:
                # If no feature is selected, randomly select one
                random_feature = np.random.randint(0, n_features)
                population[i, random_feature] = 1
                
        return population
    
    def evaluate_fitness(self, population, X, y, test_size=0.2):
        """
        Evaluate fitness using MLP classifier accuracy
        """
        fitness_values = []
        
        for wolf in population:
            # If no feature is selected, assign zero fitness
            if np.sum(wolf) == 0:
                fitness_values.append(0)
                continue
            
            # Select features based on wolf's position
            selected_features = np.where(wolf == 1)[0]
            X_selected = X.iloc[:, selected_features]
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y, test_size=test_size, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                # Feature count penalty to encourage fewer features
                n_features = len(selected_features)
                alpha = 0.01  # Weight for feature reduction
                
                # Train MLP classifier
                # Adjust hidden layer size based on feature count
                hidden_size = min(25, max(5, n_features ))
                
                mlp = MLPClassifier(
                    hidden_layer_sizes=(hidden_size,),
                    max_iter=30,
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=42
                )
                
                mlp.fit(X_train, y_train)
                
                # Evaluate
                y_pred = mlp.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Apply feature penalty
                fitness = accuracy - alpha * (n_features / X.shape[1])
                
                fitness_values.append(fitness)
                
                # Update best solution if current is better
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = wolf.copy()
                    self.best_accuracy = accuracy
                    self.selected_feature_count = n_features
                    
            except Exception as e:
                print(f"Error in GWO fitness evaluation: {e}")
                fitness_values.append(0)
        
        return np.array(fitness_values)
    
    def update_positions(self, population, fitness_values, iteration):
        """
        Update positions using GWO algorithm
        """
        # Sort the wolves based on fitness
        sorted_indices = np.argsort(fitness_values)[::-1]
        
        # Get alpha, beta, delta wolves (the best three solutions)
        alpha_idx = sorted_indices[0]
        beta_idx = sorted_indices[1] if len(sorted_indices) > 1 else alpha_idx
        delta_idx = sorted_indices[2] if len(sorted_indices) > 2 else beta_idx
        
        alpha_pos = population[alpha_idx].copy()
        beta_pos = population[beta_idx].copy()
        delta_pos = population[delta_idx].copy()
        
        # Calculate a decreasing from a_init to a_final
        a = self.a_init - iteration * ((self.a_init - self.a_final) / self.max_iter)
        
        # Update each wolf's position
        new_population = []
        
        for i in range(len(population)):
            new_pos = np.zeros_like(population[i], dtype=float)
            
            for j in range(len(population[i])):
                # Parameters for alpha
                r1_alpha = np.random.random()
                r2_alpha = np.random.random()
                A1_alpha = 2 * a * r1_alpha - a
                C1_alpha = 2 * r2_alpha
                
                D_alpha = np.abs(C1_alpha * alpha_pos[j] - population[i][j])
                X1 = alpha_pos[j] - A1_alpha * D_alpha
                
                # Parameters for beta
                r1_beta = np.random.random()
                r2_beta = np.random.random()
                A1_beta = 2 * a * r1_beta - a
                C1_beta = 2 * r2_beta
                
                D_beta = np.abs(C1_beta * beta_pos[j] - population[i][j])
                X2 = beta_pos[j] - A1_beta * D_beta
                
                # Parameters for delta
                r1_delta = np.random.random()
                r2_delta = np.random.random()
                A1_delta = 2 * a * r1_delta - a
                C1_delta = 2 * r2_delta
                
                D_delta = np.abs(C1_delta * delta_pos[j] - population[i][j])
                X3 = delta_pos[j] - A1_delta * D_delta
                
                # Average the positions (continuous values)
                new_pos[j] = (X1 + X2 + X3) / 3.0
            
            # Convert to binary using sigmoid
            sigmoid_values = 1 / (1 + np.exp(-10 * (new_pos - 0.5)))
            binary_pos = np.zeros_like(new_pos, dtype=int)
            
            for j in range(len(sigmoid_values)):
                if np.random.random() < sigmoid_values[j]:
                    binary_pos[j] = 1
                else:
                    binary_pos[j] = 0
            
            # Ensure at least one feature is selected
            if np.sum(binary_pos) == 0:
                random_feature = np.random.randint(0, len(binary_pos))
                binary_pos[random_feature] = 1
            
            new_population.append(binary_pos)
        
        return np.array(new_population)
    
    def fit(self, X, y):
        """
        Run GWO to find optimal feature subset
        """
        n_features = X.shape[1]
        
        # Initialize wolf population
        population = self.initialize_population(n_features)
        
        # For storing alpha wolf (best solution) in each iteration
        self.best_accuracy = 0
        self.selected_feature_count = 0
        
        # Optimization loop
        for iteration in range(self.max_iter):
            # Evaluate fitness
            fitness_values = self.evaluate_fitness(population, X, y)
            
            # Store best fitness for convergence curve
            self.convergence_curve.append(np.max(fitness_values))
            
            # Update positions
            population = self.update_positions(population, fitness_values, iteration)
            
            print(f"GWO Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}, "
                  f"Accuracy: {self.best_accuracy:.4f}, Selected Features: {self.selected_feature_count}")
        
        # Return the indices of selected features
        return np.where(self.best_solution == 1)[0]
    
    def plot_convergence(self):
        """
        Plot the convergence curve
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.convergence_curve) + 1), self.convergence_curve)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (with feature penalty)')
        plt.title('GWO Convergence Curve for Feature Selection')
        plt.grid(True)
        plt.show()

# Main function to execute the hybrid SCA-GWO approach
def main():
    # Load dataset
    df = load_data('Water_potability.csv')
    
    # Separate features and target
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # The last column is the target
    
    # Step 1: Initialize GWO for feature selection
    print("Starting Grey Wolf Optimization for Feature Selection...")
    gwo = GWOFeatureSelection(n_wolves=14, max_iter=10)
    
    # Check if there are missing values in the dataset
    missing_values = X.isnull().sum().sum()
    
    if missing_values > 0:
        print(f"Found {missing_values} missing values. Imputing using SCA first...")
        
        # Create a temporary imputation with mean for initial feature selection
        imputer = SimpleImputer(strategy='mean')
        X_temp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Do initial feature selection
        selected_features = gwo.fit(X_temp, y)
        
        print("Selected features:", X.columns[selected_features].tolist())
        
        # Get selected features dataset
        X_selected = X.iloc[:, selected_features]
        
        # Now check for missing values in selected features
        missing_in_selected = X_selected.isnull().sum().sum()
        
        if missing_in_selected > 0:
            print(f"Found {missing_in_selected} missing values in selected features. Imputing using SCA...")
            
            # Use SCA to impute missing values in selected features
            sca = SCAImputation(n_agents=50, max_iter=30)
            X_imputed = sca.fit(X_selected, y)
            
            print("Missing values imputed with SCA")
            print("SCA Best Fitness (KNN Accuracy):", sca.best_fitness)
            
            # Final evaluation with imputed values and selected features
            print("Final Evaluation with SCA-imputed values and GWO-selected features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Evaluate with MLP
            mlp = MLPClassifier(hidden_layer_sizes=(13,), max_iter=23, random_state=42)
            mlp.fit(X_train, y_train)
            
            y_pred = mlp.predict(X_test)
            final_accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Final MLP Accuracy with SCA-imputed & GWO-selected features: {final_accuracy:.4f}")
            
            # Plot convergence curves
            sca.plot_convergence()
            gwo.plot_convergence()
            
        else:
            # No missing values in selected features
            print("No missing values in selected features. Using original values.")
            
            # Final evaluation with selected features
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Evaluate with MLP
            mlp = MLPClassifier(hidden_layer_sizes=(15,), max_iter=23, random_state=42)
            mlp.fit(X_train, y_train)
            
            y_pred = mlp.predict(X_test)
            final_accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Final MLP Accuracy with GWO-selected features: {final_accuracy:.4f}")
            
            # Plot convergence curve
            gwo.plot_convergence()
    
    else:
        # No missing values, just do feature selection
        print("No missing values found. Proceeding with feature selection only.")
        
        selected_features = gwo.fit(X, y)
        
        print("Selected features:", X.columns[selected_features].tolist())
        
        # Final evaluation with selected features
        X_selected = X.iloc[:, selected_features]
        
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Evaluate with MLP
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        mlp.fit(X_train, y_train)
        
        y_pred = mlp.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Final MLP Accuracy with GWO-selected features: {final_accuracy:.4f}")
        
        # Plot convergence curve
        gwo.plot_convergence()

if __name__ == "__main__":
    main()