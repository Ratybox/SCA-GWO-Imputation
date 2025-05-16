import numpy as np
import random
import math
import matplotlib.pyplot as plt
from gwo import run_gwo_from_scratch
from data_handler import calculate_fitness, impute_data

def sca_objective_wrapper(imputation_solution, problem_info, gwo_pop_size, gwo_epochs):
    """
    Fonction objectif pour SCA qui utilise GWO pour la sélection de caractéristiques.
    SCA s'occupe de l'imputation, GWO de la sélection de caractéristiques.
    """
    # Exécuter GWO pour la sélection de caractéristiques
    n_features = problem_info["n_features"]
    
    # Initialiser la population GWO pour la sélection de caractéristiques (valeurs entre 0 et 1)
    initial_gwo_pop = np.random.rand(gwo_pop_size, n_features)
    
    # Définir une fonction wrapper pour GWO
    def gwo_fitness_wrapper(feature_selection):
        return -calculate_fitness(imputation_solution, feature_selection, problem_info)
    
    # Fonction pour exécuter GWO avec le wrapper ci-dessus
    def run_feature_selection_gwo():
        best_feature_selection, best_neg_fitness = run_gwo_from_scratch_feature_selection(
            initial_gwo_pop,
            gwo_fitness_wrapper,
            gwo_epochs,
            n_features
        )
        return best_feature_selection, -best_neg_fitness  # Convertir fitness négative en positive
    
    # Exécuter GWO pour la sélection de caractéristiques
    best_feature_selection, best_accuracy = run_feature_selection_gwo()
    
    return best_accuracy, best_feature_selection


def run_gwo_from_scratch_feature_selection(initial_population, fitness_function, epochs, dim):
    """Version modifiée de GWO pour la sélection de caractéristiques."""
    # Initialisation
    pop_size = len(initial_population)
    positions = np.array(initial_population)
    
    # Évaluation initiale
    fitness = np.array([fitness_function(pos) for pos in positions])
    
    # Trouver Alpha, Beta, Delta initiaux
    sorted_indices = np.argsort(fitness)
    alpha_idx, beta_idx, delta_idx = sorted_indices[0], sorted_indices[1], sorted_indices[2]
    alpha_pos, alpha_score = positions[alpha_idx].copy(), fitness[alpha_idx]
    beta_pos, beta_score = positions[beta_idx].copy(), fitness[beta_idx]
    delta_pos, delta_score = positions[delta_idx].copy(), fitness[delta_idx]
    
    # Boucle principale GWO
    for t in range(epochs):
        a_linear_component = 2 - 2 * (t / epochs)  # Paramètre a décroît linéairement
        
        # Mettre à jour les positions de tous les agents
        for i in range(pop_size):
            for j in range(dim):
                r1_alpha, r2_alpha = np.random.rand(), np.random.rand()
                a_alpha = 2 * a_linear_component * r1_alpha - a_linear_component
                c_alpha = 2 * r2_alpha
                distance_alpha = abs(c_alpha * alpha_pos[j] - positions[i, j])
                x1 = alpha_pos[j] - a_alpha * distance_alpha
                
                r1_beta, r2_beta = np.random.rand(), np.random.rand()
                a_beta = 2 * a_linear_component * r1_beta - a_linear_component
                c_beta = 2 * r2_beta
                distance_beta = abs(c_beta * beta_pos[j] - positions[i, j])
                x2 = beta_pos[j] - a_beta * distance_beta
                
                r1_delta, r2_delta = np.random.rand(), np.random.rand()
                a_delta = 2 * a_linear_component * r1_delta - a_linear_component
                c_delta = 2 * r2_delta
                distance_delta = abs(c_delta * delta_pos[j] - positions[i, j])
                x3 = delta_pos[j] - a_delta * distance_delta
                
                positions[i, j] = (x1 + x2 + x3) / 3
            
            # Appliquer les limites (entre 0 et 1 pour la sélection de caractéristiques)
            positions[i] = np.clip(positions[i], 0, 1)
            
            # Évaluer la nouvelle position
            new_fitness = fitness_function(positions[i])
            
            # Mettre à jour si meilleure
            if new_fitness < fitness[i]:
                fitness[i] = new_fitness
        
        # Mettre à jour Alpha, Beta, Delta
        sorted_indices = np.argsort(fitness)
        if fitness[sorted_indices[0]] < alpha_score:
            alpha_pos, alpha_score = positions[sorted_indices[0]].copy(), fitness[sorted_indices[0]]
        if fitness[sorted_indices[1]] < beta_score and sorted_indices[1] != sorted_indices[0]:
            beta_pos, beta_score = positions[sorted_indices[1]].copy(), fitness[sorted_indices[1]]
        if fitness[sorted_indices[2]] < delta_score and sorted_indices[2] != sorted_indices[0] and sorted_indices[2] != sorted_indices[1]:
            delta_pos, delta_score = positions[sorted_indices[2]].copy(), fitness[sorted_indices[2]]
    
    return alpha_pos, alpha_score


def run_hybrid_sca_gwo_from_scratch(problem_info, sca_pop_size, sca_epochs, gwo_pop_size, gwo_epochs):
    """
    Exécute l'hybride SCA-GWO depuis zéro.
    SCA s'occupe de l'imputation des valeurs manquantes.
    GWO s'occupe de la sélection des caractéristiques.
    """
    dim = problem_info["dim"]  # Dimension pour l'imputation
    lb = problem_info["lower_bound"]
    ub = problem_info["upper_bound"]
    
    # Initialisation SCA (pour l'imputation)
    positions = np.random.uniform(lb, ub, (sca_pop_size, dim))
    fitness = np.zeros(sca_pop_size)
    feature_selections = [None] * sca_pop_size
    
    # Évaluation initiale - Appel du wrapper GWO
    print("--- Évaluation initiale de la population SCA (avec GWO interne) ---")
    for i in range(sca_pop_size):
        fitness[i], feature_selections[i] = sca_objective_wrapper(positions[i], problem_info, gwo_pop_size, gwo_epochs)
        print(f"Agent {i+1}/{sca_pop_size} évalué. Accuracy: {fitness[i]:.6f}")
    
    # Trouver la meilleure solution initiale (plus grande accuracy)
    best_idx = np.argmax(fitness)
    best_pos = positions[best_idx].copy()
    best_fitness = fitness[best_idx]
    best_feature_selection = feature_selections[best_idx].copy()
    print(f"Meilleure Accuracy initiale: {best_fitness:.6f}")
    
    # Pour le graphique d'évolution
    history = [best_fitness]
    
    print(f"\n--- Démarrage de la boucle principale SCA (Époques: {sca_epochs}) ---")
    # Boucle principale SCA
    for t in range(sca_epochs):
        # Paramètre r1 - contrôle exploration/exploitation
        r1 = 2 - 2 * (t / sca_epochs)  # Décroît linéairement de 2 à 0
        
        # Pour chaque agent de recherche
        for i in range(sca_pop_size):
            # Pour chaque dimension
            for j in range(dim):
                # Générer r2, r3, r4
                r2 = (2 * math.pi) * np.random.rand()
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()
                
                # Mettre à jour la position basé sur sin ou cos
                if r4 < 0.5:
                    # Équation Sinus
                    positions[i, j] = positions[i, j] + (r1 * math.sin(r2) * abs(r3 * best_pos[j] - positions[i, j]))
                else:
                    # Équation Cosinus
                    positions[i, j] = positions[i, j] + (r1 * math.cos(r2) * abs(r3 * best_pos[j] - positions[i, j]))
            
            # Vérifier les limites pour l'agent entier
            positions[i] = np.clip(positions[i], lb, ub)
            
            # Évaluer la nouvelle position
            print(f"Époque {t+1}/{sca_epochs}, Évaluation de l'agent {i+1}/{sca_pop_size}...")
            new_fitness, new_feature_selection = sca_objective_wrapper(positions[i], problem_info, gwo_pop_size, gwo_epochs)
            
            # Mise à jour si meilleure (accuracy plus élevée)
            if new_fitness > fitness[i]:
                fitness[i] = new_fitness
                feature_selections[i] = new_feature_selection
        
        # Mettre à jour la meilleure solution globale
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_pos = positions[current_best_idx].copy()
            best_feature_selection = feature_selections[current_best_idx].copy()
        
        # Enregistrer pour le graphique
        history.append(best_fitness)
        
        print(f"--- Époque {t+1}/{sca_epochs} terminée. Meilleure Accuracy jusqu'à présent: {best_fitness:.6f} ---")
    
    # Créer et sauvegarder le graphique d'évolution
    plt.figure(figsize=(10, 6))
    plt.plot(range(sca_epochs + 1), history, 'b-', linewidth=2)
    plt.title('Évolution de la meilleure accuracy au fil des époques')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('evolution_accuracy.png')
    plt.show()
    
    # Préparation du résultat final
    # Caractéristiques sélectionnées (conversion en binaire)
    selected_features = np.round(best_feature_selection).astype(bool)
    feature_indices = np.where(selected_features)[0]
    
    result = {
        'best_imputation': best_pos,
        'best_feature_selection': best_feature_selection,
        'selected_feature_indices': feature_indices,
        'accuracy': best_fitness,
        'history': history
    }
    
    return result