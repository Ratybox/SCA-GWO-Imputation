#### gwo.py
import numpy as np
import random
import os # Ajout basé sur la référence
from data_handler import calculate_fitness # Garder notre fonction fitness

# --- Fonctions adaptées de gwoCodeFromScratch.py ---

# Note: Pas besoin d'initialiser alpha/beta/delta séparément au début,
# car on les détermine en triant la population initiale.

def update_pack_adapted(positions, fitness):
    """Trouve les indices et scores des meilleurs (alpha, beta, delta)."""
    sorted_indices = np.argsort(fitness)
    alpha_idx = sorted_indices[0]
    beta_idx = sorted_indices[1]
    delta_idx = sorted_indices[2]
    # Retourne les positions et scores pour clarté
    return (positions[alpha_idx].copy(), fitness[alpha_idx],
            positions[beta_idx].copy(), fitness[beta_idx],
            positions[delta_idx].copy(), fitness[delta_idx])

def update_position_adapted(positions, alpha_pos, beta_pos, delta_pos, a_linear_component, lb, ub, dim):
    """Met à jour les positions des loups basé sur alpha, beta, delta."""
    new_positions = np.copy(positions)
    pop_size = positions.shape[0]

    for i in range(pop_size):
        for j in range(dim):
            # Utiliser np.random.rand() pour la simplicité et consistance avec SCA
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

            new_positions[i, j] = (x1 + x2 + x3) / 3

        # Appliquer les limites après avoir mis à jour toutes les dimensions
        new_positions[i] = np.clip(new_positions[i], lb, ub)

    return new_positions

# --- Fonction principale GWO utilisant la logique adaptée ---

def run_gwo_from_scratch(problem_info, pop_size, epochs, initial_population=None):
    """Exécute GWO depuis zéro en utilisant la logique adaptée de la référence."""
    dim = problem_info["dim"]
    lb = problem_info["lower_bound"]
    ub = problem_info["upper_bound"]

    # Initialisation
    if initial_population is not None and len(initial_population) == pop_size:
        positions = np.array(initial_population)
        positions = np.clip(positions, lb, ub) # Assurer le respect des bornes initiales
    else:
        positions = np.random.uniform(lb, ub, (pop_size, dim))

    # Évaluation initiale
    fitness = np.array([calculate_fitness(pos, problem_info) for pos in positions])

    # Trouver Alpha, Beta, Delta initiaux
    alpha_pos, alpha_score, beta_pos, beta_score, delta_pos, delta_score = update_pack_adapted(positions, fitness)

    # Boucle principale GWO
    for t in range(epochs):
        a_linear_component = 2 - 2 * (t / epochs) # Paramètre a décroît linéairement

        # Mettre à jour les positions de tous les agents
        new_positions = update_position_adapted(positions, alpha_pos, beta_pos, delta_pos, a_linear_component, lb, ub, dim)

        # Évaluer les nouvelles positions
        new_fitness = np.array([calculate_fitness(pos, problem_info) for pos in new_positions])

        # Mise à jour de la population (remplacer si la nouvelle position est meilleure)
        for i in range(pop_size):
            if new_fitness[i] < fitness[i]:
                fitness[i] = new_fitness[i]
                positions[i] = new_positions[i]

        # Mettre à jour Alpha, Beta, Delta basé sur la population mise à jour
        current_alpha_pos, current_alpha_score, \
        current_beta_pos, current_beta_score, \
        current_delta_pos, current_delta_score = update_pack_adapted(positions, fitness)

        # Mettre à jour les leaders globaux si améliorés
        # (La logique de mise à jour en cascade de la référence est implicitement gérée
        # par la re-sélection à chaque itération)
        alpha_pos, alpha_score = current_alpha_pos, current_alpha_score
        beta_pos, beta_score = current_beta_pos, current_beta_score
        delta_pos, delta_score = current_delta_pos, current_delta_score

        # Optionnel: Affichage
        # if (t + 1) % 10 == 0 or t == epochs - 1:
        #     print(f"GWO Inner Epoch {t+1}/{epochs}, Best Fitness: {alpha_score:.6f}")

    # Retourner la meilleure solution trouvée
    return alpha_pos, alpha_score
