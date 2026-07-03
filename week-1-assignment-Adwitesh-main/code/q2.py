import numpy as np

# Random seed setting
np.random.seed(42)

# Rating Transition Matrix (Ordered 0, 1, 2, 3)
# State 2 row modified to follow the explicit assignment rule [0, 0.5, 0.5, 0]
transition_matrix = np.array([
    [0.60, 0.25, 0.05, 0.10],
    [0.00, 0.50, 0.50, 0.00],
    [0.00, 0.50, 0.50, 0.00],
    [0.00, 0.00, 0.00, 1.00]
])

states = [0, 1, 2, 3]

def run_migration_path(start_state, max_steps):
    """Simulates a single random trajectory across rating states."""
    current_state = start_state
    trajectory = [current_state]
    
    for step in range(max_steps):
        row_probabilities = transition_matrix[current_state]
        current_state = np.random.choice(states, p=row_probabilities)
        trajectory.append(current_state)
        
    return trajectory

# Part (g): Single path extraction over 100 steps
sample_path = run_migration_path(start_state=0, max_steps=100)
print(f"Part (g): \nSingle Path Terminal State (t=100): State {sample_path[-1]}")

# Part (h): Broad Multi-path Ensemble Simulation
num_paths = 1000
steps_per_path = 100

state_3_terminal_count = 0
class_12_terminal_count = 0

for trajectory_idx in range(num_paths):
    simulated_path = run_migration_path(start_state=0, max_steps=steps_per_path)
    final_rating = simulated_path[-1]
    
    if final_rating == 3:
        state_3_terminal_count += 1
    elif final_rating == 1 or final_rating == 2:
        class_12_terminal_count += 1

# Proportion Calculations
prop_default = state_3_terminal_count / num_paths
prop_spec_distressed_trap = class_12_terminal_count / num_paths

print("Part (h):")
print(f"Proportion in State 3 (Default)   : {prop_default:.4f}")
print(f"Proportion in Class {{1, 2}} (Trap) : {prop_spec_distressed_trap:.4f}")