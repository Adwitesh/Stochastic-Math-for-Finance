import numpy as np

np.random.seed(42)

def check_walk_return(dimension, path_length):
    """Simulates a simple symmetric random walk; returns True if origin is revisited."""
    # Start at the d-dimensional origin
    coordinates = np.zeros(dimension)
    
    for step in range(path_length):
        # Pick one random index/axis to move along
        chosen_axis = np.random.randint(0, dimension)
        # Choose direction: +1 or -1
        step_direction = np.random.choice([-1, 1])
        
        coordinates[chosen_axis] += step_direction
        
        # Check if we are precisely back at the origin vector
        is_at_origin = True
        for d_idx in range(dimension):
            if coordinates[d_idx] != 0:
                is_at_origin = False
                break
                
        if is_at_origin:
            return True
            
    return False

# Parameters for multi-dimensional testing
dimensions_to_test = [1, 2, 3]
horizon_lengths = [100, 1000, 10000]
simulated_trajectories = 1000

print("   EMPIRICAL RETURN FRACTIONS BY DIMENSION/STEPS ")

for dim in dimensions_to_test:
    for length in horizon_lengths:
        successful_returns = 0
        
        for iteration in range(simulated_trajectories):
            if check_walk_return(dimension=dim, path_length=length):
                successful_returns += 1
                
        empirical_return_rate = successful_returns / simulated_trajectories
        print(f"Dim d={dim} | Path Length n={length:5d} | Return Proportion: {empirical_return_rate:.4f}")

Observations = """
Observations on Return Behavior by Dimension and Path Length:

1. Spatial Effect (Dimension):
   - For d=1 and d=2, the walk is recurrent. As path length (n) grows, the 
     proportion of paths returning to the origin approaches 1.0, because the 
     lower-dimensional grid forces eventual intersections.
   - For d=3, the walk is transient. The return proportion plateaus early 
     (around 0.34) and freezes, because a 3D grid offers infinite paths to 
     escape the origin completely.

2. Temporal Effect (Path Length):
   - In recurrent dimensions (d=1, d=2), increasing n from 100 to 10,000 
     significantly raises the return rate, giving distant paths more time to return.
   - In the transient dimension (d=3), increasing n yields diminishing returns. 
     If a walk escapes the origin early on, it enters an expanding space where 
     the chance of wandering back drops asymptotically to zero.
"""

print(Observations)
        