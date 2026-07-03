import os
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Ensure the outputs directory exists
os.makedirs('../outputs', exist_ok=True)

print("--- Running Question 1 Simulations ---")

# (e) Simulate a single Brownian path and compute QV vs TV
N_q1, T_q1 = 10000, 1.0
dt_q1 = T_q1 / N_q1
dW_single = np.random.normal(0, np.sqrt(dt_q1), N_q1)
W_single = np.cumsum(np.insert(dW_single, 0, 0))

increments = np.diff(W_single)
qv_single = np.sum(increments**2)
tv_single = np.sum(np.abs(increments))

print(f"Q1(e) Empirical QV: {qv_single:.4f} (Theoretical T = {T_q1})")
print(f"Q1(e) Total Variation (TV): {tv_single:.4f}\n")

# (f) Plot empirical QV as a function of the number of partition steps N
N_steps = [10, 50, 100, 500, 1000, 5000, 10000]
qv_values = []

for n in N_steps:
    dt_n = T_q1 / n
    dW_n = np.random.normal(0, np.sqrt(dt_n), n)
    qv_values.append(np.sum(dW_n**2))

plt.figure(figsize=(10, 5))
plt.plot(N_steps, qv_values, marker='o', linestyle='-', color='b', label='Empirical QV')
plt.axhline(T_q1, color='r', linestyle='--', label='Theoretical T=1')
plt.xscale('log')
plt.xlabel('Number of Partition Steps (N)')
plt.ylabel('Quadratic Variation')
plt.title('Q1(f): Convergence of Quadratic Variation')
plt.legend()
plt.grid(True, which="both", ls="--")

# Save the plot to the outputs folder
plt.savefig('../outputs/q1_quadratic_variation_convergence.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/q1_quadratic_variation_convergence.png")
plt.show()

# (g) Estimate empirical mean and variance at t = 0.5 using 10,000 paths
paths_q1 = 10000
t_fixed = 0.5
dW_at_t = np.random.normal(0, np.sqrt(t_fixed), paths_q1)

print(f"Q1(g) at t = {t_fixed} over {paths_q1} paths:")
print(f"  Empirical Mean: {np.mean(dW_at_t):.4f} (Theoretical: 0.0)")
print(f"  Empirical Variance: {np.var(dW_at_t):.4f} (Theoretical: {t_fixed})\n")