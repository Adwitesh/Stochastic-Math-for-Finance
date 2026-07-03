import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

# Ensure the outputs directory exists
os.makedirs('../outputs', exist_ok=True)

print("--- Running Question 2 Simulations ---")

# (d) & (e) Simulate 50,000 paths via Euler-Maruyama
n_paths_q2, N_q2, T_q2 = 50000, 500, 1.0
dt_q2 = T_q2 / N_q2
mu_q2, sigma_q2, S0 = 0.07, 0.2, 100.0

S = np.full((n_paths_q2, N_q2 + 1), float(S0))
for k in range(N_q2):
    dW_q2 = np.random.normal(0, np.sqrt(dt_q2), n_paths_q2)
    S[:, k+1] = S[:, k] + mu_q2 * S[:, k] * dt_q2 + sigma_q2 * S[:, k] * dW_q2

S_terminal = S[:, -1]

# Theoretical metrics
th_mean = S0 * np.exp(mu_q2 * T_q2)
th_std = S0 * np.exp(mu_q2 * T_q2) * np.sqrt(np.exp(sigma_q2**2 * T_q2) - 1)

print(f"Q2(e) Terminal Stock Price S_1 metrics:")
print(f"  Empirical Mean: {np.mean(S_terminal):.2f} (Theoretical: {th_mean:.2f})")
print(f"  Empirical Std: {np.std(S_terminal):.2f} (Theoretical: {th_std:.2f})\n")

# Plot histogram overlayed with theoretical log-normal density
plt.figure(figsize=(10, 5))
plt.hist(S_terminal, bins=100, density=True, alpha=0.6, color='skyblue', label='Empirical $S_1$ Distribution')
x_axis = np.linspace(min(S_terminal), max(S_terminal), 500)
shape_param = sigma_q2 * np.sqrt(T_q2)
scale_param = S0 * np.exp((mu_q2 - 0.5 * sigma_q2**2) * T_q2)
plt.plot(x_axis, stats.lognorm.pdf(x_axis, s=shape_param, scale=scale_param), 'r-', lw=2, label='Theoretical Log-Normal Density')
plt.xlabel('Terminal Stock Price ($S_1$)')
plt.ylabel('Density')
plt.title('Q2(d): Terminal Stock Price Distribution vs Theory')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot to the outputs folder
plt.savefig('../outputs/q2_gbm_terminal_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/q2_gbm_terminal_distribution.png")
plt.show()

# (f) Repeat for different volatilities
volatilities = [0.1, 0.3, 0.5]
K = 110
print("Q2(f) Volatility Sensitivity Analysis:")

for sig in volatilities:
    S_v = np.full((n_paths_q2, N_q2 + 1), float(S0))
    for k in range(N_q2):
        dW_v = np.random.normal(0, np.sqrt(dt_q2), n_paths_q2)
        S_v[:, k+1] = S_v[:, k] + mu_q2 * S_v[:, k] * dt_q2 + sig * S_v[:, k] * dW_v
    
    S_v_term = S_v[:, -1]
    emp_payoff = np.mean(np.maximum(S_v_term - K, 0))
    print(f"  Sigma = {sig}: Empirical Mean = {np.mean(S_v_term):.2f}, Std = {np.std(S_v_term):.2f}, Expected Call Payoff = {emp_payoff:.2f}")
print()

# (g) Log-returns validation
log_returns = np.log(S_terminal / S0)
th_log_mean = (mu_q2 - 0.5 * sigma_q2**2) * T_q2
th_log_var = (sigma_q2**2) * T_q2

print("Q2(g) Log-Returns Analysis:")
print(f"  Empirical Log-Return Mean: {np.mean(log_returns):.4f} (Theoretical: {th_log_mean:.4f})")
print(f"  Empirical Log-Return Variance: {np.var(log_returns):.4f} (Theoretical: {th_log_var:.4f})\n")