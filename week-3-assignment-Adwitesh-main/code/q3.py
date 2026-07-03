import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set random seed for reproducibility
np.random.seed(42)

# Ensure the outputs directory exists
os.makedirs('../outputs', exist_ok=True)

print("--- Running Question 3 Simulations ---")

# (d) Simulate 10 paths of the OU process
X0, theta_q3, sigma_q3, T_q3, N_q3 = 3.0, 2.0, 0.5, 5, 5000
dt_q3 = T_q3 / N_q3
t_axis = np.linspace(0, T_q3, N_q3 + 1)
paths_count = 10

plt.figure(figsize=(10, 5))
for i in range(paths_count):
    X_path = np.zeros(N_q3 + 1)
    X_path[0] = X0
    for k in range(N_q3):
        dW_q3 = np.random.normal(0, np.sqrt(dt_q3))
        X_path[k+1] = X_path[k] - theta_q3 * X_path[k] * dt_q3 + sigma_q3 * dW_q3
    plt.plot(t_axis, X_path, alpha=0.6)

# Superimpose mean and stationary bounds
stat_std = sigma_q3 / np.sqrt(2 * theta_q3)
plt.axhline(0, color='black', linestyle='--', lw=1.5, label='Stationary Mean (0)')
plt.fill_between(t_axis, -2*stat_std, 2*stat_std, color='gray', alpha=0.15, label=r'$\pm 2$ Std Dev Band')
plt.xlabel('Time (t)')
plt.ylabel('$X_t$')
plt.title('Q3(d): Simulated Paths of the Ornstein-Uhlenbeck Process')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Save the path plot
plt.savefig('../outputs/q3_ou_paths.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/q3_ou_paths.png")
plt.show()

# (e) Verify stationary distribution empirically over 20,000 paths at T=10
n_paths_stat, T_long, N_long = 20000, 10, 5000
dt_long = T_long / N_long
X_term = np.full(n_paths_stat, float(X0))

for k in range(N_long):
    dW_long = np.random.normal(0, np.sqrt(dt_long), n_paths_stat)
    X_term = X_term - theta_q3 * X_term * dt_long + sigma_q3 * dW_long

th_stat_var = sigma_q3**2 / (2 * theta_q3)

print("Q3(e) Stationary Distribution Metrics (at T=10):")
print(f"  Empirical Mean: {np.mean(X_term):.4f} (Theoretical: 0.0)")
print(f"  Empirical Variance: {np.var(X_term):.4f} (Theoretical: {th_stat_var:.4f})\n")

# Overlay plot
plt.figure(figsize=(10, 5))
plt.hist(X_term, bins=100, density=True, alpha=0.6, color='lightgreen', label='Empirical $X_{10}$ Distribution')
x_axis_ou = np.linspace(min(X_term), max(X_term), 500)
plt.plot(x_axis_ou, stats.norm.pdf(x_axis_ou, 0, np.sqrt(th_stat_var)), 'r-', lw=2, label='Theoretical $N(0, \sigma^2 / 2\\theta)$ Density')
plt.xlabel('Terminal Value ($X_{10}$)')
plt.ylabel('Density')
plt.title('Q3(e): Empirical Terminal Distribution vs. Stationary Theory')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the distribution plot
plt.savefig('../outputs/q3_ou_stationary_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/q3_ou_stationary_distribution.png")
plt.show()

# (f) Subplots analyzing different theta impacts
thetas = [0.5, 2.0, 5.0]
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for idx, th in enumerate(thetas):
    for i in range(5):  # plot 5 paths per setting
        X_v = np.zeros(N_q3 + 1)
        X_v[0] = X0
        for k in range(N_q3):
            dW_v = np.random.normal(0, np.sqrt(dt_q3))
            X_v[k+1] = X_v[k] - th * X_v[k] * dt_q3 + sigma_q3 * dW_v
        axes[idx].plot(t_axis, X_v, alpha=0.7)
    
    th_v_std = sigma_q3 / np.sqrt(2 * th)
    axes[idx].axhline(0, color='k', linestyle='--')
    axes[idx].fill_between(t_axis, -2*th_v_std, 2*th_v_std, color='gray', alpha=0.1)
    axes[idx].set_ylabel(f'$\\theta = {th}$')
    axes[idx].set_title(f'OU Paths with $\\theta = {th}$ (Stationary Bounds width: $\\pm{2*th_v_std:.3f}$)')
    axes[idx].grid(True, alpha=0.3)

plt.xlabel('Time (t)')
plt.tight_layout()

# Save the comparison subplots
plt.savefig('../outputs/q3_theta_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/q3_theta_comparison.png")
plt.show()