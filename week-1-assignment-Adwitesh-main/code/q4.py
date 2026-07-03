import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Market Parameter Definitions
regime_transitions = np.array([
    [0.60, 0.30, 0.10, 0.00],
    [0.20, 0.50, 0.20, 0.10],
    [0.10, 0.30, 0.40, 0.20],
    [0.05, 0.15, 0.30, 0.50]
])

regime_rewards = np.array([0.012, 0.004, -0.006, -0.025])
regime_states = [0, 1, 2, 3]

total_weeks = 10000
current_regime = 0  # Initialize in Bull Regime (State 0)

# Historical tracking setups
regime_path_history = np.zeros(total_weeks, dtype=int)
regime_path_history[0] = current_regime

# Run long-term single path sequence simulation
for week in range(1, total_weeks):
    active_row_distribution = regime_transitions[current_regime]
    current_regime = np.random.choice(regime_states, p=active_row_distribution)
    regime_path_history[week] = current_regime

# Calculate Empirical Frequencies
counts_per_regime = np.zeros(4)
for r in regime_path_history:
    counts_per_regime[r] += 1
empirical_frequencies = counts_per_regime / total_weeks

# Map regimes directly to returns and compile portfolio trajectory statistics
weekly_realized_returns = np.zeros(total_weeks)
for idx in range(total_weeks):
    state_visited = regime_path_history[idx]
    weekly_realized_returns[idx] = regime_rewards[state_visited]

empirical_mean_return = np.mean(weekly_realized_returns)

# Generate Cumulative Performance Sequence
cumulative_wealth_addition = np.zeros(total_weeks)
running_total = 0.0
for idx in range(total_weeks):
    running_total += weekly_realized_returns[idx]
    cumulative_wealth_addition[idx] = running_total

# Theoretical stationary distribution values from eigenvector math for comparison check
theoretical_pi = np.array([0.2791, 0.3372, 0.2326, 0.1512])
theoretical_mean_return = np.sum(theoretical_pi * regime_rewards)

print("      SIMULATION METRICS     ")
print("Empirical Frequencies vs. Theoretical Pi:")
for state_id in range(4):
    print(f" Regime {state_id}: Sim={empirical_frequencies[state_id]:.4f} | Theo={theoretical_pi[state_id]:.4f}")
print(f"Empirical Average Return   : {empirical_mean_return:.6f}")
print(f"Theoretical Average Return : {theoretical_mean_return:.6f}")

# Ensure output workspace folder structure is available
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Plot structural performance configurations
plt.figure(figsize=(10, 5))
plt.plot(cumulative_wealth_addition, color='tab:blue', linewidth=1.5, label='Portfolio Path')
plt.title('Cumulative Portfolio Return Over Time ($10,000$ Weeks)', fontsize=12, fontweight='bold')
plt.xlabel('Weeks Running', fontsize=10)
plt.ylabel('Accumulated Nominal Portfolio Addition Value', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left')
plt.plot()