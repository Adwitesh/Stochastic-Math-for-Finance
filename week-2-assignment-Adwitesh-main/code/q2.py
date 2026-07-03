import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Calculate the true value using scipy
true_val, _ = quad(lambda x: np.exp(-x**2), 0, 1)
print(f"True Analytical Integral Value: {true_val:.8f}\n")

N_values = [10**2, 10**3, 10**4, 10**5]
estimates = []
errors = []

for N in N_values:
    # Sample uniformly from [0, 1]
    U = np.random.uniform(0, 1, N)
    
    # (b) Monte Carlo estimator
    mc_est = np.mean(np.exp(-U**2))
    estimates.append(mc_est)
    
    # (c) Compute absolute error
    abs_error = np.abs(mc_est - true_val)
    errors.append(abs_error)
    
    print(f"N = {N:6d} | Estimate: {mc_est:.6f} | Absolute Error: {abs_error:.6f}")

# (d) Plot Monte Carlo estimate vs N
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.axhline(y=true_val, color='r', linestyle='--', label='True Value')
plt.plot(N_values, estimates, marker='o', color='b', label='MC Estimate')
plt.xscale('log')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Estimated Value')
plt.title('Monte Carlo Estimate vs N')
plt.legend()

# Plot Error decay for better interpretation
plt.subplot(1, 2, 2)
plt.plot(N_values, errors, marker='s', color='purple')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Absolute Error')
plt.title('Convergence Rate (Log-Log)')

plt.tight_layout()
plt.savefig('outputs/q2_integration.png')
plt.show()