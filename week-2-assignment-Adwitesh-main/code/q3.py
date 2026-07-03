import numpy as np
import matplotlib.pyplot as plt

N_values = [10**2, 10**3, 10**4, 10**5]
true_pi = np.pi

print(f"True Value of Pi: {true_pi:.5f}\n")

estimates = []

for N in N_values:
    # (b) Generate points uniformly in square [-1, 1] x [-1, 1]
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    
    # Check if inside unit circle
    inside_circle = (x**2 + y**2) <= 1
    
    # (c) Estimate pi
    pi_est = 4 * np.sum(inside_circle) / N
    estimates.append(pi_est)
    
    # (g) Compare with true value
    print(f"N = {N:6d} | Estimated Pi: {pi_est:.5f} | Deviation: {np.abs(pi_est - true_pi):.5f}")
    
    # (d) Save visualization for N=10^3 to keep it clear but detailed
    if N == 1000:
        plt.figure(figsize=(6, 6))
        plt.scatter(x[inside_circle], y[inside_circle], color='g', s=5, label='Inside')
        plt.scatter(x[~inside_circle], y[~inside_circle], color='r', s=5, label='Outside')
        # Draw explicit circle boundary
        theta = np.linspace(0, 2*np.pi, 150)
        plt.plot(np.cos(theta), np.sin(theta), color='black', lw=2)
        plt.title(f'Sample Points for N = {N}')
        plt.axis('equal')
        plt.legend()
        plt.savefig('outputs/q3_circle_scatter.png')

# (e) Plot estimated value as a function of N
plt.figure(figsize=(8, 4))
plt.axhline(y=true_pi, color='r', linestyle='--', label='True $\pi$')
plt.plot(N_values, estimates, marker='o', color='b', label='MC Estimate')
plt.xscale('log')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Estimated $\pi$')
plt.title('$\pi$ Estimate Convergence')
plt.legend()
plt.savefig('outputs/q3_pi_convergence.png')
plt.show()