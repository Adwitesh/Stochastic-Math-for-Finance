import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
simulations = 10000
n_values = [10, 50, 200]

plt.figure(figsize=(15, 5))

for i, n in enumerate(n_values, 1):
    # Simulate Bernoulli trials (0 or 1 with p=0.5)
    X = np.random.binomial(1, 0.5, size=(simulations, n))
    Sn = np.sum(X, axis=1)
    
    # Standardize
    Zn = (Sn - n / 2) / np.sqrt(n / 4)
    
    # (g) Estimate empirical mean and variance
    emp_mean = np.mean(Zn)
    emp_var = np.var(Zn)
    print(f"n = {n:3d} | Empirical Mean: {emp_mean:7.4f} | Empirical Variance: {emp_var:7.4f}")
    
    # (d) Plot histograms
    plt.subplot(1, 3, i)
    plt.hist(Zn, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Simulated')
    
    # (e) Overlay standard Gaussian density
    x = np.linspace(-4, 4, 200)
    plt.plot(x, norm.pdf(x), 'r-', lw=2, label='$\mathcal{N}(0,1)$')
    
    plt.title(f'Empirical CLT (n = {n})')
    plt.xlabel('$Z_n$')
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.savefig('outputs/q1_clt.png')
plt.show()