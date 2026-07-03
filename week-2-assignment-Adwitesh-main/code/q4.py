import numpy as np
import matplotlib.pyplot as plt

# Given Model Inputs
S0 = 100
K = 110
simulations = 100000
volatilities = [0.1, 0.2, 0.3, 0.5]

# Setup mapping dictionary to collect plots
results = {}

plt.figure(figsize=(14, 10))

for idx, sigma in enumerate(volatilities):
    # (b) Simulate Z ~ N(0, 1) and terminal stock price ST
    Z = np.random.normal(0, 1, simulations)
    ST = S0 * np.exp(sigma * Z)  # Assuming simple geometric growth scaling model specified
    
    # (c) Compute call option payoffs
    payoffs = np.maximum(ST - K, 0)
    
    # (d) Estimate expected option payoff
    expected_payoff = np.mean(payoffs)
    print(f"Volatility (\u03c3): {sigma:.1f} | Expected Option Payoff: {expected_payoff:.4f}")
    
    # (e) & (f) Keep track for visualization plots (Highlighting standard sigma=0.2 case requested)
    if np.isclose(sigma, 0.2):
        # Stock distribution
        plt.subplot(2, 2, 1)
        plt.hist(ST, bins=50, color='darkblue', alpha=0.7, edgecolor='black')
        plt.axvline(x=K, color='r', linestyle='--', label=f'Strike (K={K})')
        plt.title('Terminal Stock Prices ($S_T$) at $\sigma=0.2$')
        plt.legend()
        
        # Payoff distribution
        plt.subplot(2, 2, 2)
        plt.hist(payoffs, bins=50, color='darkorange', alpha=0.7, edgecolor='black')
        plt.title('Option Payoffs at $\sigma=0.2$')

    # Collect outcomes for visual comparisons cross-volatility (g)
    plt.subplot(2, 2, 3)
    plt.hist(ST, bins=50, alpha=0.4, label=f'$\sigma$={sigma}')
    
plt.subplot(2, 2, 3)
plt.title('Stock Price Spread comparison')
plt.xlabel('$S_T$')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/q4_option_pricing.png')
plt.show()