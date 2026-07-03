import numpy as np

# Set seed for reproducible results
np.random.seed(42)
num_simulations = 10000

# Given analytical parameters
p_loss_b = 0.20
p_loss_a_given_b = 0.35

# Calculate conditional probability P(A | B^c) via Law of Total Probability
# P(A) = P(A|B)*P(B) + P(A|B^c)*P(B^c) -> 0.10 = 0.35*0.20 + P(A|B^c)*0.80
p_loss_a_given_not_b = (0.10 - (p_loss_a_given_b * p_loss_b)) / (1.0 - p_loss_b)

# Tracking arrays for indicators
loss_b_sims = np.zeros(num_simulations)
loss_a_sims = np.zeros(num_simulations)

for i in range(num_simulations):
    # Step 1: Sample loss status for asset B
    if np.random.rand() < p_loss_b:
        loss_b_sims[i] = 1
        # Step 2: Sample asset A conditionally if B lost
        if np.random.rand() < p_loss_a_given_b:
            loss_a_sims[i] = 1
    else:
        loss_b_sims[i] = 0
        # Step 2: Sample asset A conditionally if B did not lose
        if np.random.rand() < p_loss_a_given_not_b:
            loss_a_sims[i] = 1

# Total number of losses across both assets
total_losses = loss_a_sims + loss_b_sims

# Compute empirical metrics from arrays
both_loss_count = 0
either_loss_count = 0

for i in range(num_simulations):
    if loss_a_sims[i] == 1 and loss_b_sims[i] == 1:
        both_loss_count += 1
    if loss_a_sims[i] == 1 or loss_b_sims[i] == 1:
        either_loss_count += 1

prob_intersection = both_loss_count / num_simulations
prob_union = either_loss_count / num_simulations
expected_losses = np.mean(total_losses)
variance_losses = np.var(total_losses)

# Output Comparison Table
print(f"P(L_A intersection L_B) : {prob_intersection:.4f} (Theoretical: 0.0700)")
print(f"P(L_A union L_B)     : {prob_union:.4f} (Theoretical: 0.2300)")
print(f"Expectation E[N] : {expected_losses:.4f} (Theoretical: 0.3000)")
print(f"Variance Var(N)      : {variance_losses:.4f} (Theoretical: 0.3500)")