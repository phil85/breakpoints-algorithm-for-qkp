from breakpoints_algorithm import run_bp_algorithm

# Define knapsack capacity (or budget)
budgets = [8]

# Define items (or nodes) with ids 0, 1, ..., n
items = [0, 1, 2, 3]

# Define weights for items
weights = [2, 3, 4, 5]

# Define profits for including items i and j in the knapsack
profits = {(0, 0): 5,
           (0, 1): 1,
           (0, 2): 2,
           (0, 3): 3,
           (1, 1): 7,
           (1, 2): 1,
           (1, 3): 2,
           (2, 2): 8,
           (2, 3): 1,
           (3, 3): 10}

# Define parameters
params = {'n_lambda_values': 1600}

# Run the breakpoints algorithm
results = run_bp_algorithm(items, profits, weights, budgets, params)

# Print results
print('Objective function value: {:.1f}'.format(results.loc[0, 'ofv']))
print('Selected items: {:}'.format(results.loc[0, 'items']))
