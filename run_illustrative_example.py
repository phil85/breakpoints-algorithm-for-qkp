from breakpoints_algorithm import run_bp_algorithm

# Define the budget/capacity (you can specify multiple budgets)
budgets = [8]

# Define nodes (or elements) with ids 0, 1, ..., n - 1
elements = [0, 1, 2, 3]

# Define the node weights
weights = [2, 3, 4, 5]

# Define the utilities for selecting nodes i and j
utilities = {(0, 0): 1,
             (0, 1): 2,
             (0, 2): 11,
             (1, 1): 1,
             (1, 2): 3,
             (1, 3): 2,
             (2, 2): 1,
             (3, 3): 1}

# Run the breakpoints algorithm
results = run_bp_algorithm(elements, utilities, weights, budgets, n_lambda_values=1600)

# Print results
print('Objective function value: {:.1f}'.format(results.loc[0, 'ofv']))
print('Selected elements: {:}'.format(results.loc[0, 'elements']))
print('Total running time: {:.4f} seconds'.format(results.loc[0, 'cpu']))
print('Running time for simple parametric cut: {:.4f} seconds'.format(results.loc[0, 'cpu_cut']))
