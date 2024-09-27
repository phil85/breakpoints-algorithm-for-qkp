from breakpoints_algorithm import run_bp_algorithm
from util import load_instance

# Load a problem instance from a file
items, profits, weights, budgets = load_instance('data/synthetic_tf_2000.txt.txt')

# Define parameters for the breakpoints algorithm
params = {'n_lambda_values': 1600}

# Run the breakpoints algorithm
results = run_bp_algorithm(items, profits, weights, budgets, params)

# Print results
print(results[['budget', 'ofv', 'cpu']])
