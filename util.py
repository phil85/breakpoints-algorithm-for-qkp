import numpy as np


def load_instance(file_name):
    # Open file
    f = open(file_name, 'r')
    lines = f.readlines()

    # Read first line which contains number of items and profits and data type of profits
    n_items, n_profits, dtype = lines[0].strip('\n').split(' ')
    n_items, n_profits = int(n_items), int(n_profits)

    # Initialize set of items
    items = list(range(n_items))

    # Read profits
    profits = {}
    for i in range(1, n_profits + 1):
        i, j, weight = lines[i].strip('\n').split(' ')
        if dtype == 'int':
            profits[int(i), int(j)] = int(float(weight))
        else:
            profits[int(i), int(j)] = float(weight)

    # Read weights
    weights = [int(v) for v in lines[n_profits + 1].strip('\n').strip().split(' ')]

    # Read budgets
    budgets = [int(v) for v in lines[n_profits + 2].strip('\n').strip().split(' ')]

    # Close file
    f.close()

    # Return items, profits, weights, budgets
    return items, profits, weights, budgets


def compute_total_profit(items, n_items, profits):

    # Compute a dense profit matrix
    profit_matrix = np.zeros((n_items, n_items))

    keys = np.fromiter(profits.keys(), dtype=[('row', int), ('col', int)])
    rows, cols = keys['row'], keys['col']
    values = np.array(list(profits.values()))

    profit_matrix[rows, cols] = values
    profit_matrix[cols, rows] = values

    # Set diagonal elements to zero
    linear_utilities = np.diagonal(profit_matrix).copy()
    np.fill_diagonal(profit_matrix, 0)

    # Add linear utilities
    items = np.array(items, dtype=int)
    total_profit = linear_utilities[items].sum()

    # Add quadratic utilities
    submatrix = profit_matrix[items]
    quadratic_utilities = submatrix[:, items].sum() / 2
    total_profit += quadratic_utilities

    return total_profit