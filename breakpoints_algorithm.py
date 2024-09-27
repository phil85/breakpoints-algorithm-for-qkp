# © 2024, University of Bern, Group of Business Analytics and Operations Research, Philipp Baumann

import re
import time
import subprocess
import numpy as np
import pandas as pd
from subprocess import PIPE


def compute_ofv_from_utility_matrix(nodes, utility_matrix, linear_utilities, left_nodes, beta):

    # Add linear utilities
    ofv = linear_utilities[left_nodes].sum()

    # Add quadratic utilities
    ofv += utility_matrix[left_nodes][:, left_nodes].sum() / 2

    # Subtract beta * cut
    other_nodes = np.setdiff1d(nodes, left_nodes)
    ofv -= beta * utility_matrix[left_nodes][:, other_nodes].sum()

    return ofv


def get_initial_node(right_nodes, utility_matrix, weights, budget):

    # Get candidates
    idx = weights[right_nodes] <= budget

    if not idx.any():
        return []

    candidate_nodes = right_nodes[idx]
    candidate_utilities = utility_matrix[candidate_nodes][:, candidate_nodes].sum(axis=1)
    candidate_utilities /= weights[candidate_nodes]
    return candidate_nodes[candidate_utilities.argmax()]


def run_greedy_left(utility_matrix, linear_utilities, nodes, left_nodes, right_nodes, budget, beta, weights,
                    initial_candidate_nodes=None, initial_candidate_contributions=None,
                    initial_current_total_weight=None):

    # Start stopwatch
    tic = time.perf_counter()

    # Initialize initial_left_nodes
    initial_left_nodes = left_nodes.copy()

    # Initialize left nodes if empty
    if len(left_nodes) == 0:
        initial_node = get_initial_node(right_nodes, utility_matrix, weights, budget)
        left_nodes = np.append(left_nodes, initial_node)
        left_nodes = left_nodes.astype(int)

    # Determine current total weights
    if initial_current_total_weight is None:
        current_total_weight = weights[left_nodes].sum()
    else:
        current_total_weight = initial_current_total_weight

    # Get candidate nodes
    if initial_candidate_nodes is None:
        candidate_nodes = np.setdiff1d(right_nodes, left_nodes)
    else:
        candidate_nodes = initial_candidate_nodes

    # Update candidate nodes
    idx = weights[candidate_nodes] <= budget - current_total_weight
    candidate_nodes = candidate_nodes[idx]

    # If all candidate nodes are within budget, we can update the initial values
    if idx.all():
        update_flag = True
    else:
        update_flag = False

    # Compute candidate contributions to ofv
    if initial_candidate_contributions is None:

        # Utilities to left nodes
        candidate_contributions = (1 + beta) * utility_matrix[candidate_nodes][:, left_nodes].sum(axis=1)

        # Linear utilities
        candidate_contributions += linear_utilities[candidate_nodes]

        if beta != 0:
            # Cut to other candidate nodes
            candidate_contributions -= beta * utility_matrix[candidate_nodes][:, candidate_nodes].sum(axis=1)

            # Cut to other nodes that are not candidate nodes and not in left nodes
            other_nodes = np.setdiff1d(nodes, right_nodes)
            candidate_contributions -= beta * utility_matrix[candidate_nodes][:, other_nodes].sum(axis=1)

        weights_subset = weights[candidate_nodes]
        candidate_contributions /= weights_subset
    else:
        candidate_contributions = initial_candidate_contributions[idx]

    # Start adding nodes
    for i in range(len(candidate_nodes)):

        # Break if there are no more candidate nodes
        if len(candidate_nodes) == 0:
            break

        # Get best node
        best_node = candidate_nodes[candidate_contributions.argmax()]

        # Update left nodes
        left_nodes = np.append(left_nodes, best_node)

        # Update current budget
        current_total_weight += weights[best_node]

        # Update candidate nodes and contributions
        idx1 = candidate_nodes != best_node
        idx2 = weights[candidate_nodes] <= budget - current_total_weight
        candidate_nodes = candidate_nodes[idx1 & idx2]
        candidate_contributions = candidate_contributions[idx1 & idx2]

        # Update candidate contributions
        if len(candidate_nodes) > 0:
            weights_subset = weights[candidate_nodes]
            candidate_contributions += ((1 + 2 * beta) * utility_matrix[candidate_nodes, best_node]) / weights_subset

            # Update initial values only if all candidate nodes are within budget
            if update_flag and idx2.all():
                initial_current_total_weight = current_total_weight
                initial_candidate_nodes = candidate_nodes
                initial_candidate_contributions = candidate_contributions
                initial_left_nodes = left_nodes.copy()
            else:
                update_flag = False

    # Stop stopwatch
    cpu = time.perf_counter() - tic

    return (left_nodes, initial_left_nodes, initial_candidate_nodes, initial_candidate_contributions,
            initial_current_total_weight, cpu)


def run_greedy_right(utility_matrix, linear_utilities, nodes, right_nodes, budget, beta, weights, left_nodes=None,
                     initial_candidate_nodes=None, initial_candidate_contributions=None,
                     initial_current_total_weight=None):

    # Start stopwatch
    tic = time.perf_counter()

    if len(right_nodes) == 0:
        return right_nodes, None, None, None, 0

    # Initialize left_nodes as empty if not provided
    if left_nodes is None:
        left_nodes = []

    # Compute current total weight
    if initial_current_total_weight is None:
        current_total_weight = weights[right_nodes].sum()
    else:
        current_total_weight = initial_current_total_weight

    # Get candidate nodes
    if initial_candidate_nodes is None:
        candidate_nodes = np.setdiff1d(right_nodes, left_nodes)
    else:
        candidate_nodes = initial_candidate_nodes.copy()

    # Compute candidate contributions to ofv
    if initial_candidate_contributions is None:
        weights_subset = weights[candidate_nodes]

        # Utilities to right nodes
        candidate_contributions = (-1 - beta) * utility_matrix[candidate_nodes][:, right_nodes].sum(axis=1)

        # Linear utilities
        candidate_contributions -= linear_utilities[candidate_nodes]

        if beta != 0:
            other_nodes = np.setdiff1d(nodes, right_nodes)
            candidate_contributions += beta * utility_matrix[candidate_nodes][:, other_nodes].sum(axis=1)
        candidate_contributions /= weights_subset
    else:
        candidate_contributions = initial_candidate_contributions.copy()

    for i in range(len(candidate_nodes)):

        # Break if there are no more candidate nodes
        if current_total_weight <= budget:
            break

        # Get best node
        best_node = candidate_nodes[candidate_contributions.argmax()]
        # ofv += candidate_contributions.max() * weights[best_node]

        # Update right_nodes
        idx = right_nodes != best_node
        right_nodes = right_nodes[idx]

        # Update current budget
        current_total_weight -= weights[best_node]

        # Update candidate nodes and contributions
        idx = candidate_nodes != best_node
        candidate_nodes = candidate_nodes[idx]
        candidate_contributions = candidate_contributions[idx]

        # Update candidate contributions
        weights_subset = weights[candidate_nodes]
        candidate_contributions += ((1 + 2 * beta) * utility_matrix[candidate_nodes, best_node]) / weights_subset

    # Stop stopwatch
    cpu = time.perf_counter() - tic

    return right_nodes, candidate_nodes, candidate_contributions, current_total_weight, cpu


def run_greedy(nodes, edges, weights, budgets, beta, breakpoints, breakpoint_weights):

    # Initialize lists for results
    greedy_left_nodes = []
    greedy_left_running_times = []
    greedy_right_nodes = []
    greedy_right_running_times = []

    # Compute a dense utility matrix
    utility_matrix = np.zeros((len(nodes), len(nodes)))
    rows, cols = np.array(list(edges.keys())).T
    values = np.array(list(edges.values()))
    utility_matrix[rows, cols] = values
    utility_matrix[cols, rows] = values

    # Set diagonal elements to zero
    linear_utilities = np.diagonal(utility_matrix).copy()
    utility_matrix[np.diag_indices_from(utility_matrix)] = 0

    # Initialize initial values (later used for warm start)
    initial_candidate_nodes = None
    initial_candidate_contributions = None
    initial_current_total_weight = None
    initial_left_nodes = []

    # Run greedy left algorithm for all budgets
    for budget in budgets:

        # Get left and right nodes based on breakpoints
        left_nodes = np.array(breakpoints[np.where(breakpoint_weights <= budget)[0][-1]])

        # Initialize candidate nodes
        if len(initial_left_nodes) < len(left_nodes):
            initial_candidate_nodes = None
            initial_candidate_contributions = None
            initial_current_total_weight = None
            initial_left_nodes = left_nodes.copy()

        # Run greedy left algorithm (use all nodes as right nodes)
        right_nodes = np.array(nodes)
        (selected_nodes, initial_left_nodes, initial_candidate_nodes, initial_candidate_contributions,
         initial_current_total_weight, cpu) = (run_greedy_left(utility_matrix, linear_utilities, nodes,
                                                               initial_left_nodes, right_nodes,
                                                               budget, beta, weights, initial_candidate_nodes,
                                                               initial_candidate_contributions,
                                                               initial_current_total_weight))

        # Store results
        greedy_left_nodes.append(selected_nodes)
        greedy_left_running_times.append(cpu)

    # Run greedy right algorithm for all budgets
    for i, budget in enumerate(budgets[::-1]):

        right_nodes = np.array(breakpoints[np.where(breakpoint_weights >= budget)[0][0]])

        # Initialize selected nodes in first iteration
        if i == 0:
            selected_nodes = right_nodes.copy()

        # Initialize initial values
        if len(selected_nodes) >= len(right_nodes):
            initial_candidate_nodes = None
            initial_candidate_contributions = None
            initial_current_total_weight = None
            selected_nodes = right_nodes.copy()

        selected_nodes, candidate_nodes, candidate_contributions, current_total_weight, cpu_right = (
            run_greedy_right(utility_matrix, linear_utilities, nodes, selected_nodes, budget, beta, weights,
                             left_nodes=None,
                             initial_candidate_nodes=initial_candidate_nodes,
                             initial_candidate_contributions=initial_candidate_contributions,
                             initial_current_total_weight=initial_current_total_weight))

        # Run greedy left algorithm (use all nodes as right nodes)
        right_nodes = np.array(nodes)
        selected_nodes, _, _, _, _, cpu_left = run_greedy_left(utility_matrix, linear_utilities, nodes, selected_nodes,
                                                               right_nodes, budget, beta, weights)

        # Store results
        greedy_right_nodes.append(selected_nodes)
        greedy_right_running_times.append(cpu_right + cpu_left)

    # Reverse greedy right nodes
    greedy_right_nodes.reverse()
    greedy_right_running_times.reverse()

    return greedy_left_nodes, greedy_left_running_times, greedy_right_nodes, greedy_right_running_times


def compute_ofv_from_nodes_and_edges(items, nodes, edges):

    # Compute a dense utility matrix
    utility_matrix = np.zeros((len(nodes), len(nodes)))
    rows, cols = np.array(list(edges.keys())).T
    values = np.array(list(edges.values()))
    utility_matrix[rows, cols] = values
    utility_matrix[cols, rows] = values

    # Set diagonal elements to zero
    linear_utilities = np.diagonal(utility_matrix).copy()
    utility_matrix[np.diag_indices_from(utility_matrix)] = 0

    # Add linear utilities
    items = np.array(items, dtype=int)
    ofv = linear_utilities[items].sum()

    # Add quadratic utilities
    ofv += utility_matrix[items][:, items].sum() / 2

    return ofv


def write_input_file_for_QKPsimparamHPF(nodes, edges, weights, n_lambda_values):

    # Write input file
    with open('input.txt', 'w') as f:
        n_nodes = len(nodes)
        n_edges = len(edges)
        edge_lines = ['e {} {} {:.5f}\n'.format(i, j, edges[(i, j)]) for (i, j) in edges]
        node_lines = ['n {} {:.5f} N/A\n'.format(i, weights[i]) for i in nodes]
        content = f'q {n_nodes} {n_edges} {n_lambda_values}\n' + ''.join(edge_lines) + ''.join(node_lines)
        f.write(content)


def read_output_file_of_QKPsimparamHPF():
    # Open file and read lines
    f = open('output.txt', 'r')
    lines = f.readlines()

    # Set node offset
    node_offset = 3

    # Get nodes that switch from sink set to source set at corresponding lambda value
    # breakpoint_sets = {0: []}
    breakpoint_sets = {}
    for line in lines:
        if not line.startswith('n'):
            continue
        letter, node_id, lambda_value_pos = line.rstrip('\r\n').split()
        # Skip lines that do not start with letter 'n'
        if letter != 'n':
            continue
        # Skip source and sink node
        if node_id == '1' or node_id == '2':
            continue
        # Add node to the corresponding breakpoint set
        position = int(lambda_value_pos)
        if position not in breakpoint_sets:
            breakpoint_sets[position] = []
        breakpoint_sets[position].append(int(node_id) - node_offset)

    return breakpoint_sets


def extract_time_for_parametric_cut(text):

    # Define a regular expression pattern to match the time format
    pattern = r'Time for performing parametric cut: (\d+\.\d+) seconds'

    # Search for the pattern in the input text
    match = re.search(pattern, text)

    # If a match is found, extract the time as a float
    if match:
        time_str = match.group(1)
        time_float = float(time_str)
        return time_float
    else:
        return None


def get_breakpoints(nodes, edges, weights, n_lambda_values):

    # Start stopwatch
    tic = time.perf_counter()

    # Write input file for simple parametric flow algorithm
    write_input_file_for_QKPsimparamHPF(nodes, edges, weights, n_lambda_values)

    # Stop stopwatch
    cpu_write = time.perf_counter() - tic

    # Start stopwatch
    tic = time.perf_counter()

    # Run simple parametric cut algorithm
    result = subprocess.run(['./QKPsimparamHPF.exe', 'input.txt', 'output.txt'], stdout=PIPE)

    # Record running time
    cpu_exe = time.perf_counter() - tic

    # Record cut time
    cpu_cut = extract_time_for_parametric_cut(result.stdout.decode('utf-8'))

    # Start stopwatch
    tic = time.perf_counter()

    # Read output file
    breakpoint_sets = read_output_file_of_QKPsimparamHPF()

    # Stop stopwatch
    cpu_read = time.perf_counter() - tic

    return breakpoint_sets, cpu_write, cpu_cut, cpu_exe, cpu_read


def run_bp_algorithm(nodes, edges, weights, budgets, n_lambda_values=1600):

    # Compute breakpoints
    breakpoint_sets_dict, cpu_write, cpu_cut, cpu_exe, cpu_read = get_breakpoints(nodes, edges, weights, n_lambda_values)

    # Extract information from breakpoint sets
    n_breakpoints = len(breakpoint_sets_dict)
    positions = sorted(breakpoint_sets_dict.keys())
    total_weights_at_breakpoints = np.cumsum([sum([weights[i] for i in breakpoint_sets_dict[p]]) for p in positions])

    # Put breakpoint sets in a list
    breakpoint_sets_as_list = [breakpoint_sets_dict[positions[i]] for i in range(n_breakpoints)]

    # Run greedy
    breakpoints = [np.array([])]
    current_sublist = []
    for ls in breakpoint_sets_as_list:
        current_sublist.extend(ls)
        breakpoints.append(np.array(current_sublist))
    total_weights_at_breakpoints_with_zero = np.concatenate(([0], total_weights_at_breakpoints))
    weights = np.array(weights)

    greedy_left_nodes, greedy_left_running_times, greedy_right_nodes, greedy_right_running_times = (
        run_greedy(nodes, edges, weights, budgets, 0, breakpoints, total_weights_at_breakpoints_with_zero))

    # Initialize results
    results = pd.DataFrame()

    # Save results for each budget
    for i, budget in enumerate(budgets):

        result = pd.Series(dtype=object)
        result['budget'] = budget
        result['budget_fraction'] = '{:.4f}'.format(budget / sum(weights))
        result['n_breakpoints'] = n_breakpoints
        result['total_weights_at_breakpoints'] = ([0] + list(total_weights_at_breakpoints))
        result['items_at_breakpoints'] = breakpoint_sets_as_list
        result['items_left'] = list(greedy_left_nodes[i])
        result['items_right'] = list(greedy_right_nodes[i])
        result['cpu_left'] = greedy_left_running_times[i]
        result['cpu_right'] = greedy_right_running_times[i]
        result['cpu_write'] = cpu_write
        result['cpu_cut'] = cpu_cut
        result['cpu_exe'] = cpu_exe
        result['cpu_read'] = cpu_read

        # Compute objective function values
        result['ofv_left'] = compute_ofv_from_nodes_and_edges(result['items_left'], nodes, edges)
        result['ofv_right'] = compute_ofv_from_nodes_and_edges(result['items_right'], nodes, edges)
        result['total_weight_left'] = sum([weights[i] for i in result['items_left']])
        result['total_weight_right'] = sum([weights[i] for i in result['items_right']])
        if result['ofv_left'] > result['ofv_right']:
            result['items'] = result['items_left']
            result['ofv'] = result['ofv_left']
            result['total_weight'] = result['total_weight_left']
        else:
            result['items'] = result['items_right']
            result['ofv'] = result['ofv_right']
            result['total_weight'] = result['total_weight_right']

        # Get total running time
        result['cpu'] = (cpu_write + cpu_exe + cpu_read + greedy_left_running_times[i] + greedy_right_running_times[i])

        # Convert to dataframe
        result = result.to_frame().transpose()

        # Append result to results using concat
        results = pd.concat((results, result))

    return results
