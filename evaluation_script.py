import os
import pickle
import sys
import time

import networkx as nx
import numpy as np

def main():
    with open('amsterdam_data/pareto_front_osdpm.pickle', 'rb') as f:
        pareto_front = pickle.load(f)
    with open('amsterdam_data/graph_dict_osdpm.pickle', 'rb') as f:
        graph = pickle.load(f)

    score1, time1 = evaluate_track1(graph, pareto_front[0:5], pf_algorithm)

    print(f"Score 1: {score1}, Time 1: {time1}")

    num_samples = 100000

    param_dict = {
        0: {'a_mean': 0.05, 'a_std': 0.01, 'b_mean': 150, 'b_std': 10},  # First cost
        1: {'a_mean': 0.1, 'a_std': 0.02, 'b_mean': 10, 'b_std': 2},  # Second cost (kept low)
        2: {'a_mean': 0.05, 'a_std': 0.01, 'b_mean': 100, 'b_std': 10},  # Third cost (similar to first)
        3: {'a_mean': 0.05, 'a_std': 0.01, 'b_mean': 100, 'b_std': 10}  # Fourth cost (similar to first)
    }

    score2, time2 = evaluate_track2(5, pf_algorithm, graph, pareto_front[5:10], twopl_utility_func, param_dict, num_samples, 100000000)

    print(f"Score 2: {score2}, Time 2: {time2}")

def pf_algorithm(graph, node_a, node_b):
    pass


def twopl_utility_func(v_list, param_dict, num_samples,
                       default_params={'a_mean': 1.0, 'a_std': 1.0, 'b_mean': 0.0, 'b_std': 1.0}):
    """
    Monte Carlo integration to compute the two-parameter logistic utility.

    Parameters:
    v_list: List of tuples, each containing (cost_tuple, path)
      - cost_tuple is a tuple of 4 costs
      - path can be ignored
    param_dict: Dictionary of parameter settings for each of the 4 costs
    num_samples: Number of Monte Carlo samples
    default_params: Default parameters if not specified

    Returns:
    List of utility estimates for each input vector
    """

    def twoPL(v, a, b):
        """Two-parameter logistic function."""
        return 1.0 / (1.0 + np.exp(-a * (v - b)))

    # Initialize max product
    max_product = 0

    # Process each input vector
    for cost_tuple, _ in v_list:
        # Process each cost in the 4-tuple
        for i, cost in enumerate(cost_tuple):
            # Get parameters for this specific cost
            params = param_dict.get(i, default_params)
            a_mean = params.get('a_mean', default_params['a_mean'])
            a_std = params.get('a_std', default_params['a_std'])
            b_mean = params.get('b_mean', default_params['b_mean'])
            b_std = params.get('b_std', default_params['b_std'])

            # Sample parameters
            a_samples = np.random.normal(loc=a_mean, scale=a_std, size=num_samples)
            b_samples = np.random.normal(loc=b_mean, scale=b_std, size=num_samples)

            # Compute transformations for this cost
            trans_samples_i = twoPL(cost, a_samples, b_samples)

            for j, other_cost in enumerate(cost_tuple):
                if i == j:
                    continue

                # Get parameters for other cost
                other_params = param_dict.get(j, default_params)
                a_mean_j = other_params.get('a_mean', default_params['a_mean'])
                a_std_j = other_params.get('a_std', default_params['a_std'])
                b_mean_j = other_params.get('b_mean', default_params['b_mean'])
                b_std_j = other_params.get('b_std', default_params['b_std'])

                # Sample parameters for other cost
                a_samples_j = np.random.normal(loc=a_mean_j, scale=a_std_j, size=num_samples)
                b_samples_j = np.random.normal(loc=b_mean_j, scale=b_std_j, size=num_samples)

                # Compute transformations
                trans_samples_j = twoPL(other_cost, a_samples_j, b_samples_j)

                # Compute and update max product
                max_product = max(max_product, np.mean(trans_samples_i * trans_samples_j))

    # If only one vector, return its results directly
    return max_product


def evaluate_track1(graph, pareto_fronts, algorithm):
    # Call algorithm, time the call, get the time and the set of solutions
    # Check if the whole pareto front was found (1 or 0),
    # if there are any paths that are not in the pareto front
    # check if any of their costs are less than the costs of the paths in the pareto front
    # only check the costs that are in the pareto front and not the path node-per-node
    # return sum_i (pareto_found), total_time
    # the best one will be max pareto_found and if equal min total_time
    pareto_found = 0
    total_time = 0

    for index in range(len(pareto_fronts)):

        node_a = pareto_fronts[index]['source']
        node_b = pareto_fronts[index]['target']
        pareto_list = pareto_fronts[index]['pareto_set']

        current_time = time.time()
        solutions = algorithm(graph, node_a, node_b)
        total_time += time.time() - current_time

        for solution in solutions:
            sol_found = False
            for pareto in pareto_list:
                if all(x == y for x, y in zip(solution[0], pareto[0])):
                    pareto_found += 1
                    sol_found = True
                    break
            # we check if one of the costs is less than the lowest cost in that position in the pareto front
            if not sol_found:
                for pareto_cost, _ in pareto_list:
                    # Check if candidate dominates cost
                    if not all(c <= t for c, t in zip(solution[0], pareto_cost)) and any(c < t for c, t in zip(solution[0], pareto_cost)):
                        pareto_found += 1
                        print(f"Found a solution that is not in the pareto front: {solution[0]} vs {pareto_cost}")

    return pareto_found, total_time

def evaluate_track2(num_solutions, algorithm, graph, pareto_fronts, utility_func, param_dict, num_samples, max_time):
    # Call the algorithm, time the call, get the time and the set of solutions
    # if the algorithm exceeds the time limit, return inf (disqualified)
    # Check if the set of solutions is the correct number, calculate
    # the expected utility of the solutions
    # return sum_i (expected utility), total_time
    # the best one will be max expected utility and if equal min total_time
    total_time = 0
    total_utility = 0

    for index in range(len(pareto_fronts)):

        node_a = pareto_fronts[index]['source']
        node_b = pareto_fronts[index]['target']

        current_time = time.time()
        solutions = algorithm(graph, node_a, node_b)
        total_time += time.time() - current_time

        if total_time > max_time:
            return float('inf'), float('inf')

        if len(solutions) != num_solutions:
            return float('inf'), total_time

        total_utility += utility_func(solutions, param_dict, num_samples)

    return total_utility, total_time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
