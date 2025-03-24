import pickle
import time

import networkx as nx
import numpy as np

from ipro import subgradient
from ipro.ipro_dfs_amsterdam import DFSOracle
from ipro.outer_loops import IPRO


def main():
    with open('amsterdam_data/pareto_front_osdpm.pickle', 'rb') as f:
        pareto_front = pickle.load(f)
    with open('amsterdam_data/node_dict_osdpm.pickle', 'rb') as f:
        node_coords = pickle.load(f)
    with open('amsterdam_data/graph_dict_osdpm.pickle', 'rb') as f:
        graph = pickle.load(f)

    #score1, time1 = evaluate_track1(graph, pareto_front[0:5], ipro_dfs)
    #print(f"Score 1: {score1}, Time 1: {time1}")

    prior_values = {
        'a_mean': 1.0,
        'a_std': 0.1,
        'b_mean': 0.0,
        'b_std': 0.1,
        'num_samples': 10000
    }

    score2, time2 = evaluate_track2(5, ipro_dfs, graph, pareto_front[5:10], twopl_utility_func, prior_values, 100000000)
    print(f"Score 2: {score2}, Time 2: {time2}")

def ipro_dfs(graph, node_a, node_b):
    # Define fixed upper bounds for each objective (example values)
    upper_bounds = {'length': 5000, 'crosswalk': 5000, 'walk': 5000, 'bike': 5000}
    dimensions = 4
    G = nx.DiGraph()
    for node1 in graph:
        for node2 in graph[node1]:
            edge = graph[node1][node2]
            G.add_edge(
                node1,
                node2,
                length=edge[0],
                crosswalk=edge[1],
                walk= edge[2],
                bike=edge[3],
                vec=np.array(edge)
            )
    # Instantiate DFSOracle for this source–target pair
    oracle = DFSOracle(G, node_a, node_b, dimensions, lower_bounds_algorithm="reverse_dijkstra")
    # Reuse subgradient's DijkstraSolver as the linear solver component
    linear_solver = subgradient.DijkstraSolver(G, node_a, node_b, dimensions, upper_bounds)
    # Compute an ideal objective vector by running a per-objective Dijkstra search
    ideal = []
    for i in range(dimensions):
        path_i = oracle.dijkstra_shortest_path(i)
        if path_i:
            obj_vals = oracle.calculate_objectives(path_i)
            ideal.append(obj_vals[i])
        else:
            ideal.append(float('inf'))
    ideal = np.array(ideal)
    nadir = upper_bounds.copy()  # For simplicity, use upper_bounds as nadir
    print("Ideal objectives: ", ideal)
    # Create an IPRO instance using the DFSOracle and linear solver.
    ipro_instance = IPRO(
        problem_id="amsterdam_dfs",
        dimensions=dimensions,
        oracle=oracle,
        linear_solver=linear_solver,
        direction='minimize',
        max_iterations=10,
        tolerance=1e-8
    )
    pareto_set = ipro_instance.solve()
    print(f'Full Pareto front: {ipro_instance.pf}')
    print(f'Best solutions: {pareto_set}')
    modified_pf = [([-x for x in numbers], labels) for numbers, labels in pareto_set]
    return modified_pf

def twopl_utility_func(v_list, prior_values):
    """
    Monte Carlo integration to compute the integral
      ∫ max_v [ twoPL(v; a_i, b_i) * twoPL(v; a_j, b_j) ] p(a_i)p(b_i)p(a_j)p(b_j) da_i db_i da_j db_j
    where p() are Gaussian densities with the specified mean and std.

    Parameters:
      v_list: List of vectors (each vector is a list of numbers).
      a_mean, a_std: Mean and standard deviation for the Gaussian prior on a.
      b_mean, b_std: Mean and standard deviation for the Gaussian prior on b.
      num_samples: Number of Monte Carlo samples.
    """
    a_mean = prior_values['a_mean']
    a_std = prior_values['a_std']
    b_mean = prior_values['b_mean']
    b_std = prior_values['b_std']
    num_samples = prior_values['num_samples']

    def twoPL(v, a, b):
        """
        Two-parameter logistic function.
        twoPL(v; a, b) = 1 / (1 + exp(-a * (v - b)))
        """
        return 1.0 / (1.0 + np.exp(-a * (v - b)))

    # Flatten the list of vectors into one array of v values.
    v_all = [x for vec in v_list for x in vec]
    v_array = np.array(v_all)

    # Sample parameters from Gaussian priors using NumPy.
    a_i_samples = np.random.normal(loc=a_mean, scale=a_std, size=num_samples)
    b_i_samples = np.random.normal(loc=b_mean, scale=b_std, size=num_samples)
    a_j_samples = np.random.normal(loc=a_mean, scale=a_std, size=num_samples)
    b_j_samples = np.random.normal(loc=b_mean, scale=b_std, size=num_samples)

    # Compute the product twoPL(x; a_i, b_i) * twoPL(x; a_j, b_j) for each sample and for all v values.
    # Reshape the samples to (num_samples, 1) for broadcasting.
    prod = twoPL(v_array, a_i_samples[:, np.newaxis], b_i_samples[:, np.newaxis]) * \
           twoPL(v_array, a_j_samples[:, np.newaxis], b_j_samples[:, np.newaxis])

    # For each sample, take the maximum product over all v values.
    max_prod = np.max(prod, axis=1)

    # Estimate the integral as the average of the max values.
    integral_estimate = np.mean(max_prod)
    std_error = np.std(max_prod) / np.sqrt(num_samples)

    print("Monte Carlo Integral Estimate:", integral_estimate)
    print("Standard Error:", std_error)

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

    return pareto_found, total_time

def evaluate_track2(num_solutions, algorithm, graph, pareto_fronts, utility_func, prior_values, max_time):
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

        for solution in solutions:
            total_utility += utility_func(solution, prior_values)

    return total_utility, total_time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
