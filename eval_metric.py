import numpy as np


def twoPL(v, a, b):
    """
    Two-parameter logistic function.
    twoPL(v; a, b) = 1 / (1 + exp(-a * (v - b)))
    """
    return 1.0 / (1.0 + np.exp(-a * (v - b)))


def evaluate_metric(v_list, a_mean, a_std, b_mean, b_std, num_samples):
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


if __name__ == "__main__":
    # Example usage:
    # List of vectors (each vector is a list of numbers)
    v_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

    # Gaussian prior parameters for 'a' and 'b'
    a_mean = 1.0
    a_std = 1.0
    b_mean = 1.0
    b_std = 1.0

    # Number of Monte Carlo samples
    num_samples = 10000000

    main(v_list, a_mean, a_std, b_mean, b_std, num_samples)
