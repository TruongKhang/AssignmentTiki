import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(3)

"""
    Sampling from mixed gaussian distribution
    Input:
        weights: a list of weights in which each element represents the contribution 
                    of the respective gaussian distribution
        list_gauss_dist: a list of tuples in which each tuple is a pair (mean, covariance_matrix)
        size: number of samples to take (default: 1)
    Return:
        <size> samples generated from mixed gaussian distribution 
"""
def sample_mixed_gauss_dist(weights, list_gauss_dist, size=1):
    samples = list()
    for s in range(size):
        # step 1: sampling an indicator i from a categorical distribution
        one_hot_vector = np.random.multinomial(1, weights)
        i = np.where(one_hot_vector == 1)[0][0]
        # step 2: sampling from gaussian distribution i
        mean_i, cov_i = list_gauss_dist[i]
        data_point = np.random.multivariate_normal(mean_i, cov_i)
        samples.append(data_point)
    return samples

"""
    Calculate value of probability density function (pdf) of mixed gaussian distribution
    Input:
        X: value of vector random variable, in this case is [x, y]
        weights: weights of mixed gaussian distribution
        list_gauss_dist: list of components in mixed distribution in which each component is gauss distribution
    Return:
        value of pdf
"""
def mixed_gauss_dist(X, weights, list_gauss_dist):
    pdf = 0
    for i, weight in enumerate(weights):
        mean_i, cov_i = list_gauss_dist[i]
        X = X - mean_i
        power = np.sum(X**2) #np.linalg.multi_dot([X.reshape(1,-1), np.linalg.inv(cov_i), X.reshape(-1,1)])
        pdf_gauss = np.exp(-0.5*power)/ np.pi #(np.pi*np.sqrt(np.linalg.det(cov_i)))
        pdf += weight * pdf_gauss
    return pdf


"""
    Initialize randomly the mean values for gaussian disstributions
    Each gaussian distribution is a bivariate-distribution of (x, y) with mean (mean_x, mean_y)
    Input: 
        num_dist: number of gaussian distribution 
        min_val: minimum value 
        max_val: maximum value
    Output:
        numpy array shaped (num_dist, 2)
"""
def generate_mean_values(num_dist, min_val, max_val):
    arr_mean = np.random.uniform(low=min_val, high=max_val, size=(num_dist, 2))
    return arr_mean

"""
    Initialize randomly the weights for mixed gaussian distribution
"""
def generate_weights(num_dist):
    weights = np.random.rand(num_dist)
    weights /= np.sum(weights)
    return weights

"""
    Generate dataset following the requirements 
"""
def generate_data():
    # The first mixed gaussian distribution 
    weights_0 = generate_weights(5)
    list_mean_values_0 = generate_mean_values(5, -1, 0)
    cov_maxtrix = np.array([[1,0], [0,1]]) # assume that X and Y are independent
    list_gauss_dist0 = [(mean_i, cov_maxtrix) for mean_i in list_mean_values_0]
    data_0 = sample_mixed_gauss_dist(weights_0, list_gauss_dist0, size=100)

    # The second mixed gaussian distribution
    weights_1 = generate_weights(5)
    list_mean_values_1 = generate_mean_values(5, 0, 1)
    cov_maxtrix = np.array([[1,0], [0,1]]) # assume that X and Y are independent
    list_gauss_dist1 = [(mean_i, cov_maxtrix) for mean_i in list_mean_values_1]
    data_1 = sample_mixed_gauss_dist(weights_1, list_gauss_dist1, size=100)
    return (data_0, weights_0, list_gauss_dist0), (data_1, weights_1, list_gauss_dist1)

if __name__ == '__main__':
    dist0, dist1 = generate_data()
    data_0, data_1 = dist0[0], dist1[0]
    color = ['r', 'b']
    for i, data in enumerate([data_0, data_1]):
        label = color[i]
        for point in data:
            plt.scatter(point[0], point[1], c=label)
    plt.show()
    