import numpy as np
import time
import matplotlib.pyplot as plt
import datetime

def generate_data_array(data_points, dimensions):
    data_array = np.random.randn(data_points, dimensions)
    np.savetxt('generated_data.csv', data_array, delimiter=',')

def serial_PCA(out_dims):
    start_time = time.time()
    data_array = np.genfromtxt('generated_data.csv', delimiter=',')
    
    for column in range(len(data_array[0])):
        mean = np.average(data_array[:, column])
        data_array[:, column] -= mean
        std_dev = np.std(data_array[:, column])
        data_array[:, column] /= std_dev
    covariance_matrix = np.zeros([len(data_array[0]), len(data_array[0])])
    
    for i in range(len(covariance_matrix[0])):
        for j in range(len(covariance_matrix[0])):
            if i == j:
                covariance_matrix[i, j] = np.var(data_array[:, j])
            elif i < j:
                covariance = sum(data_array[:, i] * data_array [:, j]) / len(data_array[:])
                covariance_matrix[i, j] = covariance
            elif i > j:
                covariance_matrix[i, j] = covariance_matrix[j, i]           
                
    eigenvals, eigenvecs = np.linalg.eig(covariance_matrix)
    
    eigdict = {}
    for i in range(len(eigenvals)):
        eigdict[eigenvals[i]] = eigenvecs[:, i]
    
    sorted_eig_vecs = np.zeros([len(covariance_matrix[0]), len(covariance_matrix[0])])
    
    i = 0
    for key, value in (sorted(eigdict.items(), reverse=True)):
        sorted_eig_vecs[:, i] = value
        i += 1
    del(i)
    
    output = np.zeros([np.shape(data_array)[0], out_dims])
    
    for i in range(len(data_array)):
        for j in range(out_dims):
            output[i, j] = sum(data_array[i, :] * sorted_eig_vecs[:, j])
    
    total_time = time.time() - start_time
    return output, total_time
    
def evaluate_pca_serial_time(n_dimension_range: tuple, n_dimension_step: int, n_sample_size_range: tuple, n_sample_size_step: int):
    
    @np.vectorize
    def Z_function(samples, dimensions):
        generate_data_array(samples, dimensions)
        print('Serial Samples: ' + str(samples) + ' Serial Dimensions: ' + str(dimensions) + ' Start Time: ' + str(datetime.datetime.now()))
        _, total_time = serial_PCA(dimensions)
        return total_time
    
    dimensions = range(n_dimension_range[0], n_dimension_range[1], n_dimension_step)
    samples = range(n_dimension_range[0], n_sample_size_range[1], n_sample_size_step)
    dimensions, samples = np.meshgrid(dimensions, samples)
    times = Z_function(samples, dimensions) 
    
    return dimensions, samples, times
            
if __name__ == "__main__":
    evaluate_pca_serial_time((10, 600), 100, (10, 600), 100)