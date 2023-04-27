import numpy as np
import time
import multiprocessing as mp
import math
import itertools
import datetime

def generate_data_array(data_points, dimensions):
    """Randomly generates a data array of normally distrubted data points.

    Args:
        data_points (_type_): Number of samples in the array to generate.
        dimensions (_type_): Number of dimensions per sample in the array to generate.
    """
    data_array = np.random.randn(data_points, dimensions)
    np.savetxt('generated_data.csv', data_array, delimiter=',')
    
def normalize_column(column: int):
    """Normalizes a column of data by subtracting the mean and dividing by the standard deviation.

    Args:
        column (int): Column to normalize.

    Returns:
        _type_: Normalized column.
    """
    global data_array
    try:
        x = data_array[0,0]
    except NameError:
        data_array = np.genfromtxt('generated_data.csv', delimiter=',')
    column = data_array[:, column]
    mean = np.average(column)
    column -= mean
    std_dev = np.std(column)
    column /= std_dev
    return column
    
def calculate_covariance(i, j):
    """Calculates the covariance between two columns of data.

    Args:
        i (_type_): First column to calculate covariance between.
        j (_type_): Second column to calculate covariance between.

    Returns:
        _type_: Covariance between the two columns.
    """
    global standardized_data
    try:
        x = standardized_data[0,0]
    except NameError:
        standardized_data = np.genfromtxt('standardized_data.csv', delimiter=',')
    row = standardized_data[:, i]
    column = standardized_data[:, j]
    if i == j:
        return np.var(row), i, j
    elif i < j:
        covariance = sum(row * column) / len(row)
        return covariance, i, j
            
def multiply_matrices(i, j):
    """Multiplies two matrices together.

    Args:
        i (_type_): Row to multiply.
        j (_type_): Column to multiply.

    Returns:
        _type_: Result of the multiplication.
    """
    global eigenvector_data
    global standardized_data
    try:
        x = eigenvector_data[0,0]
    except NameError:
        eigenvector_data = np.genfromtxt('eigenvector_data.csv', delimiter=',')
    try:
        x = standardized_data[0,0]
    except NameError:
        standardized_data = np.genfromtxt('standardized_data.csv', delimiter=',')
    row = standardized_data[i, :]
    column = eigenvector_data[:, j]
    answer = sum(row*column)
    return answer, i, j

def parallel_PCA(out_dims, generate_processes, cpus):
    """Performs PCA on a data array in parallel.

    Args:
        out_dims (_type_): Output dimensions.
        generate_processes (_type_): Boolean to determine whether to generate processes.
        cpus (_type_): Number of cpus to use.

    Returns:
        _type_: Output data array.
    """
    start_time = time.time()
    
    data_array = np.genfromtxt('generated_data.csv', delimiter=',')
    
    if generate_processes == 1:
        pool = mp.Pool(processes=cpus)
    
    chunk_size = math.floor(np.shape(data_array)[1]/8) if math.floor(np.shape(data_array)[1]/8) != 0 else 1
    data_array = pool.map(normalize_column, [i for i in range(len(data_array[1]))], chunksize=chunk_size)
    data_array = np.transpose(np.array(data_array))
    
    np.savetxt('standardized_data.csv', data_array, delimiter=',')
    
    row_list = [i for i in range(len(data_array[0]))]
    covariance_index_list = list(itertools.product(row_list, row_list))
    covariance_index_list = [i for i in covariance_index_list if i[1] >= i[0]]
    chunk_size = math.floor(len(covariance_index_list)/8) if  math.floor(len(covariance_index_list)/8) != 0 else 1
    covariance_array = pool.starmap(calculate_covariance, covariance_index_list, chunksize=chunk_size)
    del(row_list, covariance_index_list)
    
    empty_covariance_array = np.zeros([np.shape(data_array)[1], np.shape(data_array)[1]])
    
    for element in covariance_array:
        empty_covariance_array[element[1], element[2]] = element[0]
        if element[1] != element[2]:
            empty_covariance_array[element[2], element[1]] = element[0]
    
    covariance_array = empty_covariance_array
    del(empty_covariance_array)
    
    eigenvals, eigenvecs = np.linalg.eig(covariance_array)

    eigdict = {}
    for i in range(len(eigenvals)):
        eigdict[eigenvals[i]] = eigenvecs[:, i]   
    del (eigenvals, eigenvecs)
    
    sorted_eig_vecs = np.zeros([len(covariance_array[0]), len(covariance_array[0])])
    
    i = 0
    for key, value in (sorted(eigdict.items(), reverse=True)):
        sorted_eig_vecs[:, i] = value
        i += 1
    del(i, eigdict)
    
    np.savetxt('eigenvector_data.csv', sorted_eig_vecs[:, 0: out_dims], delimiter=',')
    
    out_data_points_list = [i for i in range(len(data_array))]
    out_dims_list = [i for i in range(out_dims)]
    out_array_indeces = list(itertools.product(out_data_points_list, out_dims_list))
    del(out_data_points_list, out_dims_list)
    
    chunk_size = math.floor(len(out_array_indeces)/8) if  math.floor(len(out_array_indeces)/8) != 0 else 1
    multiply_matrices_result = pool.starmap(multiply_matrices, out_array_indeces, chunksize=chunk_size) 
    
    result = np.zeros([np.shape(data_array)[0], out_dims])
    
    for answer in multiply_matrices_result:
        result[answer[1], answer[2]] = answer[0]    
    
    pool.terminate()
    time_end = time.time() - start_time
    
    return time_end
    
def evaluate_pca_parallel_time(n_dimension_range: tuple, n_dimension_step: int, n_sample_size_range: tuple, n_sample_size_step: int, generate_processes:bool, cpus: int):
    """Evaluate PCA execution time over a meshgrid of input dimensions.

    Args:
        n_dimension_range (tuple):
        n_dimension_step (int):
        n_sample_size_range (tuple):
        n_sample_size_step (int):
        generate_processes (bool): Boolean to say whether or not this function will generate processes upon execution. This is just to stop the script from generating processes
        upon import.
        cpus (int): Argument for how many CPUs to use when evaluating the parallel PCA.
    """
    
    @np.vectorize
    def Z_function(samples, dimensions):
        """Evaluate PCA execution time over a meshgrid of input dimensions.

        Args:
            samples (_type_): Numpy array of sample sizes to evaluate.
            dimensions (_type_): Numpy array of dimensions to evaluate.

        Returns:
            _type_: Numpy array of dimensions, samples, and execution times.
        """
        generate_data_array(samples, dimensions)
        print('Parallel Samples: ' + str(samples) + ' Parallel Dimensions: ' + str(dimensions) + ' Start Time: ' + str(datetime.datetime.now()))
        total_time = parallel_PCA(dimensions, generate_processes, cpus)
        return total_time
    
    dimensions = range(n_dimension_range[0], n_dimension_range[1], n_dimension_step)
    samples = range(n_sample_size_range[0], n_sample_size_range[1], n_sample_size_step)
    dimensions, samples = np.meshgrid(dimensions, samples)
    times = Z_function(samples, dimensions)      

    return dimensions, samples, times
            
if __name__ == "__main__":
    evaluate_pca_parallel_time((100, 600), 100, (100, 600), 100, 1, 4)
    