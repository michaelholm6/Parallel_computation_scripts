import Serial_PCA as serial
import Parallel_PCA as parallel
import matplotlib.pyplot as plt

if __name__ == "__main__":
    evaluation_data_points = [[100, 200], 50, [100, 200], 50, 8]
    serial_dimensions, serial_samples, serial_times = serial.evaluate_pca_serial_time(evaluation_data_points[0], evaluation_data_points[1], evaluation_data_points[2], evaluation_data_points[3])
    parallel_dimensions, parallel_samples, parallel_times = parallel.evaluate_pca_parallel_time(evaluation_data_points[0], evaluation_data_points[1], evaluation_data_points[2], evaluation_data_points[3], 1, evaluation_data_points[4])
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_surface(serial_dimensions, serial_samples, serial_times, cmap = 'winter', edgecolor='none')
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Samples')
    ax.set_zlabel('Time (s)')
    ax.set_title('Serial Analysis')
    
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot_surface(parallel_dimensions, parallel_samples, parallel_times, cmap = 'winter', edgecolor='none')
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Samples')
    ax.set_zlabel('Time (s)')
    ax.set_title('Parallel Analysis \n CPUs = ' + str(evaluation_data_points[4]))
    
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot_surface(parallel_dimensions, parallel_samples, serial_times/parallel_times, cmap = 'winter', edgecolor='none')
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Samples')
    ax.set_zlabel('Speedup Factor')
    ax.set_title('Speedup Analysis \n CPUs = ' + str(evaluation_data_points[4]))
    
    plt.show()