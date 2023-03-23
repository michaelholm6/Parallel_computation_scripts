/*=======================================================================*/
/* Approximates pi with the n-point quadrature rule 4 / (1+x**2)         */
/* applied to the integral of x from 0 to 1.                             */
/*=======================================================================*/

//#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;

const double M_pi = 3.14159265358979323846; /* reference value */

/*Serial version of estimating pi*/
double calc_pi (unsigned n) {
  double h   = 1.0 / n;
  double sum = 0.0;
  double x;
  int i;
  
  auto startSerial = chrono::high_resolution_clock::now();
  for (i=0; i<n; i++) {
    x = (i + 0.5) * h;
    sum += 4.0 / ( 1.0 + x*x );
  }
    auto endSerial = chrono::high_resolution_clock::now();
    double serial_time_taken = chrono::duration_cast<chrono::nanoseconds>(endSerial-startSerial).count();
    printf("Serial time taken: %f \n", serial_time_taken);
  return h * sum;
}



/*Parallel version of estimating pi*/
double calc_pi_parallel (unsigned n, int threads) {
  double h   = 1.0 / n;
  double sum = 0.0;
  double x;
  int i;

auto startParallel = chrono::high_resolution_clock::now();

  #pragma omp parallel for firstprivate(x) reduction(+: sum) num_threads(threads)
  for (i=0; i<n; i++) {
    x = (i + 0.5) * h;
    sum += 4.0 / ( 1.0 + x*x );
  }

auto endParallel = chrono::high_resolution_clock::now();
double parallel_time_taken = chrono::duration_cast<chrono::nanoseconds>(endParallel-startParallel).count();
printf("Parallel time taken: %f \n", parallel_time_taken);
return h * sum;
}


/*In the main function, I'm executing both a serial and a parallel version, 
outputting the error from both methods, and the time taken. */
int main(int argc, char* argv[]) {
  int n;

  if ( argc != 3 ) {
    fprintf(stderr, "usage: pi <num_iterations>\n");
    return 1;
  }

  n = atoi(argv[1]);


  if ( n > 0 ) {
    double pi = calc_pi(n);
    double err = pi - M_pi;
    printf("Calculated serial pi is %19.15f\n", pi);
    printf("Referenced pi is %19.15f\n", M_pi);
    printf("  Error in serial pi is %19.15f (%f%%)\n", err, err*100/M_pi);
  }
  
int threads;
threads = atoi(argv[2]);

if ( n > 0 ) {
    double pi = calc_pi_parallel(n, threads);
    double err = pi - M_pi;
    printf("Calculated parallel pi is %19.15f\n", pi);
    printf("Referenced parallel pi is %19.15f\n", M_pi);
    printf("  Error in parallel pi is %19.15f (%f%%)\n", err, err*100/M_pi);
  }

  return 0;
}


