#include <stdio.h>
#include <omp.h>
#include <iostream>

using namespace std;
int fib(int n)
{
  int x, y;
  if (n < 2) 
	  return n;

  x = fib(n - 1);

  y = fib(n - 2);

  return x+y;

}

/*Creating a parallel fib algo. Creates a task for the two recursive functions
that must be created. This is slower than the the serial version.*/
int fib_parallel(int n, int threads)
{

int x, y, tid, thread;
if (n < 2) 
    return n;

#pragma omp parallel num_threads(threads)

#pragma omp task
{
{
x = fib(n - 1);
}

{
y = fib(n - 2);
tid = omp_get_thread_num();
printf("Hello from thread = %d\n", tid);
}
}

return x+y;

}

/*Executing both the serial and the parallel version and, showing the results,
and times of both versions. */
int main()
{
  int n,fibonacci, threads;
  double starttime;
  printf("Please insert n, to calculate fib(n): \n");
  scanf("%d",&n);
  printf("Please insert threads, to calculate fib(n): \n");
  scanf("%d",&threads);
  starttime=omp_get_wtime();

  fibonacci=fib(n);

  printf("serial fib(%d)=%d \n",n,fibonacci);
  printf("serial calculation took %lf sec\n",omp_get_wtime()-starttime);

  starttime=omp_get_wtime();

  fibonacci=fib_parallel(n, threads);

  printf("parallel fib(%d)=%d \n",n,fibonacci);
  printf("parallel calculation took %lf sec\n",omp_get_wtime()-starttime);

  return 0;
}
