#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;

#define VECLEN 100000000



int main(int argc, char* argv[])
{
    /*Necessary initializations*/
    int i, len = VECLEN;
    double *a, *b;
    double sumSerial, sumParallel = 0.0;

    /*Creating vectors to dot product*/
    printf("starting omp_dotprod_serial\n");
    a = (double*) malloc (len*sizeof(double));
    b = (double*) malloc (len*sizeof(double));
    
    for (i; i<len; i++)
    {
        a[i] = 10.0;
        b[i] = a[i];
    }


    /*Starting serial dot product*/
    auto startSerial = chrono::high_resolution_clock::now();
    for (i=0; i<len; i++)
    {
        sumSerial += (a[i] * b[i]);
    }
    auto endSerial = chrono::high_resolution_clock::now();
    double time_taken_serial = chrono::duration_cast<chrono::nanoseconds>(endSerial-startSerial).count();
    printf("done. Serial version: sum = %f \n", sumSerial);
    printf("Time taken: %f \n", time_taken_serial);



    /*Starting parallel dot product*/
    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(16) reduction(+:sumParallel)
    for(i=0; i<len; i++)
    {
        sumParallel += (a[i] * b[i]);
    }
    auto end = chrono::high_resolution_clock::now();
    double time_taken_parallel = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
    printf("done. Parallel version: sum = %f \n", sumParallel);
    printf("Time taken: %f \n", time_taken_parallel);


    free (a); free(b);
}