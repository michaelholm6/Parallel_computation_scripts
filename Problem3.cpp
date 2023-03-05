#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 50
#define CHUNKSIZE 5

int main (int argc, char *argv[])
{
    int i, chunk, tid;
    float a[N], b[N], c[N];

    /*Some initializations*/

    for (i=0; i<N; i++)
       a[i] = b[i] = i * 1.0;
    chunk = CHUNKSIZE;

    /*There were two problems in this section. One is that the #pragma omp parallel
    line doesn't need to have brackets after it. This is because it acts on the 
    next for loop. The second problem is that the omp_get_thread_num() functions
    needs to be inside a parallel section. This parallel section doesn't start until
    after the for loop starts.*/
    #pragma omp parallel for shared(a,b,c,chunk) private(i, tid) schedule(static, chunk)
        for (i=0; i< N; i++)
        {
            tid = omp_get_thread_num();
            c[i] = a[i] + b[i];
            printf("tid = %d i = %d c[i] = %f\n", tid, i, c[i]);
        }
        return 0;
}