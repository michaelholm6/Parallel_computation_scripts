#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>

using namespace std;

int main(){
//start timing
auto start = chrono::high_resolution_clock::now();
//run on gpu
#pragma omp target
{
#pragma omp parallel for
for (long i = 0; i < 1000000; i++){
    for (long j = 0; j < 2000000; i++){
        int tid = omp_get_thread_num();
        printf("Thread number: %d \n", tid);
    }
}
}
//end timing
auto end = chrono::high_resolution_clock::now();
//calculate time
float time_taken = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
//print time
printf("Addition time taken in nanoseconds: %f \n", time_taken);
}