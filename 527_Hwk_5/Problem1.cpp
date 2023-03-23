#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include<cmath>

using namespace std;

//creating matrices
int matrix1[5] = {1, 2, 3, 4, 5};
int matrix2[5] = {1, 2, 3, 4, 5};

//creating parallel addition script
int* matrix_addition_parallel(int* firstMatrix, int* secondMatrix){
int* Answer = new int[sizeof(firstMatrix)];
//starting timing
auto start = chrono::high_resolution_clock::now();
//running on gpu
#pragma omp target
{
//parallel for loop
#pragma omp parallel for
for (int i = 0; i < sizeof(firstMatrix); i++)
{
    Answer[i] = firstMatrix[i] + secondMatrix[i];
}
}
//end timing
auto end = chrono::high_resolution_clock::now();
float time_taken_parallel_addition = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
printf("Addition time taken in nanoseconds: %f \n", time_taken_parallel_addition);
return Answer;
}

//parallel matrix multiplication
int* matrix_multiplication(int* firstMatrix, int* secondMatrix)
{
int* Answer = new int[sizeof(firstMatrix)];
//start timing
auto start = chrono::high_resolution_clock::now();
int test;
//run on gpu
#pragma omp target
{
    //parallel for loop
#pragma omp parallel for
for (int i = 0; i < sizeof(firstMatrix); i++)
{
    Answer[i] = firstMatrix[i] * secondMatrix[i];
}
}
//end timing
auto end = chrono::high_resolution_clock::now();
//calculate time
float time_taken_parallel_multiplication = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
printf("Multiplication time taken in nanoseconds: %f \n", time_taken_parallel_multiplication);
return Answer;
}

int main()
{
    int* addition_answer = matrix_addition_parallel(matrix1, matrix2);
    int* multiplication_answer = matrix_multiplication(matrix1, matrix2);
    return 0;
}


