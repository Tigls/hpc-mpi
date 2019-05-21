/*
 * Лабораторна робота 2, варіант 9
 * з дисціпліни Високопродуктивні обчислення
 * Хорт Дмитро
 * ФІОТ 5 курс, група ІП-з82мп
 * 05.2019
*/

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>
#include <time.h>

double function(double x)
{
    return 5 * x - pow(sin(x), 2);
}

bool check_Runge(double I2, double I, double epsilon)
{
    return (fabs(I2 - I) / 3.) < epsilon;
}

/* Інтегрування методом парабол */
double integrate_parabola_method(double start, double finish, double epsilon)
{
    int num_iterations = 1;
    double last_res = 0.;
    double res = -1.;
    double h = 0;
    while (!check_Runge(res, last_res, epsilon))
    {
        num_iterations *= 2;
        last_res = res;
        res = 0.;
        double term2 = 0.;
        double term3 = 0.;
        h = (finish - start) / num_iterations;
        double term1 = function(start);
        for (int i = 1; i < num_iterations; i+=2)
        {
            term2 += function(start + i * h);
            term3 += function(start + (i+1) * h);
        }
        res += h / 3 * (term1 + 4*term2 + 2*term3);
    }
    return res;
}

int main(int argc, char* argv[])
{
    int np;
    int rank;
    clock_t start_clock;
    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double input[3];
    if (rank == 0)
    {
        FILE* fp = fopen("input.txt", "r");
        if (fp == NULL)
        {
            printf("Failed to open the file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < 3; i++)
            fscanf(fp, " %lg", &input[i]);
        fclose(fp);
        start_clock = clock();
    }
    MPI_Bcast(input, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double start = input[0];
    double finish = input[1];
    double epsilon = input[2];
    double step = (finish - start) / np;
    double result_part = integrate_parabola_method(start + rank * step, start + (rank + 1) * step, epsilon);
    double total_result;
    MPI_Reduce(&result_part, &total_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        clock_t milliseconds = (clock() - start_clock) * 1000 / CLOCKS_PER_SEC;
        printf("\nExecution time:\t%ld milliseconds\n", milliseconds);
        printf("Result %f", total_result);
    }

    MPI_Finalize();
    return 0;
}