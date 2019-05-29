/*
 * Лабораторна робота 1, варіант 9
 * з дисціпліни Високопродуктивні обчислення
 * Хорт Дмитро
 * ФІОТ 5 курс, група ІП-з82мп
 * 05.2019
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

const double EPSILON = 1E-8;
const int VALUE_TAG = 1;
const char* input_file_name = "input.txt";

double factorial(int value) {
    if (value < 0) {
        return NAN;
    }
    else if (value == 0) {
        return 1;
    }
    else {
        double fact = 1;
        for (int i = 2; i <= value; i++) {
            fact *= i;
        }
        return fact;
    }
}

double calc_series_term(int term_number, double value) {
    int k = 2 * term_number + 1;
    return (pow(-1, term_number) / factorial(k)) * pow(value, k);
}

int main(int argc, char* argv[]) {
    clock_t start_clock;
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    double x = 0;
    if (rank == 0)
    {
        FILE *input_file = fopen(input_file_name, "r");
        if (!input_file)
        {
            fprintf(stderr, "Can't open input file!\n\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        fscanf(input_file, "%lf", &x);
        fclose(input_file);
        start_clock = clock();
    }
    if (rank == 0)
    {
        for (int i = 1; i < np; i++)
            MPI_Send(&x, 1, MPI_DOUBLE, i, VALUE_TAG, MPI_COMM_WORLD);
    }
    else
        MPI_Recv(&x, 1, MPI_DOUBLE, 0, VALUE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    double sum = .0;
    double sum_temp = .0;
    int need_break = false;
    double element = 0;

    for (int step = rank+1; step < 100000; step += np) {
        element = calc_series_term(step, x);
        sum_temp += element;
        if (fabs(element) < EPSILON) {
            need_break = true;
            MPI_Bcast(&need_break, 1, MPI_INT, rank, MPI_COMM_WORLD);
        }
        if (need_break) {
            break;
        }
    }
    MPI_Reduce(&sum_temp, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        clock_t milliseconds = (clock() - start_clock);
        printf("\nExecution time:\t%ld mikroseconds\n", milliseconds);
        printf("\n%.15lf sin(%2.2f) lab1\n", sum+x, x);
        printf("\n%.15lf sin(%2.2f) math.h\n", sin(x), x);
    }
    MPI_Finalize();
    return 0;
}