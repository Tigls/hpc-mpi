/*
 * Лабораторна робота 4, варіант 9
 * з дисціпліни Високопродуктивні обчислення
 * Хорт Дмитро
 * ФІОТ 5 курс, група ІП-з82мп
 * 05.2019
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./labs/linalg.h"

const char *input_file_MA = "input.txt";
void forward_elimination (double **origin, double *master_row, size_t n) {
    if(**origin == 0)
        return;
    double k = **origin / master_row[0];
    for (int i = 0; i < n; i++) {
        (*origin)[i] = (*origin)[i] - k * master_row[i];
    }
    **origin = k;
}

/* Основна функція (обчислення визначника) */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    struct my_matrix *MA;
    int n;

    /* Зчитування матриці з файлу */
    if(rank == 0)
    {
        MA = read_matrix(input_file_MA);
        if(MA -> rows != MA -> cols) {
            fatal_error("Matrix is not square!", 4);
        }
        n = MA -> rows;
    }

    /* Розсилка всім задачам розмірності матриць та векторів */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int part = n / p; // кількість рядків, що зберігається в даній задачі
    struct my_matrix *MAh = matrix_alloc(n, part, .0);

    /* Розсилка рядків матриці з задачі 0 в інші задачі */
    if (rank == 0) {
        MPI_Scatter(MA->data, n * part, MPI_DOUBLE, MAh->data, n * part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, MAh->data, n * part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    double * pivot_row = (double*) calloc(n, sizeof(double));
    int pivot = 0;

    /* LU-розклад */
    for(int i = 0; i < n - 1; i++, pivot++)
    {
        int row_index = pivot % part;
        int offset = row_index * n;

        /* Перевірка діагоналі на нульові елементи */
        if (MAh -> data[offset + pivot] == 0) {
            printf("Zero pivot detected. Pivot element can not be zero");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }


        /* Розсилка ведучого рядка */
        if (rank == pivot / part) {
            for (int r = 0; r < n; r++) {
                pivot_row[r] = MAh->data[offset+r];
            }

        }
        MPI_Bcast(pivot_row, n, MPI_DOUBLE, pivot / part, MPI_COMM_WORLD);


        /* Застосування методу Гаусса */
        for(int j = pivot + 1; j < n; j++) {
            if (rank == j / part){
                int offset1 = (j % part) * n;
                double *save = &MAh->data[offset1 + pivot];
                forward_elimination(&save, &pivot_row[pivot], n - pivot);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Вивід на LU-матриці в термінал
    double *LU_matrix = (double *)calloc(n*n, sizeof(double));
    MPI_Allgather(&MAh->data, n * part, MPI_DOUBLE, LU_matrix, n * part, MPI_DOUBLE, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("LU-Matrix\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%2.2f ", LU_matrix[i*n + j]);
            }
            printf("\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Обислення детермінанта */
    double prod = 1.;
    for (int i = 0; i < n; i++) {
        if (rank == i / part) {
            prod *= MAh->data[(i % part) * n + i];
        }
    }

    /* Згортка добутків елементів головної діагоналі та вивід результату в задачі 0 */
    if(rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &prod, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
        printf("\nDeterminant - %2.1f", prod);
    }
    else {
        MPI_Reduce(&prod, NULL, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    }

    return MPI_Finalize();
}

