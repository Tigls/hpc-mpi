#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linalg.h"

const char *input_file_MA = "MA.txt";
const int ROW_TAG = 0x1;
/* Основна функція (програма обчислення визначника) */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int np, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    struct my_matrix *MA;
    int N;
    /* Зчитування матриці з файлу */
    if(rank == 0)
    {
        MA = read_matrix(input_file_MA);
        if(MA->rows != MA->cols) {
            fatal_error("Matrix is not square!", 4);
        }
        N = MA->rows;
    }
    /* Розсилка всім задачам розмірності матриць та векторів */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int part = N / np; // кількість рядків, що зберігається в даній задачі
    struct my_matrix *MAh = matrix_alloc(N, part, .0);

    /* Створення та реєстрація типу даних для рядка елементів матриці */
    MPI_Datatype matrix_rows;
    MPI_Type_vector(N*part, 1, np, MPI_DOUBLE, &matrix_rows);
    MPI_Type_commit(&matrix_rows);
    /* Створення та реєстрація типу даних для структури вектора */
    MPI_Datatype vector_struct;
    MPI_Aint extent;
    MPI_Type_extent(MPI_INT, &extent); 		// визначення розміру в байтах
    MPI_Aint offsets[] = {0, extent};
    int lengths[] = {1, N+1};
    MPI_Datatype oldtypes[] = {MPI_INT, MPI_DOUBLE};
    MPI_Type_struct(2, lengths, offsets, oldtypes, &vector_struct);
    MPI_Type_commit(&vector_struct);
    /* Розсилка рядків матриці з задачі 0 в інші задачі */
    if(rank == 0)
    {
        for(int i = 1; i < np; i++)
        {
            MPI_Send(&(MA->data[i]), 1, matrix_rows, i, ROW_TAG, MPI_COMM_WORLD);
        }
        /* Копіювання елементів рядків даної задачі */
        for(int i = 0; i < part; i++)
        {
            int row_index = i*np;
            for(int j = 0; j < N; j++)
            {
                MAh->data[j*part + i] = MA->data[j*N + row_index];
            }
        }
        free(MA);
    }
    else
    {
        MPI_Recv(MAh->data, N*part, MPI_DOUBLE, 0, ROW_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    /* Поточне значення вектору l_i */
    struct my_vector *current_l = vector_alloc(N, .0);
    /* Частина рядків матриці L */
    struct my_matrix *MLh = matrix_alloc(N, part, .0);
    /* Основний цикл ітерації (кроки) */
    for(int step = 0; step < N-1; step++)
    {
        /* Вибір задачі, що містить стовпець з ведучім елементом та обчислення
        * поточних значень вектору l_i */
        if (step % np == rank)
        {
            int col_index = (step - (step % np)) / np;
            MLh->data[step*part + col_index] = 1.;
            for(int i = step+1; i < N; i++)
            {
                MLh->data[i*part + col_index] = MAh->data[i*part + col_index] /
                                                MAh->data[step*part + col_index];
            }
            for(int i = 0; i < N; i++)
            {
                current_l->data[i] = MLh->data[i*part + col_index];
            }
        }
        /* Розсилка поточних значень l_i */
        MPI_Bcast(current_l, 1, vector_struct, step % np, MPI_COMM_WORLD);
        /* Модифікація рядків матриці МА відповідно до поточного l_i */
        for(int i = step+1; i < N; i++)
        {
            for(int j = 0; j < part; j++)
            {
                MAh->data[i*part + j] -= MAh->data[step*part + j] * current_l->data[i];
            }
        }
    }
    /* Обислення добутку елементів, які знаходяться на головній діагоналі
    * основної матриці (з урахуванням номеру стовпця в задачі) */
    double prod = 1.;
    for(int i = 0; i < part; i++)
    {
        int row_index = i*np + rank;
        prod *= MAh->data[row_index*part + i];
    }
    /* Згортка добутків елементів головної діагоналі та вивід результату в задачі 0 */
    if(rank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, &prod, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
        printf("%lf", prod);
    }
    else
    {
        MPI_Reduce(&prod, NULL, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
    }
    /* Повернення виділених ресурсів */
    MPI_Type_free(&matrix_columns);
    MPI_Type_free(&vector_struct);
    return MPI_Finalize();
}