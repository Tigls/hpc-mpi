#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./labs/linalg.h"

const char *input_file_MA = "MA.txt";
const int ROW_TAG = 0x1;
/* Основна функція (програма обчислення визначника) */
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

    /* Створення та реєстрація типу даних для рядка елементів матриці */
    MPI_Datatype matrix_rows;
    MPI_Type_vector(n * part, 1, p, MPI_DOUBLE, &matrix_rows);
    MPI_Type_commit(&matrix_rows);
    /* Створення та реєстрація типу даних для структури вектора */
    MPI_Datatype vector_struct;
    MPI_Aint extent;
    MPI_Type_extent(MPI_INT, &extent); 		// визначення розміру в байтах
    MPI_Aint offsets[] = {0, extent};
    int lengths[] = {1, n+1};
    MPI_Datatype oldtypes[] = {MPI_INT, MPI_DOUBLE};
    MPI_Type_struct(2, lengths, offsets, oldtypes, &vector_struct);
    MPI_Type_commit(&vector_struct);

    /* Розсилка рядків матриці з задачі 0 в інші задачі */
    if (rank == 0) {
        MPI_Scatter(MA -> data, n * part, MPI_DOUBLE, MAh -> data, n * part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(MA);
    } else {
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, MAh->data, n * part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    free(MA);

    /* Поточне значення вектору l_i */
    struct my_vector *current_l = vector_alloc(n, .0);
    /* Частина рядків матриці L */
    struct my_matrix *MLh = matrix_alloc(n, part, .0);
    /* Основний цикл ітерації (кроки) */
    for(int pivot = 0; pivot < n - 1; pivot++)
    {
        /* Вибір задачі, що містить стовпець з ведучім елементом та обчислення
        * поточних значень вектору l_i */
        if (pivot / part == rank)
        {
            int row_index = pivot;
            MLh -> data[pivot * part + row_index] = 1.;
            for(int i = pivot + 1; i < n; i++)
            {
                MLh -> data[i * part + row_index] = MAh -> data[i * part + row_index] /
                                                    MAh -> data[pivot * part + row_index];
            }
            for(int i = 0; i < n; i++)
            {
                current_l -> data[i] = MLh -> data[i * part + row_index];
            }
        }
        /* Розсилка поточних значень l_i */
        MPI_Bcast(current_l, 1, vector_struct, pivot % p, MPI_COMM_WORLD);
        /* Модифікація рядків матриці МА відповідно до поточного l_i */
        for(int i = pivot + 1; i < n; i++)
        {
            for(int j = 0; j < part; j++)
            {
                MAh -> data[i * part + j] -= MAh -> data[pivot * part + j] * current_l -> data[i];
            }
        }
    }
    /* Обислення добутку елементів, які знаходяться на головній діагоналі
    * основної матриці (з урахуванням номеру стовпця в задачі) */
    double prod = 1.;
    for(int i = 0; i < part; i++)
    {
        int row_index = i * p + rank;
        prod *= MAh -> data[row_index * part + i];
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
    MPI_Type_free(&matrix_rows);
    MPI_Type_free(&vector_struct);
    return MPI_Finalize();
}