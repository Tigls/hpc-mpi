#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./labs/linalg.h"

void forw_elim (double **origin, double *master_row, size_t n) {
    if(**origin == 0)
        return;
    double k = **origin / master_row[0];
    for (int i = 0; i < n; i++) {
        (*origin)[i] = (*origin)[i] - k * master_row[i];
    }
    **origin = k;
}

const char *input_file_MA = "MA.txt";

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
        MPI_Scatter(MA->data, n * part, MPI_DOUBLE, MAh->data, n * part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//         free(MA);
    } else {
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, MAh->data, n * part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//         free(MA);
    }

    struct my_vector *current_l = vector_alloc(n, .0);
    struct my_matrix *MLh = matrix_alloc(n, part, .0);
    double * pivot_row = (double*) calloc(n, sizeof(double));
    int pivot = 0;


    /* LU-розклад */
    for(int i = 0; i < n - 1; i++, pivot++)
    {
        int row_index = pivot % part;
        int offset = row_index * n;
        if (rank == pivot / part) {
            for (int r = 0; r < n; r++) {
                pivot_row[r] = MAh->data[row_index+r];
            }
        }
        MPI_Bcast(pivot_row, n, MPI_DOUBLE, pivot / part, MPI_COMM_WORLD);

        if (MAh -> data[offset + pivot] == 0) {
            printf("Zero pivot detected. Pivot element can not be zero");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        for(int j = pivot + 1; j < n; j++) {
            if (rank == j % p){
                int k = (j % part) * n;
                double *save = &MAh->data[k + pivot];
                forw_elim(&save, &pivot_row[pivot], n - pivot);
            }
        }
    }
    // Вивід на LU-матриці на екран
    double *LU_matrix = (double *)calloc(n*n, sizeof(double));
    MPI_Allgather(&MAh->data[0], n * part, MPI_DOUBLE, LU_matrix, n * part, MPI_DOUBLE, MPI_COMM_WORLD);
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

    double determinant = 0;
    /* Згортка добутків елементів головної діагоналі та вивід результату в задачі 0 */
    if(rank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, &prod, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);
        printf("\nDeterminant - %2.1f", prod);
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

