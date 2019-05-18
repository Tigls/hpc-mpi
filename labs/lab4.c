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
        // free(MA);
    } else {
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, MAh->data, n * part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // free(MA);
    }


    struct my_vector *current_l = vector_alloc(n, .0);
    struct my_vector *current_l1 = vector_alloc(n, .0);
    struct my_matrix *MLh = matrix_alloc(n, part, .0);
    /* Основний цикл ітерації (кроки) */
    for(int pivot = 0; pivot < n - 1; pivot++)
    {
        double main_pivot;
        int row_index = pivot % part;
        int offset = row_index * n;

        MLh -> data[offset + pivot] = 1.;
        if (MAh -> data[offset + pivot] == 0) {
            printf("Zero pivot detected. Pivot element can not be zero");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        if (rank == pivot / part) {
            main_pivot = MAh->data[offset + pivot];
        }
        MPI_Bcast(&main_pivot, 1, MPI_DOUBLE, pivot / part, MPI_COMM_WORLD);
        // Calc L coeficients
        for(int i = pivot + 1; i < n; i++) {
            if (rank == i % p){
                int k = (i % part) * n;
                MLh->data[k + pivot] = MAh->data[k + pivot] / main_pivot;
                if (pivot == 0) {
                    //printf("1arg MAL %2.3f\n", MLh->data[k + pivot]);
                }
            }
        }
        // Copy L coeficients to l-vector
        for(int i = 0; i < n; i++) {
            if(rank == i % p) {
                int k = (i % part) * n + pivot;
                current_l->data[i] = MLh->data[k];
                if(pivot ==0) {
                    //printf("rank %d l-vector %2.5f\n", rank, current_l->data[i]);
                }
            }
        }
        double * current_l2 = (double*) calloc(n, sizeof(double));
        for (int i = 0; i < n; i++) {
            MPI_Allgather(&MLh[1], 1, MPI_DOUBLE, current_l2, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if(rank == 2) {
            for(int i = 0; i <n; i++) {
                printf("rank %d l-vecot %2.2f\n", rank, current_l[i]);
            }
        }
        for(int i = 0; i < n; i++) {
            if (rank == i % p) {
                int k = (i % part) * n + pivot;
                MAh->data[k] -= MAh->data[k] * current_l->data[i];
                if(pivot ==0) {
                    //printf("final matrix %2.2f\n", MAh->data[k]);
                }
            }
        }
    }
    /* Обислення добутку елементів, які знаходяться на головній діагоналі
    * основної матриці (з урахуванням номеру стовпця в задачі) */
    double prod = 1.;
    for(int i = 0; i < part; i++)
    {
        int row_index = i % part + rank;
        prod *= MAh -> data[row_index*n + i];
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