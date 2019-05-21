/*
 * Лабораторна робота 3, варіант 9
 * з дисціпліни Високопродуктивні обчислення
 * Хорт Дмитро
 * ФІОТ 5 курс, група ІП-з82мп
 * 05.2019
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MATRIX_INPUT_FILE "input.txt"
#define RESULT_FILE "output.txt"

int main(int argc, char* argv[])
{
    clock_t start_clock;
    MPI_Init(&argc, &argv);
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Ініціалізація змінних */
    double *matrix;
    double *matrix_col;
    double *B;
    double *Y;
    double *X;
    double *matrixU_row;
    double *matrixL_row;
    double* l;
    double* l_sum;
    int n;

    /* Зчитування матриці */
    if(rank == 0)
    {
        FILE *file = fopen(MATRIX_INPUT_FILE, "r");
        if (file == NULL)
        {
            printf("Unable to open input file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(file, "%d", &n);

        /* Перевірка чи є розмір матриці мультиплікатором кількості процесів */
        if (n % p != 0)
        {
            printf("Matrix size must be a multiple number of processors");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        /* Виділення пам'яті під вектора та масиви */
        matrix = (double*) calloc(n * n, sizeof(double));
        matrix_col = (double*) calloc(n * n, sizeof(double));
        B = (double*) calloc(n, sizeof(double));
        l_sum = (double*) calloc(n * n, sizeof(double));;

        /* Кількість елементів матриці в кожному процесі */
        int block_size = n * n / p;

        /* Заповнюємо матрицю елементами циклічно по стовпчикам */
        for (int i = 0; i < n * n; i++) {
            fscanf(file, "%lf", &matrix[i]);
        }
        for (int i = 0; i < n * n; i++) {
            // номер рядка:	i % n
            // номер колонки:	i / n * p % n + i / block_size
            matrix_col[i] = matrix[i % n * n + i / n * p % n + i / block_size];
        }
        for (int i = 0; i < n; i++)
            fscanf(file, "%lf", &B[i]);

        fclose(file);
        start_clock = clock();
    }

    // Розсилка матриці усім процесам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Кількість елементів матриці в кожному процесі */
    int block_size = n * n / p;
    double *block = (double*) calloc(block_size, sizeof(double));

    /* Розсилка елементів процесам */
    if (rank == 0)
        MPI_Scatter(matrix_col, block_size, MPI_DOUBLE, block, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, block, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* LU-декомпозиція */
    for (int pivot = 0; pivot < n - 1; pivot++)
    {
        /* Масив елементів  */
        const int l_size = n - pivot - 1;
        l = (double*) calloc(l_size, sizeof(double));

        /* Розраховуємо L елементи */
        if (rank == pivot % p)
        {
            int offset = pivot / p * n;

            /* Перевірка ведучого елемента на 0 */
            if (0 == block[offset + pivot]) {
                printf("Zero pivot detected. Pivot element can not be zero");
                MPI_Abort(MPI_COMM_WORLD, 3);
            }

            for (int i = 0; i < l_size; i++) {
                l[i] = block[offset + pivot + i + 1] / block[offset + pivot];
            }
        }

        /* Розсилка масиву L усім процесам */
        MPI_Bcast(l, l_size, MPI_DOUBLE, pivot % p, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int i = 0; i < l_size; i++)
                l_sum[n * pivot + i + (pivot + 1)] = l[i];
        }


        /*  Віднімаємо ведучий рядок з коефіцієнтом L від усіх рядочків нижче за ведучий */
        for (int column = rank > pivot ? 0 : (pivot - rank) / p; column < n / p; column++) {
            int offset = column * n;
            for (int i = 0; i < l_size; i++)
                block[offset + pivot + i + 1] -= l[i] * block[offset + pivot];
        }
    }

    /* Збір матриці з усіх процесів */
    double *matrixLU = NULL;
    if (rank == 0) {
        matrixLU = (double*) calloc(n * n, sizeof(double));
    }

    MPI_Gather(block, block_size, MPI_DOUBLE, matrixLU, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Вивід результатів на знаходженяя рішення СЛАР */
    if (rank == 0)
    {
        clock_t milliseconds = (clock() - start_clock);
        printf("\nExecution time:\t%ld mikroseconds\n", milliseconds);
        matrixU_row = (double*) calloc(n * n, sizeof(double));
        matrixL_row = (double*) calloc(n * n, sizeof(double));
        Y = (double*) calloc(n, sizeof(double));
        X = (double*) calloc(n, sizeof(double));

        for (int i = 0; i < n * n; i++) {
            int a = i % n * n + i / n * p % n + i / block_size;
            matrixU_row[a] = matrixLU[i];
            matrixL_row[i % n*n + i/n] = l_sum[i];
        }
        printf("\nMatrix [U] \n");
        for (int i = 0; i < n * n; i++) {
            printf("%9.3f", matrixU_row[i]);
            if ((i+1) % n == 0) {
                printf("\n");
            }
        }
        printf("\nMatrix [L]\n");
        for (int i = 0; i < n * n; i++) {
            printf("%9.3f", matrixL_row[i]);
            if ((i+1) % n == 0) {
                printf("\n");
            }
        }
        /* Пошук Y; LY=b */
        for (int i = 0; i < n; i++) {
            Y[i] = B[i];
            for (int j = 0; j < i; j++) {
                Y[i] -= matrixL_row[i * n + j] * Y[j];
            }
        }
        printf("\n[Y]: \n");
        for (int i = 0; i < n; i++) {
            printf("%9.3f", Y[i]);
        }

        /* Пошук X; UX=Y */
        for (int i = n - 1; i >= 0; i--) {
            X[i] = Y[i];
            for (int j = i + 1; j < n; j++) {
                X[i] -= matrixU_row[i * n + j] * X[j];
            }
            X[i] /= matrixU_row[i * n + i];
        }

        FILE *file = fopen(RESULT_FILE, "w");
        if (file == NULL)
        {
            printf("Failed to open output result file");
            MPI_Abort(MPI_COMM_WORLD, 4);
        }

        printf("\n\n[X]: \n");
        for (int i = 0; i < n; i++) {
            printf("%9.3f", X[i]);
        }
        fprintf(file, "Number of processes:\t%d\n", p);
        fclose(file);
    }

    return MPI_Finalize();
}