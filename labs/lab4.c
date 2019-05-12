#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define MATRIX_INPUT_FILE "matrix.txt"
#define RESULT_FILE "det_result.txt"

int lab4(int argc, char* argv[])
{
    /* Execution start time */
    clock_t start_clock;

    /* MPI Initialization */
    MPI_Init(&argc, &argv);

    /* Get total number of processes and current process rank */
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Source matrix. Used in process 0 only */
    double *matrix;
    double *B;
    double *Y;
    double *X;
    double *matrixLU_row;
    double* l;

    /* Source matrix size */
    int n;

    /* Read matrix in process 0 */
    if(rank == 0)
    {
        /* Open file for reading */
        FILE *file = fopen(MATRIX_INPUT_FILE, "r");
        if (file == NULL)
        {
            printf("Unable to open input file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Read matrix size */
        fscanf(file, "%d", &n);

        /* Check if matrix size is a multiple number of processors */
        if (n % p != 0)
        {
            printf("Matrix size must be a multiple number of processors");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        /* Allocate matrix */
        matrix = (double*) calloc(n * n, sizeof(double));
        B = (double*) calloc(n, sizeof(double));

        /* Total count of matrix elements for each process */
        int block_size = n * n / p;

        /* Fill matrix with elements from file in order of columns cycles */
        for (int i = 0; i < n * n; i++)
            // row number:	i % n
            // column number:	i / n * p % n + i / block_size
            fscanf(file, "%lf", &matrix[i % n * n + i / n * p % n + i / block_size]);

        for (int i = 0; i < n; i++)
            fscanf(file, "%lf", &B[i]);

        /* Close file */
        fclose(file);

        /* Initialize start time value */
        start_clock = clock();
    }

    // Send matrix size to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Total count of matrix elements for each process */
    int block_size = n * n / p;
    double *block = (double*) calloc(block_size, sizeof(double));

    /* Send a block of matrix elements to each process using "Cycles of rows" scattering */
    if (rank == 0)
        MPI_Scatter(matrix, block_size, MPI_DOUBLE, block, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, block, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Perform matrix LU decomposition */
    for (int pivot = 0; pivot < n - 1; pivot++)
    {
        /* Array of coefficients for rows below pivot row*/
        const int l_size = n - pivot - 1;
        l = (double*) calloc(l_size, sizeof(double));

        /* Determine process that contains pivot column */
        if (rank == pivot % p)
        {
            /* Offset of pivot column in the block */
            int offset = pivot / p * n;

            /* Check pivot for zero */
            if (0 == block[offset + pivot])
            {
                printf("Zero pivot detected. Pivot element can not be zero");
                MPI_Abort(MPI_COMM_WORLD, 3);
            }

            /* Fill array with coefficients values */
            for (int i = 0; i < l_size; i++)
                l[i] = block[offset + pivot + i + 1] / block[offset + pivot];
        }

        /* Send array of coefficients to all processes */
        MPI_Bcast(l, l_size, MPI_DOUBLE, pivot % p, MPI_COMM_WORLD);

        /* Subtract pivot row multiplied with corresponding coefficient from each row below pivot row
         * Accept changes only for columns to the right of the pivot column */
        for (int column = rank > pivot ? 0 : (pivot - rank) / p; column < n / p; column++)
        {
            /* Offset of current column in the block */
            int offset = column * n;
            /* Perform subtraction */
            for (int i = 0; i < l_size; i++)
                block[offset + pivot + i + 1] -= l[i] * block[offset + pivot];
        }
    }

    /* Calculate partial determinant value in each process using available columns */
    double *matrixLU = NULL;
    if (rank == 0) {
        matrixLU = (double*) calloc(n * n, sizeof(double));
    }
    /* int num_col = n / p;
    for (int i = 0; i < num_col; i++) {
        MPI_Gather(
                block + i * sizeof(double) * n,
                block_size / num_col, MPI_DOUBLE,
                matrixLU + i * sizeof(double) * n * n / num_col,
                block_size / num_col,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } */

    MPI_Gather(block, block_size, MPI_DOUBLE, matrixLU, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Write result to file */
    if (rank == 0)
    {

        matrixLU_row = (double*) calloc(n * n, sizeof(double));
        Y = (double*) calloc(n, sizeof(double));
        X = (double*) calloc(n, sizeof(double));

        for (int i = 0; i < n * n; i++) {
            matrixLU_row[i] = matrixLU[i % n * n + i / n * p % n + i / block_size];
        }
        //***** FINDING Y; LY=b*********//
        for (int i = 0; i < n; i++) {
            Y[i] = B[i];
            for (int j = 0; j < i; j++) {
                Y[i] -= l[i*j] * Y[j];
            }
        }
        printf("\n\n[Y]: \n");
        for (int i = 0; i < n; i++) {
            printf("%9.3f", Y[i]);
        }

        //********** FINDING X; UX=Y***********//
        for (int i = n - 1; i >= 0; i--) {
            X[i] = Y[i];
            for (int j = i + 1; j < n; j++) {
                X[i] -= matrixLU_row[i*j] * X[j];
            }
            X[i] /= matrixLU_row[i*i];
        }

        /* Calculate execution time in milliseconds*/
        clock_t milliseconds = (clock() - start_clock) * 1000 / CLOCKS_PER_SEC;

        /* Create file */
        FILE *file = fopen(RESULT_FILE, "w");
        if (file == NULL)
        {
            printf("Failed to open output result file");
            MPI_Abort(MPI_COMM_WORLD, 4);
        }

        /* Write result */
        printf("\n\nVector [X]: \n");
        for (int i = 0; i < n; i++) {
            printf("%9.3f", X[i]);
        }
        fprintf(file, "Number of processes:\t%d\n", p);
        fprintf(file, "Execution time:\t%ld milliseconds\n", milliseconds);

        /* Close file */
        fclose(file);
    }

    return MPI_Finalize();
}