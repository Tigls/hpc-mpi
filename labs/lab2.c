#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>

double function(double x)
{
    return 5 * x - pow(sin(x), 2);
}

bool check_Runge(double I2, double I, double epsilon)
{
    return (fabs(I2 - I) / 3.) < epsilon;
}

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

void write_double_to_file(const char* filename, double data)
{
    FILE* fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Failed to open the file\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(fp, " % lg\n", data);
    fclose(fp);
}
int main(int argc, char* argv[])
{
    int np;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double input[3];
    if (rank == 0)
    {
        FILE* fp = fopen("in.txt", "r");
        if (fp == NULL)
        {
            printf("Failed to open the file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < 3; i++)
            fscanf(fp, " %lg", &input[i]);
        fclose(fp);
    }
    MPI_Bcast(input, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double start = input[0];
    double finish = input[1];
    double epsilon = input[2];
    double step = (finish - start) / np;
    double res = integrate_parabola_method(start + rank * step, start +
                                                               (rank + 1) * step, epsilon / np);
    if (rank != 0)
    {
        MPI_Send(&res, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    }
    if (rank == 0)
    {
        MPI_Request recv_reqs[np-1]; // np - 1
        MPI_Status status[np-1];
        double resall[np-1];
        for (int i = 0; i < (np - 1); i++)
        {
            MPI_Irecv(&resall[i], 1, MPI_DOUBLE, (i + 1), (i + 1), MPI_COMM_WORLD, &recv_reqs[i]);
        }
        MPI_Waitall(np - 1, recv_reqs, status);
        for (int i = 0; i < (np - 1); i++)
        {
            res += resall[i];
        }
        write_double_to_file("out.txt", res);
    }
    MPI_Finalize();
    return 0;
}
