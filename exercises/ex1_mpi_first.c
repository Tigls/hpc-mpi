#include <stdio.h>
#include <mpi.h>

int ex1(int args, char* argv[])
{
	MPI_Init(&args, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int np;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	printf("I am task %d. There are %d tasks total.\n", rank, np);
	if (rank == 1)
	{
		printf("I am task 1 and I do things differently.\n");
	}
	MPI_Finalize();
	return 0;
}