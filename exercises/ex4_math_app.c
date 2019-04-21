#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

const double EPSILON = 1E-100;
const int VALUE_TAG = 1;
const int TERM_NUMBER_TAG = 2;
const int TERM_TAG = 3;
const int BREAK_TAG = 4;
const char* input_file_name = "in.txt";
const char* output_file_name = "out.txt";

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
	return pow(value, term_number) / factorial(term_number);
}

int ex4(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank;
	int np;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	double exponent;
	if (rank == 0) {
		FILE* input_file = fopen(input_file_name, "r");
		if (!input_file) {
			fprintf(stderr, "Can't open input file! \n\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			return 1;
		}
		fscanf(input_file, "%lf", &exponent);
		fclose(input_file);
	}
	if (rank == 0) {
		for (int i = 1; i < np; i++) {
			MPI_Send(&exponent, 1, MPI_DOUBLE, i, VALUE_TAG, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Recv(&exponent, 1, MPI_DOUBLE, 0, VALUE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	int last_term_number = 0;
	double sum = .0;
	for (int step = 0; step < 1000; step++) {
		int term_number;
		if (rank == 0) {
			term_number = last_term_number++;
			int current_term_number = last_term_number;
			for (int i = 1; i < np; i++) {
				MPI_Send(&current_term_number, 1, MPI_INT, i, TERM_NUMBER_TAG, MPI_COMM_WORLD);
				current_term_number++;
			}
			last_term_number = current_term_number;
		}
		else {
			MPI_Recv(&term_number, 1, MPI_INT, 0, TERM_NUMBER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		double term = calc_series_term(term_number, exponent);
		int need_break = false;
		if (rank == 0) {
			double current_term = term;
			sum += current_term;
			if (current_term < EPSILON) {
				need_break = true;
			}
			for (int i = 1; i < np; i++) {
				MPI_Recv(&current_term, 1, MPI_DOUBLE, i, TERM_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				sum += current_term;
				if (current_term < EPSILON) {
					need_break = true;
					break;
				}
			}
			for (int i = 1; i < np; i++) {
				MPI_Send(&need_break, 1, MPI_INT, i, BREAK_TAG, MPI_COMM_WORLD);
			}
		}
		else {
			MPI_Send(&term, 1, MPI_DOUBLE, 0, TERM_TAG, MPI_COMM_WORLD);
			MPI_Recv(&need_break, 1, MPI_INT, 0, BREAK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (need_break) {
			break;
		}
	}
	if (rank == 0) {
		FILE* output_file = fopen(output_file_name, "w");
		if (!output_file) {
			fprintf(stderr, "Can't open output file! \n\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
			return 2;
		}
		fprintf(output_file, "%.15lf\n", sum);
	}
	MPI_Finalize();
	return 0;
}