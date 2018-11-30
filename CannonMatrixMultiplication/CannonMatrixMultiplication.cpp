// CannonMatrixMultiplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<math.h>

int allocMatrix(int*** mat, int rows, int cols) {
	// Allocate rows*cols contiguous items
	int* p = (int*)malloc(sizeof(int*) * rows * cols);
	if (!p) {
		return -1;
	}
	// Allocate row pointers
	*mat = (int**)malloc(rows * sizeof(int*));
	if (!mat) {
		free(p);
		return -1;
	}

	// Set up the pointers into the contiguous memory
	for (int i = 0; i < rows; i++) {
		(*mat)[i] = &(p[i * cols]);
	}
	return 0;
}

int freeMatrix(int ***mat) {
	free(&((*mat)[0][0]));
	free(*mat);
	return 0;
}

void matrixMultiply(int **a, int **b, int rows, int cols, int ***c) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int val = 0;
			for (int k = 0; k < rows; k++) {
				val += a[i][k] * b[k][j];
 			}
			(*c)[i][j] = val;
		}
	}
}

int main(int argc, char* argv[]) {
	MPI_Comm cartComm;
	int dim[2], period[2], reorder;
	int coord[2], id;
	FILE *fp;
	int **A = NULL, **B = NULL, **C = NULL;
	int **localA = NULL, **localB = NULL, **localC = NULL;
	int rows = 0;
	int columns;
	int count = 0;
	int worldSize;
	int procDim;
	int blockDim;
	int left, right, up, down;

	// Initialize the MPI environment
	MPI_Init(&argc, &argv);

	// World size
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	
	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		printf("World size is: %d\n", worldSize);

		int n;
		char ch;

		// Determine matrix dimensions
		fp = fopen("A.txt", "r");
		if (fp == NULL) {
			return 1;
		}
		while (fscanf(fp, "%d", &n) != EOF) {
			ch = fgetc(fp);
			if (ch == '\n') {
				rows = rows + 1;
			}
			count++;
		}
		columns = count / rows;

		// Check matrix and world size
		if (columns != rows) {
			printf("[ERROR] Matrix must be square!\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		double sqroot = sqrt(worldSize);
		if ((sqroot - floor(sqroot)) != 0) {
			printf("[ERROR] Number of processes must be a perfect square!\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		int intRoot = (int)sqroot;
		if (columns%intRoot != 0 || rows%intRoot != 0) {
			printf("[ERROR] Number of rows/columns not divisible by %d!\n", intRoot);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		procDim = intRoot;
		blockDim = columns / intRoot;

		fseek(fp, 0, SEEK_SET);

		if (allocMatrix(&A, rows, columns) != 0) {
			printf("[ERROR] Matrix alloc for A failed!\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		if (allocMatrix(&B, rows, columns) != 0) {
			printf("[ERROR] Matrix alloc for B failed!\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		// Read matrix A
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				fscanf(fp, "%d", &n);
				A[i][j] = n;
			}
		}
		printf("A matrix:\n");
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				printf("%d\t", A[i][j]);
			}
			printf("\n");
		}
		fclose(fp);

		// Read matrix B
		fp = fopen("B.txt", "r");
		if (fp == NULL) {
			return 1;
		}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				fscanf(fp, "%d", &n);
				B[i][j] = n;
			}
		}
		printf("B matrix:\n");
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				printf("%d\t", B[i][j]);
			}
			printf("\n");
		}
		fclose(fp);

		if (allocMatrix(&C, rows, columns) != 0) {
			printf("[ERROR] Matrix alloc for C failed!\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	// Create 2D Cartesian grid of processes
	MPI_Bcast(&procDim, 1,MPI_INT,0,MPI_COMM_WORLD);
	dim[0] = procDim; dim[1] = procDim;
	period[0] = 1; period[1] = 1;
	reorder = 1;
	//printf("before cart create: dim[0]:%d dim[1]:%d\n", dim[0], dim[1]);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cartComm);

	// Allocate local blocks for A and B
	MPI_Bcast(&blockDim, 1, MPI_INT, 0, MPI_COMM_WORLD);
	allocMatrix(&localA, blockDim, blockDim);
	allocMatrix(&localB, blockDim, blockDim);

	// Create datatype to describe the subarrays of the global array
	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int globalSize[2] = { rows, columns };
	int localSize[2] = { blockDim, blockDim };
	int starts[2] = { 0,0 };
	MPI_Datatype type, subarrtype;
	MPI_Type_create_subarray(2, globalSize, localSize, starts, MPI_ORDER_C, MPI_INT, &type);
	MPI_Type_create_resized(type, 0, blockDim * sizeof(int), &subarrtype);
	MPI_Type_commit(&subarrtype);

	int *globalptrA = NULL;
	if (rank == 0) {
		globalptrA = &(A[0][0]);
	}
	int *globalptrB = NULL;
	if (rank == 0) {
		globalptrB = &(B[0][0]);
	}
	int *globalptrC = NULL;
	if (rank == 0) {
		globalptrC = &(C[0][0]);
	}

	// Scatter the array to all processors
	int* sendCounts = (int*)malloc(sizeof(int) * worldSize);
	int* displacements = (int*)malloc(sizeof(int) * worldSize);

	if (rank == 0) {
		for (int i = 0; i < worldSize; i++) {
			sendCounts[i] = 1;
		}
		int disp = 0;
		for (int i = 0; i < procDim; i++) {
			for (int j = 0; j < procDim; j++) {
				displacements[i * procDim + j] = disp;
				disp += 1;
			}
			disp += (blockDim - 1)* procDim;
		}
	}

	MPI_Scatterv(globalptrA, sendCounts, displacements, subarrtype, &(localA[0][0]),
		rows * columns / (worldSize), MPI_INT,
		0, MPI_COMM_WORLD);
	MPI_Scatterv(globalptrB, sendCounts, displacements, subarrtype, &(localB[0][0]),
		rows * columns / (worldSize), MPI_INT,
		0, MPI_COMM_WORLD);

	// Print block for each rank
	/*for (int p = 0; p< worldSize; p++) {
		if (rank == p) {
			printf("Local block A on rank %d is:\n", rank);
			for (int i = 0; i < blockDim; i++) {
				putchar('|');
				for (int j = 0; j < blockDim; j++) {
					printf("%d ", localA[i][j]);
				}
				printf("|\n");
			}

			printf("Local block B on rank %d is:\n", rank);
			for (int i = 0; i < blockDim; i++) {
				putchar('|');
				for (int j = 0; j < blockDim; j++) {
					printf("%d ", localB[i][j]);
				}
				printf("|\n");
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}*/

	if (allocMatrix(&localC, blockDim, blockDim) != 0) {
		printf("[ERROR] Matrix alloc for localC in rank %d failed!\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// Initial skew
	MPI_Cart_coords(cartComm, rank, 2, coord);
	printf("Rank %d coordinates are %d %d\n", rank, coord[0], coord[1]);
	MPI_Cart_shift(cartComm, 1, coord[0], &left, &right);
	//printf("For rank %d: left is %d right is %d\n", rank, left, right);
	//printf("Rank %d sending to %d and receiving from %d\n", rank, left, right);
	MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, subarrtype, left, 1, right, 1, cartComm, MPI_STATUS_IGNORE);

	MPI_Cart_shift(cartComm, 0, coord[1], &up, &down);
	printf("For rank %d: up is %d down is %d\n", rank, up, down);
	printf("Rank %d sending to %d and receiving from %d\n", rank, up, down);
	MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, subarrtype, up, 1, down, 1, cartComm, MPI_STATUS_IGNORE);

	//Multiply and shift -> repeat blockDim - 1 times
	// Init C
	for (int i = 0; i < blockDim; i++) {
		for (int j = 0; j < blockDim; j++) {
			localC[i][j] = 0;
		}
	}

	int** multiplyRes = NULL;
	if (allocMatrix(&multiplyRes, blockDim, blockDim) != 0) {
		printf("[ERROR] Matrix alloc for multiplyRes in rank %d failed!\n", rank);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	for (int k = 0; k < blockDim; k++) {
		matrixMultiply(localA, localB, blockDim, blockDim, &multiplyRes);
		for (int i = 0; i < blockDim; i++) {
			for (int j = 0; j < blockDim; j++) {
				localC[i][j] += multiplyRes[i][j];
			}
		}
		// Shift once (left and up)
		MPI_Cart_shift(cartComm, 1, 1, &left, &right);
		MPI_Cart_shift(cartComm, 0, 1, &up, &down);
		MPI_Sendrecv_replace(&(localA[0][0]), blockDim * blockDim, subarrtype, left, 1, right, 1, cartComm, MPI_STATUS_IGNORE);
		MPI_Sendrecv_replace(&(localB[0][0]), blockDim * blockDim, subarrtype, up, 1, down, 1, cartComm, MPI_STATUS_IGNORE);
	}
	
	//matrixMultiply(localA, localB, blockDim, blockDim, &localC);
	/*for (int i = 0; i < blockDim; i++) {
		for (int j = 0; j < blockDim; j++) {
			localC[i][j] = localB[i][j];
		}
	}*/

	// Gather results
	MPI_Gatherv(&(localC[0][0]), rows * columns / worldSize, MPI_INT,
		globalptrC, sendCounts, displacements, subarrtype,
		0, MPI_COMM_WORLD);

	freeMatrix(&localC);

	if (rank == 0) {
		printf("C is:\n");
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				printf("%d ", C[i][j]);
			}
			printf("\n");
		}
	}

	//if (rank == 5)
	//{
	//	MPI_Cart_coords(comm, rank, 2, coord);
	//	printf("Rank %d coordinates are %d %d\n", rank, coord[0], coord[1]);
	//	//fflush(stdout);
	//}
	/*if (rank == 0)
	{
	for (int i = 0; i < dim[0]; i++) {
	for (int j = 0; j < dim[1]; j++) {
	coord[0] = i; coord[1] = j;
	MPI_Cart_rank(cartComm, coord, &id);
	printf("The processor at position (%d, %d) has rank %d\n", coord[0], coord[1], id);
	}
	}
	}*/


	// Finalize the MPI environment
	MPI_Finalize();

	return 0;
}

