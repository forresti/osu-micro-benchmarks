/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int
main (int argc, char *argv[])
{
    int myid, numprocs;
    struct timespec tp_before, tp_after;
    long duration = 0, min, max, avg;

    clock_gettime(CLOCK_REALTIME, &tp_before);
    MPI_Init(&argc, &argv);
    clock_gettime(CLOCK_REALTIME, &tp_after);

    duration = (tp_after.tv_sec - tp_before.tv_sec) * 1e3;
    duration += (tp_after.tv_nsec - tp_before.tv_nsec) / 1e6;

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    MPI_Reduce(&duration, &min, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&duration, &max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&duration, &avg, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    avg = avg/numprocs;

    if(myid == 0) {
        printf("nprocs: %d, min: %ld, max: %ld, avg: %ld\n", numprocs, min, max, avg);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

