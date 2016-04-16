#define BENCHMARK "OSU MPI%s Non-blocking Barrier Latency Test"
/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include "osu_coll.h"

int main(int argc, char *argv[])
{
    int i = 0, rank, size = 0;
    int numprocs;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double test_time = 0.0, test_total = 0.0;
    double tcomp = 0.0, tcomp_total=0.0, latency_in_secs=0.0;
    double wait_time = 0.0, init_time = 0.0;
    double init_total = 0.0, wait_total = 0.0;
    double timer = 0.0;
    int po_ret;

    set_header(HEADER);
    set_benchmark_name("osu_ibarrier");
    enable_accel_support();
    po_ret = process_options(argc, argv);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    options.show_size = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Request request;
    MPI_Status status;

    switch (po_ret) {
        case po_bad_usage:
            print_bad_usage_message(rank);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        case po_help_message:
            print_help_message(rank);
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        case po_version_message:
            print_version_message(rank);
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        case po_okay:
            break;
    }

    if(numprocs < 2) {
        if(rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_Finalize();

        return EXIT_FAILURE;
    }

    print_preamble_nbc(rank);

    options.skip = options.skip_large;
    options.iterations = iterations_large;
    timer = 0.0;

    for(i=0; i < options.iterations + options.skip ; i++) {
        t_start = MPI_Wtime();
        MPI_Ibarrier(MPI_COMM_WORLD, &request);
        MPI_Wait(&request,&status);
        t_stop = MPI_Wtime();

        if(i>=options.skip){
            timer+=t_stop-t_start;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    latency = (timer * 1e6) / options.iterations;

    /* Comm. latency in seconds, fed to dummy_compute */
    latency_in_secs = timer/options.iterations;

    init_arrays(latency_in_secs);

    MPI_Barrier(MPI_COMM_WORLD);

    timer = 0.0; tcomp_total = 0; tcomp = 0;
    init_total = 0.0; wait_total = 0.0;
    test_time = 0.0, test_total = 0.0;

    for(i=0; i < options.iterations + options.skip ; i++) {
            t_start = MPI_Wtime();

            init_time = MPI_Wtime();
            MPI_Ibarrier(MPI_COMM_WORLD, &request);
            init_time = MPI_Wtime() - init_time;

            tcomp = MPI_Wtime();
            test_time = dummy_compute(latency_in_secs, &request);
            tcomp = MPI_Wtime() - tcomp;

            wait_time = MPI_Wtime();
            MPI_Wait(&request,&status);
            wait_time = MPI_Wtime() - wait_time;

            t_stop = MPI_Wtime();

            if(i>=options.skip){
                timer += t_stop-t_start;
                tcomp_total += tcomp;
                test_total += test_time;
                init_total += init_time;
                wait_total += wait_time;
            }
            MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier (MPI_COMM_WORLD);

    calculate_and_print_stats(rank, size, numprocs,
                                  timer, latency,
                                  test_total, tcomp_total,
                                  wait_total, init_total);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

/* vi: set sw=4 sts=4 tw=80: */
