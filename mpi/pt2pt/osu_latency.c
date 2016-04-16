#define BENCHMARK "OSU MPI%s Latency Test"
/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <osu_pt2pt.h>

int
main (int argc, char *argv[])
{
    int myid, numprocs, i;
    int size;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0;
    int po_ret = process_options(argc, argv, LAT);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
           fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    set_header(HEADER);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (0 == myid) {
        switch (po_ret) {
            case po_cuda_not_avail:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case po_openacc_not_avail:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case po_bad_usage:
            case po_help_message:
                usage("osu_latency");
                break;
        }
    }

    switch (po_ret) {
        case po_cuda_not_avail:
        case po_openacc_not_avail:
        case po_bad_usage:
            MPI_Finalize();
            exit(EXIT_FAILURE);
        case po_help_message:
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        case po_okay:
            break;
    }

    if(numprocs != 2) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (allocate_memory(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    print_header(myid, LAT);

    
    /* Latency test */
    for(size = 0; size <= MAX_MSG_SIZE; size = (size ? size * 2 : 1)) {
        touch_data(s_buf, r_buf, myid, size);

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = options.loop_large;
            options.skip = options.skip_large;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(myid == 0) {
            for(i = 0; i < options.loop + options.skip; i++) {
                if(i == options.skip) t_start = MPI_Wtime();

                MPI_Send(s_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
                MPI_Recv(r_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &reqstat);
            }

            t_end = MPI_Wtime();
        }

        else if(myid == 1) {
            for(i = 0; i < options.loop + options.skip; i++) {
                MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &reqstat);
                MPI_Send(s_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
        }

        if(myid == 0) {
            double latency = (t_end - t_start) * 1e6 / (2.0 * options.loop);

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
        }
    }

    free_memory(s_buf, r_buf, myid);
    MPI_Finalize();

    if (none != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

