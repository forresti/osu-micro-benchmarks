#define BENCHMARK "OSU MPI%s Bi-Directional Bandwidth Test"
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

int main(int argc, char *argv[])
{
    int myid, numprocs, i, j;
    int size;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0, t = 0.0;
    int window_size = 64;
    int po_ret = process_options(argc, argv, BW);

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
                usage("osu_bibw");
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

    print_header(myid, BW);

    /* Bi-Directional Bandwidth test */
    for(size = 1; size <= MAX_MSG_SIZE; size *= 2) {
        /* touch the data */
        touch_data(s_buf, r_buf, myid, size);

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = options.loop_large;
            options.skip = options.skip_large;
            window_size = WINDOW_SIZE_LARGE;
        }

        if(myid == 0) {
            for(i = 0; i < options.loop + options.skip; i++) {
                if(i == options.skip) {
                    t_start = MPI_Wtime();
                }

                for(j = 0; j < window_size; j++) {
                    MPI_Irecv(r_buf, size, MPI_CHAR, 1, 10, MPI_COMM_WORLD,
                            recv_request + j);
                }

                for(j = 0; j < window_size; j++) {
                    MPI_Isend(s_buf, size, MPI_CHAR, 1, 100, MPI_COMM_WORLD,
                            send_request + j);
                }

                MPI_Waitall(window_size, send_request, reqstat);
                MPI_Waitall(window_size, recv_request, reqstat);
            }

            t_end = MPI_Wtime();
            t = t_end - t_start;

        }

        else if(myid == 1) {
            for(i = 0; i < options.loop + options.skip; i++) {
                for(j = 0; j < window_size; j++) {
                    MPI_Irecv(r_buf, size, MPI_CHAR, 0, 100, MPI_COMM_WORLD,
                            recv_request + j);
                }

                for (j = 0; j < window_size; j++) {
                    MPI_Isend(s_buf, size, MPI_CHAR, 0, 10, MPI_COMM_WORLD,
                            send_request + j);
                }

                MPI_Waitall(window_size, send_request, reqstat);
                MPI_Waitall(window_size, recv_request, reqstat);
            }
        }

        if(myid == 0) {
            double tmp = size / 1e6 * options.loop * window_size * 2;

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, tmp / t);
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


