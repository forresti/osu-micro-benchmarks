#define BENCHMARK "OSU MPI Multi Latency Test"
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

#define MAX_MSG_SIZE (1<<22)
#define MAX_STEPS    (22+1)

char *s_buf, *r_buf;

static void multi_latency(int rank, int pairs);

int main(int argc, char* argv[])
{
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    int rank, nprocs; 
    int pairs;

    int po_ret = process_options(argc, argv, LAT);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }
    set_header(HEADER);
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    pairs = nprocs/2;

    if (0 == rank) {
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
                usage("osu_multi_lat");
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

    if (posix_memalign((void**)&s_buf, align_size, MAX_MSG_SIZE)) {
        fprintf(stderr, "Error allocating host memory\n");
        return EXIT_FAILURE;
    }

    if (posix_memalign((void**)&r_buf, align_size, MAX_MSG_SIZE)) {
        fprintf(stderr, "Error allocating host memory\n");
        return EXIT_FAILURE;
    }

    memset(s_buf, 0, MAX_MSG_SIZE);
    memset(r_buf, 0, MAX_MSG_SIZE);

    if(rank == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    multi_latency(rank, pairs);
    
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    free(r_buf);
    free(s_buf);

    return EXIT_SUCCESS;
}

static void multi_latency(int rank, int pairs)
{
    int size, partner;
    int i;
    double t_start = 0.0, t_end = 0.0,
           latency = 0.0, total_lat = 0.0,
           avg_lat = 0.0;

    MPI_Status reqstat;


    for(size = 0; size <= MAX_MSG_SIZE; size  = (size ? size * 2 : 1)) {

        MPI_Barrier(MPI_COMM_WORLD);

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = options.loop_large;
            options.skip = options.skip_large;
        } else {
            options.loop = options.loop; 
            options.skip = options.skip;
        }

        if (rank < pairs) {
            partner = rank + pairs;

            for (i = 0; i < options.loop + options.skip; i++) {

                if (i == options.skip) {
                    t_start = MPI_Wtime();
                    MPI_Barrier(MPI_COMM_WORLD);
                }

                MPI_Send(s_buf, size, MPI_CHAR, partner, 1, MPI_COMM_WORLD);
                MPI_Recv(r_buf, size, MPI_CHAR, partner, 1, MPI_COMM_WORLD,
                         &reqstat);
            }

            t_end = MPI_Wtime();

        } else {
            partner = rank - pairs;

            for (i = 0; i < options.loop + options.skip; i++) {

                if (i == options.skip) {
                    t_start = MPI_Wtime();
                    MPI_Barrier(MPI_COMM_WORLD);
                }

                MPI_Recv(r_buf, size, MPI_CHAR, partner, 1, MPI_COMM_WORLD,
                         &reqstat);
                MPI_Send(s_buf, size, MPI_CHAR, partner, 1, MPI_COMM_WORLD);
            }

            t_end = MPI_Wtime();
        }

        latency = (t_end - t_start) * 1.0e6 / (2.0 * options.loop);

        MPI_Reduce(&latency, &total_lat, 1, MPI_DOUBLE, MPI_SUM, 0, 
                   MPI_COMM_WORLD);

        avg_lat = total_lat/(double) (pairs * 2);

        if(0 == rank) {
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, avg_lat);
            fflush(stdout);
        }
    }
}

/* vi: set sw=4 sts=4 tw=80: */
