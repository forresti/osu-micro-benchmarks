#define BENCHMARK "OSU MPI%s Reduce_scatter Latency Test"
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
    int i, numprocs, rank, size;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer=0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    float *sendbuf, *recvbuf;
    int *recvcounts;
    int po_ret;
    size_t bufsize;

    set_header(HEADER);
    set_benchmark_name("osu_scatter");
    enable_accel_support();
    po_ret = process_options(argc, argv);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

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
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (options.max_message_size > options.max_mem_limit) {
        options.max_message_size = options.max_mem_limit;
    }

    options.min_message_size /= sizeof(float);
    if (options.min_message_size < DEFAULT_MIN_MESSAGE_SIZE) {
        options.min_message_size = DEFAULT_MIN_MESSAGE_SIZE;
    }

    if (allocate_buffer((void**)&recvcounts, numprocs*sizeof(int), none)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    bufsize = sizeof(float)*(options.max_message_size/sizeof(float));
    if (allocate_buffer((void**)&sendbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(sendbuf, options.accel, 1, bufsize);

    bufsize = sizeof(float)*((options.max_message_size/numprocs + 1)/sizeof(float));
    if (allocate_buffer((void**)&recvbuf, bufsize,
                options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(recvbuf, options.accel, 0, bufsize);

    print_preamble(rank);

    for(size=options.min_message_size; size*sizeof(float) <= options.max_message_size; size *= 2) {

        if(size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

        int portion=0, remainder=0;
        portion=size/numprocs;
        remainder=size%numprocs;

        for (i=0; i<numprocs; i++){
            recvcounts[i]=0;
            if(size<numprocs){
                if(i<size)
                    recvcounts[i]=1;
            }
            else{
                if((remainder!=0) && (i<remainder)){
                    recvcounts[i]+=1;
                }
                recvcounts[i]+=portion;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        timer=0.0;
        for(i=0; i < options.iterations + options.skip ; i++) {
            t_start = MPI_Wtime();

            MPI_Reduce_scatter( sendbuf, recvbuf, recvcounts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );
            t_stop=MPI_Wtime();
            if(i>=options.skip){

            timer+=t_stop-t_start;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        latency = (double)(timer * 1e6) / options.iterations;

        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
        avg_time = avg_time/numprocs;

        print_stats(rank, size * sizeof(float), avg_time, min_time, max_time);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free_buffer(recvcounts, none);
    free_buffer(sendbuf, options.accel);
    free_buffer(recvbuf, options.accel);

    MPI_Finalize();

    if (none != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}
