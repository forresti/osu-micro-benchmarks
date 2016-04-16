#define BENCHMARK "OSU MPI%s Non-blocking All-to-Allv Personalized Exchange Latency Test"
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
    setbuf(stdout, NULL);
    int i = 0, rank, size;
    int numprocs;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double tcomp = 0.0, tcomp_total=0.0, latency_in_secs=0.0;
    double test_time = 0.0, test_total = 0.0;
    double timer=0.0;
    double wait_time = 0.0, init_time = 0.0;
    double init_total = 0.0, wait_total = 0.0;

    char *sendbuf=NULL;
    char *recvbuf=NULL;
    int *rdispls=NULL, *recvcounts=NULL, *sdispls=NULL, *sendcounts=NULL;
    MPI_Datatype *stypes = NULL, *rtypes = NULL;
    int po_ret;
    size_t bufsize;
    int disp = 0;
    set_header(HEADER);
    set_benchmark_name("osu_ialltoallw");
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
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (options.max_message_size * numprocs > options.max_mem_limit) {
        options.max_message_size = options.max_mem_limit / numprocs;
    }
     
    if (allocate_buffer((void**)&recvcounts, numprocs*sizeof(int), none)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (allocate_buffer((void**)&sendcounts, numprocs*sizeof(int), none)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (allocate_buffer((void**)&rdispls, numprocs*sizeof(int), none)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (allocate_buffer((void**)&sdispls, numprocs*sizeof(int), none)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (allocate_buffer((void**)&stypes, numprocs*sizeof(MPI_Datatype), none)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (allocate_buffer((void**)&rtypes, numprocs*sizeof(MPI_Datatype), none)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    bufsize = options.max_message_size * numprocs;
    if (allocate_buffer((void**)&sendbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(sendbuf, options.accel, 1, bufsize);

    if (allocate_buffer((void**)&recvbuf, bufsize,
                options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(recvbuf, options.accel, 0, bufsize);

    print_preamble_nbc(rank);

    for(size=options.min_message_size; size <=options.max_message_size; size *= 2) {
        if(size > LARGE_MESSAGE_SIZE) {
            options.skip = SKIP_LARGE;
            options.iterations = options.iterations_large;
        }
        else {
            options.skip = SKIP;
        }
        
        disp =0;
        for ( i = 0; i < numprocs; i++) {
            recvcounts[i] = size;
            sendcounts[i] = size;
            rdispls[i] = disp;
            sdispls[i] = disp;
            disp += size;
            stypes[i] = MPI_CHAR;
            rtypes[i] = MPI_CHAR;
        }
        MPI_Barrier(MPI_COMM_WORLD);
 
        timer = 0.0;     
          
        for(i=0; i < options.iterations + options.skip ; i++) {
            t_start = MPI_Wtime();
            MPI_Ialltoallw(sendbuf, sendcounts, sdispls, stypes, 
                          recvbuf, recvcounts, rdispls, rtypes, 
                          MPI_COMM_WORLD, &request);
            MPI_Wait(&request,&status);
            
            t_stop = MPI_Wtime();
            
            if(i>=options.skip){
                timer += t_stop-t_start;
            } 
            MPI_Barrier(MPI_COMM_WORLD);
        }  
        
        MPI_Barrier(MPI_COMM_WORLD);

        latency = (timer * 1e6) / options.iterations;
        
        latency_in_secs = timer/options.iterations; 

        init_arrays(latency_in_secs);

        disp =0;
        for ( i = 0; i < numprocs; i++) {
            recvcounts[i] = size;
            sendcounts[i] = size;
            rdispls[i] = disp;
            sdispls[i] = disp;
            disp += size;

        }

        MPI_Barrier(MPI_COMM_WORLD);

        timer = 0.0; tcomp_total = 0; tcomp = 0;        
        init_total = 0.0; wait_total = 0.0;
        test_time = 0.0, test_total = 0.0;
 
        for(i=0; i < options.iterations + options.skip ; i++) {
            t_start = MPI_Wtime();

            init_time = MPI_Wtime();
            MPI_Ialltoallw(sendbuf, sendcounts, sdispls, stypes,
                          recvbuf, recvcounts, rdispls, rtypes, 
                          MPI_COMM_WORLD, &request);
            init_time = MPI_Wtime() - init_time;

            tcomp = MPI_Wtime();             
            test_time = dummy_compute(latency_in_secs, &request); 
            tcomp = MPI_Wtime() - tcomp;

            wait_time = MPI_Wtime();
            MPI_Wait(&request,&status);
            wait_time = MPI_Wtime() - wait_time;

            t_stop = MPI_Wtime();
            
            if(i>=options.skip){
                test_total += test_time;
                timer += t_stop-t_start;
                tcomp_total += tcomp;
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
    }  

    free_buffer(rdispls, none);
    free_buffer(sdispls, none);
    free_buffer(recvcounts, none);
    free_buffer(sendcounts, none);
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

/* vi: set sw=4 sts=4 tw=80: */
