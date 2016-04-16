#define BENCHMARK "OSU UPC++ Scatter Latency Test"
/*
 * Copyright (C) 2002-2015 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <stdio.h>
#include <upcxx.h>
#include <stdlib.h>
#include <osu_common.h>
#include <osu_coll.h>

using namespace std;
using namespace upcxx;

#define root 0
#define VERIFY 0

int
main (int argc, char *argv[])
{
    init(&argc, &argv);

    global_ptr<char> src;
    global_ptr<char> dst;
    global_ptr<double> time_src;
    global_ptr<double> time_dst;

    double avg_time, max_time, min_time;
    int i = 0, size;
    int skip;
    int64_t t_start = 0, t_stop = 0, timer=0;
    int max_msg_size = 1<<20, full = 0;

    if (process_args(argc, argv, myrank(), &max_msg_size, &full, HEADER)) {
        return 0;
    }

    if (ranks() < 2) {
        if (myrank() == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }
        return -1;
    }

    src = allocate<char> (root, max_msg_size*sizeof(char)*ranks());
    dst = allocate<char> (myrank(), max_msg_size*sizeof(char));

    assert(src != NULL);
    assert(dst != NULL);

    time_src = allocate<double> (myrank(), 1); //for each node's local result
    time_dst = allocate<double> (root, 1); //for reduction result on root

    assert(time_src != NULL);
    assert(time_dst != NULL);

    /*
     * put a barrier since allocate is non-blocking in upc++
     */
    barrier();

    print_header(HEADER, myrank(), full);

    for (size=1; size <=max_msg_size; size *= 2) {
        if (size > LARGE_MESSAGE_SIZE) {
            skip = SKIP_LARGE;
            iterations = iterations_large;
        } else {
            skip = SKIP;
        }

        timer=0;
        for (i=0; i < iterations + skip ; i++) {
            t_start = getMicrosecondTimeStamp();
            upcxx_scatter((char *)src, (char *)dst, size*sizeof(char), root);
            t_stop = getMicrosecondTimeStamp();

            if (i>=skip) {
                timer+=t_stop-t_start;
            }
            barrier();
        }

        barrier();

        double* lsrc = (double *)time_src;
        lsrc[0] = (1.0 * timer) / iterations;

        upcxx_reduce<double>((double *)time_src, (double *)time_dst, 1, root,
                UPCXX_MAX, UPCXX_DOUBLE);
        if (myrank()==root) {
            double* ldst = (double *)time_dst;
            max_time = ldst[0];
        }

        upcxx_reduce<double>((double *)time_src, (double *)time_dst, 1, root,
                UPCXX_MIN, UPCXX_DOUBLE);
        if (myrank()==root) {
            double* ldst = (double *)time_dst;
            min_time = ldst[0];
        }

        upcxx_reduce<double>((double *)time_src, (double *)time_dst, 1, root,
                UPCXX_SUM, UPCXX_DOUBLE);
        if (myrank()==root) {
            double* ldst = (double *)time_dst;
            avg_time = ldst[0]/ranks();
        }

        barrier();

        print_data(myrank(), full, size*sizeof(char), avg_time, min_time,
                max_time, iterations);
    }

    deallocate(src);
    deallocate(dst);
    deallocate(time_src);
    deallocate(time_dst);

    finalize();

    return EXIT_SUCCESS;
}

/* vi: set sw=4 sts=4 tw=80: */
