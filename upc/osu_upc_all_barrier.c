#define BENCHMARK "OSU UPC Barrier Latency Test"
/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>
#include "osu_coll.h"
#include "osu_common.h"
#include <stdlib.h>

#include <upc.h>
#include <upc_collective.h>

#ifdef PACKAGE_VERSION
#   define HEADER "# " BENCHMARK " v" PACKAGE_VERSION "\n"
#else
#   define HEADER "# " BENCHMARK "\n"
#endif

#define SYNC_MODE (UPC_IN_ALLSYNC | UPC_OUT_ALLSYNC)

shared double avg_time, max_time, min_time;
shared double latency[THREADS];

int main(int argc, char *argv[])
{
    int i = 0;
    int skip;
    int64_t t_start = 0, t_stop = 0, timer=0;
    int full = 0;

    if (process_args(argc, argv, MYTHREAD, NULL, &full, HEADER)) {
        return 0;
    }
    
    if(THREADS < 2) {
        if(MYTHREAD == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }
        return -1;
    }
    print_header(HEADER, MYTHREAD, full);
    upc_barrier;

    skip = SKIP;    
    timer=0;        
    for(i=0; i < iterations + skip ; i++) {        
        t_start = TIME();
        upc_barrier;
        t_stop = TIME();

        if(i>=skip){
            timer+=t_stop-t_start;
        } 
    }
    upc_barrier;
    latency[MYTHREAD] = (1.0 * timer) / iterations;

    upc_all_reduceD(&min_time, latency, UPC_MIN, THREADS, 1, NULL, SYNC_MODE);
    upc_all_reduceD(&max_time, latency, UPC_MAX, THREADS, 1, NULL, SYNC_MODE);
    upc_all_reduceD(&avg_time, latency, UPC_ADD, THREADS, 1, NULL, SYNC_MODE);
    if(!MYTHREAD)
        avg_time = avg_time/THREADS;
    
    print_data(MYTHREAD, full, 0, avg_time, min_time, max_time, iterations);
    upc_barrier;
    return EXIT_SUCCESS;
}

/* vi: set sw=4 sts=4 tw=80: */
