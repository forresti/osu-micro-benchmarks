#define BENCHMARK "OSU OpenSHMEM Put Test"
/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <shmem.h>
#include "osu_common.h"

#define MESSAGE_ALIGNMENT 64
#define MAX_MSG_SIZE (1<<20)
#define MYBUFSIZE (MAX_MSG_SIZE + MESSAGE_ALIGNMENT)

char s_buf_original[MYBUFSIZE];
char r_buf_original[MYBUFSIZE];

int skip = 1000;
int loop = 10000;
int skip_large = 10;
int loop_large = 100;
int large_message_size = 8192;

#ifdef PACKAGE_VERSION
#   define HEADER "# " BENCHMARK " v" PACKAGE_VERSION "\n"
#else
#   define HEADER "# " BENCHMARK "\n"
#endif

#ifndef FIELD_WIDTH
#   define FIELD_WIDTH 20
#endif

#ifndef FLOAT_PRECISION
#   define FLOAT_PRECISION 2
#endif

static void usage(int myid)
{
    if(myid == 0) {
        fprintf(stderr, "Invalid arguments. Usage: <prog_name> <heap|global>\n");
    }
}

int main(int argc, char *argv[])
{
    int myid, numprocs, i;
    int size;
    char *s_buf, *r_buf;
    char *s_buf_heap, *r_buf_heap;
    int align_size;
    int64_t t_start = 0, t_end = 0;
    int use_heap = 0;   //default uses global

    start_pes(0);
    myid = _my_pe();
    numprocs = _num_pes();

    if(numprocs != 2) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        return EXIT_FAILURE;
    }

    if(argc != 2) {
        usage(myid);

        return EXIT_FAILURE;
    }

    if(0 == strncmp(argv[1], "heap", strlen("heap"))){
        use_heap = 1;
    } else if(0 == strncmp(argv[1], "global", strlen("global"))){
        use_heap = 0;
    } else {
        usage(myid);
        return EXIT_FAILURE;
    }

    align_size = MESSAGE_ALIGNMENT;

    /**************Allocating Memory*********************/

    if(use_heap){

        s_buf_heap = shmalloc(MYBUFSIZE);
        r_buf_heap = shmalloc(MYBUFSIZE);

        s_buf =
            (char *) (((unsigned long) s_buf_heap + (align_size - 1)) /
                      align_size * align_size);

        r_buf =
            (char *) (((unsigned long) r_buf_heap + (align_size - 1)) /
                      align_size * align_size);
    } else {

        s_buf =
            (char *) (((unsigned long) s_buf_original + (align_size - 1)) /
                      align_size * align_size);

        r_buf =
            (char *) (((unsigned long) r_buf_original + (align_size - 1)) /
                      align_size * align_size);
    }

    /**************Memory Allocation Done*********************/

    if(myid == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);
    }

    for(size = 1; size <= MAX_MSG_SIZE; size = (size ? size * 2 : 1)) {
        
        /* touch the data */
        for(i = 0; i < size; i++) {
            s_buf[i] = 'a';
            r_buf[i] = 'b';
        }

        if(size > large_message_size) {
            loop = loop_large = 100;
            skip = skip_large = 0;
        }

        shmem_barrier_all();

        if(myid == 0) 
            {
                for(i = 0; i < loop + skip; i++) {
                    if(i == skip) t_start = TIME();

                    shmem_putmem(r_buf, s_buf, size, 1);
                    shmem_quiet();
                }

                t_end = TIME();
            }
        shmem_barrier_all();

        if(myid == 0) {
            double latency = (1.0 * (t_end-t_start)) / loop;

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
        }
    }

    shmem_barrier_all();

    if(use_heap){
        shfree(s_buf_heap);
        shfree(r_buf_heap);
    }

    shmem_barrier_all();
    return EXIT_SUCCESS;
}

/* vi: set sw=4 sts=4 tw=80: */
