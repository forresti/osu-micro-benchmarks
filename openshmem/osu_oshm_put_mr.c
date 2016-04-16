#define BENCHMARK "OSU OpenSHMEM Put Message Rate Test"
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
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <shmem.h>
#include "osu_common.h"

#define ITERS_SMALL     (500)          
#define ITERS_LARGE     (50)
#define LARGE_THRESHOLD (8192)
#define MAX_MSG_SZ (1<<22)

#define MESSAGE_ALIGNMENT (1<<12)
#define MYBUFSIZE (MAX_MSG_SZ * ITERS_LARGE + MESSAGE_ALIGNMENT)

char global_msg_buffer[MYBUFSIZE];

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

#ifndef MEMORY_SELECTION
#   define MEMORY_SELECTION 1
#endif

struct pe_vars {
    int me;
    int npes;
    int pairs;
    int nxtpe;
};

struct pe_vars
init_openshmem (void)
{
    struct pe_vars v;

    start_pes(0);
    v.me = _my_pe();
    v.npes = _num_pes();
    v.pairs = v.npes / 2;
    v.nxtpe = v.me < v.pairs ? v.me + v.pairs : v.me - v.pairs;

    return v;
}

static void
print_usage (int myid)
{
    if (myid == 0) {
        if (MEMORY_SELECTION) {
            fprintf(stderr, "Usage: osu_oshm_put_mr <heap|global>\n");
        }

        else {
            fprintf(stderr, "Usage: osu_oshm_put_mr\n");
        }
    }
}

void
check_usage (int me, int npes, int argc, char * argv [])
{
    if (MEMORY_SELECTION) {
        if (2 == argc) {
            /*
             * Compare more than 4 and 6 characters respectively to make sure
             * that we're not simply matching a prefix but the entire string.
             */
            if (strncmp(argv[1], "heap", 10)
                && strncmp(argv[1], "global", 10)) {
                print_usage(me);
                exit(EXIT_FAILURE);
            }
        }

        else {
            print_usage(me);
            exit(EXIT_FAILURE);
        }
    }

    if (2 > npes) {
        if (0 == me) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        exit(EXIT_FAILURE);
    }
}

void
print_header (int myid)
{
    if(myid == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Messages/s");
        fflush(stdout);
    }
}

char *
allocate_memory (int me, long align_size, int use_heap)
{
    char * msg_buffer;

    if (!use_heap) {
        return global_msg_buffer;
    }

    msg_buffer = (char *)shmalloc(MAX_MSG_SZ * ITERS_LARGE + align_size);

    if (NULL == msg_buffer) {
        fprintf(stderr, "Failed to shmalloc (pe: %d)\n", me);
        exit(EXIT_FAILURE);
    }

    return msg_buffer;
}

char *
align_memory (unsigned long address, int const align_size)
{
    return (char *) ((address + (align_size - 1)) / align_size * align_size);
}

double
message_rate (struct pe_vars v, char * buffer, int size, int iterations)
{
    int64_t begin, end; 
    int i, offset;

    /*
     * Touch memory
     */
    memset(buffer, size, MAX_MSG_SZ * ITERS_LARGE);

    shmem_barrier_all();

    if (v.me < v.pairs) {
        begin = TIME();

        for (i = 0, offset = 0; i < iterations; i++, offset += size) {
            shmem_putmem(&buffer[offset], &buffer[offset], size, v.nxtpe);
        }

        shmem_quiet();
        end = TIME();

        return ((double)iterations * 1e6) / ((double)end - (double)begin);
    }

    return 0;
}

void
print_message_rate (int myid, int size, double rate)
{
    if (myid == 0) { 
        fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION,
                rate);
        fflush(stdout);
    }
}

void
benchmark (struct pe_vars v, char * msg_buffer)
{
    static double pwrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
    static long psync[_SHMEM_REDUCE_SYNC_SIZE];
    static double mr, mr_sum;
    unsigned long size, i;

    memset(psync, _SHMEM_SYNC_VALUE, sizeof(long[_SHMEM_REDUCE_SYNC_SIZE]));

    /*
     * Warmup
     */
    if (v.me < v.pairs) {
        for (i = 0; i < (ITERS_LARGE * MAX_MSG_SZ); i += MAX_MSG_SZ) {
            shmem_putmem(&msg_buffer[i], &msg_buffer[i], MAX_MSG_SZ, v.nxtpe);
        }
    }
    
    shmem_barrier_all();

    /*
     * Benchmark
     */
    for (size = 1; size <= MAX_MSG_SZ; size <<= 1) {
        i = size < LARGE_THRESHOLD ? ITERS_SMALL : ITERS_LARGE;

        mr = message_rate(v, msg_buffer, size, i);
        shmem_double_sum_to_all(&mr_sum, &mr, 1, 0, 0, v.npes, pwrk, psync);
        print_message_rate(v.me, size, mr_sum);
    }
}

int
main (int argc, char *argv[])
{
    struct pe_vars v;
    char * msg_buffer, * aligned_buffer;
    long alignment;
    int use_heap;

    /*
     * Initialize
     */
    v = init_openshmem();
    check_usage(v.me, v.npes, argc, argv);
    print_header(v.me);

    /*
     * Allocate Memory
     */
    use_heap = !strncmp(argv[1], "heap", 10);
    alignment = use_heap ? sysconf(_SC_PAGESIZE) : 4096;
    msg_buffer = allocate_memory(v.me, alignment, use_heap);
    aligned_buffer = align_memory((unsigned long)msg_buffer, alignment);
    memset(aligned_buffer, 0, MAX_MSG_SZ * ITERS_LARGE);

    /*
     * Time Put Message Rate
     */
    benchmark(v, aligned_buffer);

    /*
     * Finalize
     */
    if (use_heap) {
        shfree(msg_buffer);
    }
    
    return EXIT_SUCCESS;
}
