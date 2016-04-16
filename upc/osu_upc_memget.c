#define BENCHMARK "OSU UPC MEMGET Test"
/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <upc.h>
#include <stdio.h>
#include <string.h>

#define MAX_MSG_SIZE         (1<<22)
#define SKIP_LARGE  10
#define LOOP_LARGE  100
#define LARGE_MESSAGE_SIZE  8192

int skip = 1000;
int loop = 10000;

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

void wtime(double *t)
{
  static int sec = -1;
  struct timeval tv;
  gettimeofday(&tv, (void *)0);
  if (sec < 0) sec = tv.tv_sec;
  *t = (tv.tv_sec - sec)*1.0e+6 + tv.tv_usec;
}

int main(int argc, char **argv) 
{
    int iters=0;
    double t_start, t_end;
    int peerid = (MYTHREAD+1)%THREADS; 
    int iamsender = 0;
    int i;

    if( THREADS == 1 ) {
        if(MYTHREAD == 0) {
            fprintf(stderr, "This test requires at least two UPC threads\n");
        }
        return 0;
    }

    if ( MYTHREAD < THREADS/2 )
        iamsender = 1;

    shared char *data = upc_all_alloc(THREADS, MAX_MSG_SIZE*2);
    shared [] char *remote = (shared [] char *)(data + peerid);
    char *local = ((char *)(data+MYTHREAD)) + MAX_MSG_SIZE;

    if ( !MYTHREAD ) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "# [ pairs: %d ]\n", THREADS/2);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);
    }

    for (int size = 1; size <= MAX_MSG_SIZE; size*=2) {

        if ( iamsender )
            for(i = 0; i < size; i++) {
                local[i] = 'a';
            }
        else
            for(i = 0; i < size; i++) {
                local[i] = 'b';
            }

        upc_barrier;

        if(size > LARGE_MESSAGE_SIZE) {
            loop = LOOP_LARGE;
            skip = SKIP_LARGE;
        }

        if( iamsender )
        {
            for ( i = 0; i < loop + skip; i++) {
                if(i == skip) {
                    upc_barrier;
                    wtime(&t_start);
                }

                upc_memget(local, remote, size);
            }

            upc_barrier;

            wtime(&t_end);
            if( !MYTHREAD )
            {
                double latency = (t_end - t_start)/(1.0 * loop);
                fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                        FLOAT_PRECISION, latency);
                fflush(stdout);
            }
        } else 
        {
            upc_barrier;
            upc_barrier;
        }

    }
    return 0;
}
