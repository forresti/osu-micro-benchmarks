/*
 * Copyright (C) 2002-2015 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" double getMicrosecondTimeStamp (void);
#endif /* #ifdef __cplusplus */

double
getMicrosecondTimeStamp (void)
{
    double retval;
    struct timeval tv;

    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }

    retval = tv.tv_sec * (double)1e6 + tv.tv_usec;

    return retval;
}
