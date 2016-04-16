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
#include <time.h>

int64_t getMicrosecondTimeStamp() 
{
    int64_t retval;
    struct timeval tv; 
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        abort();
    }   
    retval = ((int64_t)tv.tv_sec) * 1000000 + tv.tv_usec;
    return retval;
}
