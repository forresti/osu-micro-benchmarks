/*
 * Copyright (C) 2002-2015 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#ifndef _OSU_COMMON_H_
#define _OSU_COMMON_H_

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

#define TIME() getMicrosecondTimeStamp()

#ifdef __cplusplus
extern "C" {
#endif /* #ifdef __cplusplus */

double getMicrosecondTimeStamp();

#ifdef __cplusplus
}
#endif /* #ifdef __cplusplus */

#endif /* _OSU_COMMON_H */
