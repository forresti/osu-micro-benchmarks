/*
 * Copyright (C) 2003-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <getopt.h>

#ifdef _ENABLE_OPENACC_
#include <openacc.h>
#endif

#ifdef _ENABLE_CUDA_
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define MAX_ALIGNMENT 65536

#ifndef FIELD_WIDTH
#   define FIELD_WIDTH 20
#endif

#ifndef FLOAT_PRECISION
#   define FLOAT_PRECISION 2
#endif

#define CHECK(stmt)                                              \
do {                                                             \
   int errno = (stmt);                                           \
   if (0 != errno) {                                             \
       fprintf(stderr, "[%s:%d] function call failed with %d \n",\
        __FILE__, __LINE__, errno);                              \
       exit(EXIT_FAILURE);                                       \
   }                                                             \
   assert(0 == errno);                                           \
} while (0)

#define MPI_CHECK(stmt)                                          \
do {                                                             \
   int mpi_errno = (stmt);                                       \
   if (MPI_SUCCESS != mpi_errno) {                               \
       fprintf(stderr, "[%s:%d] MPI call failed with %d \n",     \
        __FILE__, __LINE__,mpi_errno);                           \
       exit(EXIT_FAILURE);                                       \
   }                                                             \
   assert(MPI_SUCCESS == mpi_errno);                             \
} while (0)

#ifdef _ENABLE_CUDA_
#   define CUDA_ENABLED 1
#else
#   define CUDA_ENABLED 0
#endif

#ifdef _ENABLE_OPENACC_
#   define OPENACC_ENABLED 1
#else
#   define OPENACC_ENABLED 0
#endif

/*structures, enumerators and such*/
/* Window creation */
typedef enum {
    WIN_CREATE=0,
#if MPI_VERSION >= 3
    WIN_ALLOCATE,
    WIN_DYNAMIC
#endif
} WINDOW;

/* Synchronization */
typedef enum {
    LOCK=0,
    PSCW,
    FENCE,
#if MPI_VERSION >= 3
    FLUSH,
    FLUSH_LOCAL,
    LOCK_ALL,
#endif
} SYNC;

enum po_ret_type {
    po_cuda_not_avail,
    po_openacc_not_avail,
    po_bad_usage,
    po_help_message,
    po_okay,
};

enum accel_type {
    none,
    cuda,
    openacc
};

enum options_type {
   all_sync,
   active_sync
};

struct options_t {
    char rank0;
    char rank1;
    enum accel_type accel;
    int loop;
    int loop_large;
    int skip;
    int skip_large;
};

extern struct options_t options;

/*variables*/
extern char const *win_info[20];
extern char const *sync_info[20];

#ifdef _ENABLE_CUDA_
extern CUcontext cuContext;
#endif

extern MPI_Aint disp_remote;
extern MPI_Aint disp_local;

/*function declarations*/
void usage (int, char const *);
int  process_options (int, char **, WINDOW*, SYNC*, int);
void allocate_memory(int, char *, char *, char **, char **,
            char **win_base, int, WINDOW, MPI_Win *);
void free_memory (void *, void *, MPI_Win, int);
void allocate_atomic_memory(int, char *, char *, char *, 
            char *, char **, char **, char **, char **,
            char **win_base, int, WINDOW, MPI_Win *);
void free_atomic_memory (void *, void *, void *, void *, MPI_Win, int);
int init_accel ();
int cleanup_accel ();
