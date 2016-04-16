#define BENCHMARK "OSU MPI_Get%s Bandwidth Test"
/*
 * Copyright (C) 2003-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.            
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include "osu_1sc.h"

#define MAX_ALIGNMENT 65536
#define MAX_SIZE (1<<22)

#define SKIP_LARGE  10
#define LOOP_LARGE  30
#define WINDOW_SIZE_LARGE 32
#define LARGE_MESSAGE_SIZE 8192

#define MYBUFSIZE ((MAX_SIZE * WINDOW_SIZE_LARGE) + MAX_ALIGNMENT)

#ifdef PACKAGE_VERSION
#   define HEADER "# " BENCHMARK " v" PACKAGE_VERSION "\n"
#else
#   define HEADER "# " BENCHMARK "\n"
#endif

double  t_start = 0.0, t_end = 0.0;
char    sbuf_original[MYBUFSIZE];
char    rbuf_original[MYBUFSIZE];
char    *sbuf=NULL, *rbuf=NULL;

void print_header (int, WINDOW, SYNC); 
void print_bw (int, int, double);
void run_get_with_lock (int, WINDOW);
void run_get_with_fence (int, WINDOW);
#if MPI_VERSION >= 3
void run_get_with_lock_all (int, WINDOW);
void run_get_with_flush (int, WINDOW);
void run_get_with_flush_local (int, WINDOW);
#endif
void run_get_with_pscw (int, WINDOW);

int main (int argc, char *argv[])
{
    int         rank,nprocs;
    int         po_ret = po_okay;
#if MPI_VERSION >= 3
    WINDOW      win_type=WIN_ALLOCATE;
    SYNC        sync_type=FLUSH;
#else
    WINDOW      win_type=WIN_CREATE;
    SYNC        sync_type=LOCK;
#endif   
 
    po_ret = process_options(argc, argv, &win_type, &sync_type, all_sync);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
           fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    if (0 == rank) {
        switch (po_ret) {
            case po_cuda_not_avail:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case po_openacc_not_avail:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case po_bad_usage:
            case po_help_message:
                usage(all_sync, "osu_get_bw");
                break;
        }
    }

    switch (po_ret) {
        case po_cuda_not_avail:
        case po_openacc_not_avail:
        case po_bad_usage:
            MPI_Finalize();
            exit(EXIT_FAILURE);
        case po_help_message:
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        case po_okay:
            break;
    }

    if(nprocs != 2) {
        if(rank == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }

    print_header(rank, win_type, sync_type);

    switch (sync_type){
        case LOCK:
            run_get_with_lock(rank, win_type);
            break;
        case PSCW:
            run_get_with_pscw(rank, win_type);
            break;
        case FENCE: 
            run_get_with_fence(rank, win_type);
            break;
#if MPI_VERSION >= 3
        case LOCK_ALL:
            run_get_with_lock_all(rank, win_type);
            break;
        case FLUSH_LOCAL: 
            run_get_with_flush_local(rank, win_type);
            break;
        default: 
            run_get_with_flush(rank, win_type);
            break;
#endif
    }

    MPI_CHECK(MPI_Finalize());

    if (none != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

void print_header (int rank, WINDOW win, SYNC sync)
{
    if(rank == 0) {
        switch (options.accel) {
            case cuda:
                printf(HEADER, "-CUDA");
                break;
            case openacc:
                printf(HEADER, "-OPENACC");
                break;
            default:
                printf(HEADER, "");
                break;
        }

        fprintf(stdout, "# Window creation: %s\n",
                win_info[win]);
        fprintf(stdout, "# Synchronization: %s\n",
                sync_info[sync]);

        switch (options.accel) {
            case cuda:
            case openacc:
                printf("# Rank 0 Memory on %s and Rank 1 Memory on %s\n",
                        'D' == options.rank0 ? "DEVICE (D)" : "HOST (H)",
                        'D' == options.rank1 ? "DEVICE (D)" : "HOST (H)");
            default:
                fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Bandwidth (MB/s)");
                fflush(stdout);
        }
    }
}

void print_bw(int rank, int size, double t)
{
    if (rank == 0) {
        double tmp = size / 1e6 * options.loop * WINDOW_SIZE_LARGE;

        fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                FLOAT_PRECISION, tmp / t);
        fflush(stdout);
    }
}

#if MPI_VERSION >= 3
/*Run GET with flush local */
void run_get_with_flush_local (int rank, WINDOW type)
{
    double t = 0.0;
    int size, i, j;
    MPI_Aint disp = 0;
    MPI_Win     win;

    int window_size = WINDOW_SIZE_LARGE;
    for (size = 1; size <= MAX_SIZE; size = size * 2) {
        allocate_memory(rank, sbuf_original, rbuf_original, &sbuf, &rbuf, &sbuf, size*window_size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = LOOP_LARGE;
            options.skip = SKIP_LARGE;
        }
        if (rank == 0) {
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
            for (i = 0; i < options.skip + options.loop; i++) {
                if (i == options.skip) {
                    t_start = MPI_Wtime ();
                }
                for(j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Get(rbuf+(j*size), size, MPI_CHAR, 1, disp + (j * size), size, MPI_CHAR,
                            win));
                }
                MPI_CHECK(MPI_Win_flush_local(1, win));
            }
            t_end = MPI_Wtime();
            MPI_CHECK(MPI_Win_unlock(1, win ));
            t = t_end - t_start;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_bw(rank, size, t);

        MPI_Win_free(&win);
    }
}

/*Run GET with flush */
void run_get_with_flush (int rank, WINDOW type)
{
    double t= 0.0;
    int size, i, j;
    MPI_Aint disp = 0;
    MPI_Win     win;

    int window_size = WINDOW_SIZE_LARGE;
    for (size = 1; size <= MAX_SIZE; size = size * 2) {
        allocate_memory(rank, sbuf_original, rbuf_original, &sbuf, &rbuf, &sbuf, size*window_size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = LOOP_LARGE;
            options.skip = SKIP_LARGE;
        }

        if (rank == 0) {
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
            for (i = 0; i < options.skip + options.loop; i++) {
                if (i == options.skip) {
                    t_start = MPI_Wtime ();
                }
                for(j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Get(rbuf+(j*size), size, MPI_CHAR, 1, disp + (j * size), size, MPI_CHAR,
                            win));
                }
                MPI_CHECK(MPI_Win_flush(1, win));
            }
            t_end = MPI_Wtime();
            MPI_CHECK(MPI_Win_unlock(1, win));
            t = t_end - t_start;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_bw(rank, size, t);

        MPI_Win_free(&win);
    }
}

/*Run GET with Lock_all/unlock_all */
void run_get_with_lock_all (int rank, WINDOW type)
{
    double t = 0.0;
    int size, i, j;
    MPI_Aint disp = 0;
    MPI_Win     win;

    int window_size = WINDOW_SIZE_LARGE;
    for (size = 1; size <= MAX_SIZE; size = size * 2) {
        allocate_memory(rank, sbuf_original, rbuf_original, &sbuf, &rbuf, &sbuf, size*window_size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = LOOP_LARGE;
            options.skip = SKIP_LARGE;
        }
        if (rank == 0) {
            for (i = 0; i < options.skip + options.loop; i++) {
                if (i == options.skip) {
                    t_start = MPI_Wtime ();
                }
                MPI_CHECK(MPI_Win_lock_all(0, win));
                for(j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Get(rbuf+(j*size), size, MPI_CHAR, 1, disp + (j * size), size, MPI_CHAR,
                            win));
                }
                MPI_CHECK(MPI_Win_unlock_all(win));
            }
            t_end = MPI_Wtime();
            t = t_end - t_start;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_bw(rank, size, t);

        MPI_Win_free(&win);
    }
}
#endif

/*Run GET with Lock/unlock */
void run_get_with_lock(int rank, WINDOW type)
{
    double t = 0.0;
    int size, i, j;
    MPI_Aint disp = 0;
    MPI_Win     win;

    int window_size = WINDOW_SIZE_LARGE;
    for (size = 1; size <= MAX_SIZE; size = size * 2) {
        allocate_memory(rank, sbuf_original, rbuf_original, &sbuf, &rbuf, &sbuf, size*window_size, type, &win);

#if MPI_VERSION >= 3
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
#endif

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = LOOP_LARGE;
            options.skip = SKIP_LARGE;
        }
        if (rank == 0) {
            for (i = 0; i < options.skip + options.loop; i++) {
                if (i == options.skip) {
                    t_start = MPI_Wtime ();
                }
                MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
                for(j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Get(rbuf+(j*size), size, MPI_CHAR, 1, disp + (j * size), size, MPI_CHAR,
                            win));
                }
                MPI_CHECK(MPI_Win_unlock(1, win ));
            }
            t_end = MPI_Wtime();
            t = t_end - t_start;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_bw(rank, size, t);

        MPI_Win_free(&win);
    }
}

/*Run GET with Fence */
void run_get_with_fence(int rank, WINDOW type)
{
    double t = 0.0; 
    int size, i, j;
    MPI_Aint disp = 0;
    MPI_Win     win;

    int window_size = WINDOW_SIZE_LARGE;
    for (size = 1; size <= MAX_SIZE; size = size * 2) {
        allocate_memory(rank, sbuf_original, rbuf_original, &sbuf, &rbuf, &sbuf, size*window_size, type, &win);

#if MPI_VERSION >= 3
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
#endif

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = LOOP_LARGE;
            options.skip = SKIP_LARGE;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if(rank == 0) {
            for (i = 0; i < options.skip + options.loop; i++) {
                if (i == options.skip) {
                    t_start = MPI_Wtime ();
                }
                MPI_CHECK(MPI_Win_fence(0, win));
                for(j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Get(rbuf+(j*size), size, MPI_CHAR, 1, disp + (j * size), size, MPI_CHAR,
                            win));
                }
                MPI_CHECK(MPI_Win_fence(0, win));
            }
            t_end = MPI_Wtime ();
            t = t_end - t_start;
        } else {
            for (i = 0; i < options.skip + options.loop; i++) {
                MPI_CHECK(MPI_Win_fence(0, win));
                MPI_CHECK(MPI_Win_fence(0, win));
            }
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_bw(rank, size, t);

        MPI_Win_free(&win);
    }
}

/*Run GET with Post/Start/Complete/Wait */
void run_get_with_pscw(int rank, WINDOW type)
{
    double t = 0.0; 
    int destrank, size, i, j;
    MPI_Aint disp = 0;
    MPI_Win     win;
    MPI_Group       comm_group, group;
    MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &comm_group));

    int window_size = WINDOW_SIZE_LARGE;
    for (size = 1; size <= MAX_SIZE; size = (size ? size * 2 : 1)) {
        allocate_memory(rank, sbuf_original, rbuf_original, &sbuf, &rbuf, &sbuf, size*window_size, type, &win);

#if MPI_VERSION >= 3
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
#endif

        if (size > LARGE_MESSAGE_SIZE) {
            options.loop = LOOP_LARGE;
            options.skip = SKIP_LARGE;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank == 0) {

            destrank = 1;
            MPI_CHECK(MPI_Group_incl (comm_group, 1, &destrank, &group));
            for (i = 0; i < options.skip + options.loop; i++) {
                MPI_CHECK(MPI_Win_start(group, 0, win));
                if (i == options.skip) {
                    t_start = MPI_Wtime ();
                }
                for(j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Get(rbuf + j*size, size, MPI_CHAR, 1, disp + (j*size), size, MPI_CHAR,
                            win));
                }
                MPI_CHECK(MPI_Win_complete(win));
            }
            t_end = MPI_Wtime();
            t = t_end - t_start;
        } else {

            destrank = 0;
            MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));
            for (i = 0; i < options.skip + options.loop; i++) {
                MPI_CHECK(MPI_Win_post(group, 0, win));
                MPI_CHECK(MPI_Win_wait(win));
            }
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_bw(rank, size, t);

        MPI_CHECK(MPI_Group_free(&group));

        MPI_Win_free(&win);
    }
    MPI_CHECK(MPI_Group_free(&comm_group));
}
/* vi: set sw=4 sts=4 tw=80: */
