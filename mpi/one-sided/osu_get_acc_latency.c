#define BENCHMARK "OSU MPI_Get_accumulate latency Test"
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

#define MAX_ALIGNMENT 65536
#define MAX_SIZE (1<<22)
#define MYBUFSIZE (MAX_SIZE + MAX_ALIGNMENT)

#define SKIP_LARGE  10
#define LOOP_LARGE  100
#define LARGE_MESSAGE_SIZE  8192

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

int     skip = 1000;
int     loop = 10000;
double  t_start = 0.0, t_end = 0.0;
char    sbuf_original[MYBUFSIZE];
char    rbuf_original[MYBUFSIZE];
char    cbuf_original[MYBUFSIZE];
char    *sbuf=NULL, *rbuf=NULL, *cbuf=NULL;
MPI_Aint sdisp_remote;
MPI_Aint sdisp_local;

/* Window creation */
typedef enum {
    WIN_CREATE=0,
    WIN_DYNAMIC,
    WIN_ALLOCATE
} WINDOW;

/* Synchronization */
typedef enum {
    LOCK=0,
    FLUSH,
    FLUSH_LOCAL,
    LOCK_ALL,
    PSCW,
    FENCE
} SYNC;

/*Header printout*/
char const *win_info[20] = {
    "MPI_Win_create",
    "MPI_Win_create_dynamic",
    "MPI_Win_allocate",
};

char const *sync_info[20] = {
    "MPI_Win_lock/unlock",
    "MPI_Win_flush",
    "MPI_Win_flush_local",
    "MPI_Win_lock_all/unlock_all",
    "MPI_Win_post/start/complete/wait",
    "MPI_Win_fence",
};

enum po_ret_type {
    po_bad_usage,
    po_help_message,
    po_okay,
};

void print_header (int, WINDOW, SYNC); 
void print_latency (int, int);
void print_help_message (int);
int  process_options (int, char **, WINDOW*, SYNC*, int);
void run_get_acc_with_lock (int, WINDOW);
void run_get_acc_with_fence (int, WINDOW);
void run_get_acc_with_lock_all (int, WINDOW);
void run_get_acc_with_flush (int, WINDOW);
void run_get_acc_with_flush_local (int, WINDOW);
void run_get_acc_with_pscw (int, WINDOW);
void allocate_memory (int, char *, int, WINDOW, MPI_Win *win);

int main (int argc, char *argv[])
{
    SYNC        sync_type=FLUSH; 
    int         rank,nprocs;
    int         page_size;
    int         po_ret = po_okay;
    WINDOW      win_type=WIN_ALLOCATE;
 
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    if(nprocs != 2) {
        if(rank == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }

    po_ret = process_options(argc, argv, &win_type, &sync_type, rank);
    switch (po_ret) {
        case po_bad_usage:
            print_help_message(rank);
            MPI_CHECK(MPI_Finalize());
            return EXIT_FAILURE;
        case po_help_message:
            print_help_message(rank);
            MPI_CHECK(MPI_Finalize());
            return EXIT_SUCCESS;
    }

    page_size = getpagesize();
    assert(page_size <= MAX_ALIGNMENT);
    
    sbuf =
        (char *) (((unsigned long) sbuf_original + (page_size - 1)) /
                page_size * page_size);
    memset(sbuf, 0, MAX_SIZE);

    rbuf =
        (char *) (((unsigned long) rbuf_original + (page_size - 1)) /
                page_size * page_size);
    memset(rbuf, 0, MAX_SIZE);

    cbuf =
        (char *) (((unsigned long) cbuf_original + (page_size - 1)) /
                page_size * page_size);
    memset(cbuf, 0, MAX_SIZE);

    print_header(rank, win_type, sync_type);

    switch (sync_type){
        case LOCK:
            run_get_acc_with_lock(rank, win_type);
            break;
        case LOCK_ALL:
            run_get_acc_with_lock_all(rank, win_type);
            break;
        case PSCW:
            run_get_acc_with_pscw(rank, win_type);
            break;
        case FLUSH_LOCAL:
            run_get_acc_with_flush_local(rank, win_type);
            break;
        case FENCE: 
            run_get_acc_with_fence(rank, win_type);
            break;
        default: 
            run_get_acc_with_flush(rank, win_type);
            break;
    }

    MPI_CHECK(MPI_Finalize());

    return EXIT_SUCCESS;
}

void print_help_message (int rank)
{
    if (rank) return;

    printf("Usage: ./osu_get_acc_latency -w <win_option>  -s < sync_option> [-x ITER] [-i ITER]\n");
    printf("  -x ITER       number of warmup iterations to skip before timing"
            "(default 100)\n");
    printf("  -i ITER       number of iterations for timing (default 10000)\n");
    printf("\n");
    printf("win_option:\n");
    printf("  create            use MPI_Win_create to create an MPI Window object\n");
    printf("  allocate          use MPI_Win_allocate to create an MPI Window object\n");
    printf("  dynamic           use MPI_Win_create_dynamic to create an MPI Window object\n");
    printf("\n");

    printf("sync_option:\n");
    printf("  lock              use MPI_Win_lock/unlock synchronizations calls\n");
    printf("  flush             use MPI_Win_flush synchronization call\n");
    printf("  flush_local       use MPI_Win_flush_local synchronization call\n");
    printf("  lock_all          use MPI_Win_lock_all/unlock_all synchronization calls\n");
    printf("  pscw              use Post/Start/Complete/Wait synchronization calls \n");
    printf("  fence             use MPI_Win_fence synchronization call\n");
    printf("\n");

    fflush(stdout);
}

int process_options(int argc, char *argv[], WINDOW *win, SYNC *sync, int rank)
{
    extern char *optarg;
    extern int  optind;
    extern int opterr;
    int c;

    char const * optstring = "+w:s:h:x:i:";

    if (rank) {
        opterr = 0;
    }

    while((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
            case 'x':
                skip = atoi(optarg);  
                break;
            case 'i':
                loop = atoi(optarg);
                break;
            case 'w':
                if (0 == strcasecmp(optarg, "create")) {
                    *win = WIN_CREATE;
                }
                else if (0 == strcasecmp(optarg, "allocate")) {
                    *win = WIN_ALLOCATE;
                }
                else if (0 == strcasecmp(optarg, "dynamic")) {
                    *win = WIN_DYNAMIC;
                }
                else {
                    return po_bad_usage;
                }
                break;
            case 's':
                if (0 == strcasecmp(optarg, "lock")) {
                    *sync = LOCK;
                }
                else if (0 == strcasecmp(optarg, "flush")) {
                    *sync = FLUSH;
                }
                else if (0 == strcasecmp(optarg, "flush_local")) {
                    *sync = FLUSH_LOCAL;
                }
                else if (0 == strcasecmp(optarg, "lock_all")) {
                    *sync = LOCK_ALL;
                }
                else if (0 == strcasecmp(optarg, "pscw")) {
                    *sync = PSCW;
                }
                else if (0 == strcasecmp(optarg, "fence")) {
                    *sync = FENCE;
                }
                else {
                    return po_bad_usage;
                }
                break;
            case 'h':
                return po_help_message;
            default:
                return po_bad_usage;
        }
    }
    return po_okay;
}

void allocate_memory(int rank, char *rbuf, int size, WINDOW type, MPI_Win *win)
{
    MPI_Status  reqstat;

    switch (type){
        case WIN_DYNAMIC:
            MPI_CHECK(MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, win));
            MPI_CHECK(MPI_Win_attach(*win, (void *)rbuf, size));
            MPI_CHECK(MPI_Get_address(rbuf, &sdisp_local));
            if(rank == 0){
                MPI_CHECK(MPI_Send(&sdisp_local, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(&sdisp_remote, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD, &reqstat));
            }
            else{
                MPI_CHECK(MPI_Recv(&sdisp_remote, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD, &reqstat));
                MPI_CHECK(MPI_Send(&sdisp_local, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD));
            }
            break;
        case WIN_CREATE:
            MPI_CHECK(MPI_Win_create(rbuf, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            break;
        default:
            MPI_CHECK(MPI_Win_allocate(size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, rbuf, win));
            break;
    }
}

void print_header (int rank, WINDOW win, SYNC sync)
{
    if(rank == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "# Window creation: %s\n",
                win_info[win]);
        fprintf(stdout, "# Synchronization: %s\n",
                sync_info[sync]);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);
    }
}

void print_latency(int rank, int size)
{
    if (rank == 0) {
        fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                FLOAT_PRECISION, (t_end - t_start) * 1.0e6 /  loop);
        fflush(stdout);
    }
}

/*Run Get_accumulate with flush */
void run_get_acc_with_flush(int rank, WINDOW type)
{
    int size, i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    for (size = 0; size <= MAX_SIZE; size = (size ? size * 2 : size + 1)) {
        allocate_memory(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

        if(size > LARGE_MESSAGE_SIZE) {
             loop = LOOP_LARGE;
             skip = SKIP_LARGE;
        }

        if(rank == 0) {
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win));
            for (i = 0; i <  skip +  loop; i++) {
                if (i ==  skip) {
                    t_start = MPI_Wtime ();
                }
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 1, disp, size,
                    MPI_CHAR, MPI_SUM, win));
                MPI_CHECK(MPI_Win_flush(1, win));
            }
            t_end = MPI_Wtime ();
            MPI_CHECK(MPI_Win_unlock(1, win));
        }                

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        
        print_latency(rank, size);
        
        MPI_Win_free(&win);
    }
}

/*Run Get_accumulate with flush local*/
void run_get_acc_with_flush_local(int rank, WINDOW type)
{
    int size, i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    for (size = 0; size <= MAX_SIZE; size = (size ? size * 2 : size + 1)) {
        allocate_memory(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

        if(size > LARGE_MESSAGE_SIZE) {
             loop = LOOP_LARGE;
             skip = SKIP_LARGE;
        }

        if(rank == 0) {
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win));
            for (i = 0; i <  skip +  loop; i++) {
                if (i ==  skip) {
                    t_start = MPI_Wtime ();
                }
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 1, disp, size,
                    MPI_CHAR, MPI_SUM, win));
                MPI_CHECK(MPI_Win_flush_local(1, win));
            }
            t_end = MPI_Wtime ();
            MPI_CHECK(MPI_Win_unlock(1, win));
        }                

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        
        print_latency(rank, size);

        MPI_Win_free(&win);
    }
}

/*Run Get_accumulate with Lock_all/unlock_all */
void run_get_acc_with_lock_all(int rank, WINDOW type)
{
    int size, i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    for (size = 0; size <= MAX_SIZE; size = (size ? size * 2 : size + 1)) {
        allocate_memory(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

        if(size > LARGE_MESSAGE_SIZE) {
             loop = LOOP_LARGE;
             skip = SKIP_LARGE;
        }

        if(rank == 0) {
            for (i = 0; i <  skip +  loop; i++) {
                if (i ==  skip) {
                    t_start = MPI_Wtime ();
                }
                MPI_CHECK(MPI_Win_lock_all(0, win));
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 1, disp, size,
                    MPI_CHAR, MPI_SUM, win));
                MPI_CHECK(MPI_Win_unlock_all(win));
            }
            t_end = MPI_Wtime ();
        }                

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_latency(rank, size);
        
        MPI_Win_free(&win);
    }
}

/*Run Get_accumulate with Lock/unlock */
void run_get_acc_with_lock(int rank, WINDOW type)
{
    int size, i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    for (size = 0; size <= MAX_SIZE; size = (size ? size * 2 : size + 1)) {
        allocate_memory(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

        if(size > LARGE_MESSAGE_SIZE) {
             loop = LOOP_LARGE;
             skip = SKIP_LARGE;
        }

        if(rank == 0) {
            for (i = 0; i <  skip +  loop; i++) {
                if (i ==  skip) {
                    t_start = MPI_Wtime ();
                }
                MPI_CHECK(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win));
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 1, disp, size,
                    MPI_CHAR, MPI_SUM, win));
                MPI_CHECK(MPI_Win_unlock(1, win));
            }
            t_end = MPI_Wtime ();
        }                

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        print_latency(rank, size);
        
        MPI_Win_free(&win);
    }
}

/*Run Get_accumulate with Fence */
void run_get_acc_with_fence(int rank, WINDOW type)
{
    int size, i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    for (size = 0; size <= MAX_SIZE; size = (size ? size * 2 : size + 1)) {
        allocate_memory(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

        if(size > LARGE_MESSAGE_SIZE) {
             loop = LOOP_LARGE;
             skip = SKIP_LARGE;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if(rank == 0) {
            for (i = 0; i <  skip +  loop; i++) {
                if (i ==  skip) {
                    t_start = MPI_Wtime ();
                }
                MPI_CHECK(MPI_Win_fence(0, win));
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 1, disp, size,
                    MPI_CHAR, MPI_SUM, win));
                MPI_CHECK(MPI_Win_fence(0, win));
                MPI_CHECK(MPI_Win_fence(0, win));
            }
            t_end = MPI_Wtime ();
        } else {
            for (i = 0; i <  skip +  loop; i++) {
                MPI_CHECK(MPI_Win_fence(0, win));
                MPI_CHECK(MPI_Win_fence(0, win));
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 0, disp, size,
                    MPI_CHAR, MPI_SUM, win));
                MPI_CHECK(MPI_Win_fence(0, win));
            }
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank == 0) {
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, (t_end - t_start) * 1.0e6 /  loop / 2);
            fflush(stdout);
        }

        MPI_Win_free(&win);
    }
}

/*Run GET with Post/Start/Complete/Wait */
void run_get_acc_with_pscw(int rank, WINDOW type)
{
    int destrank, size, i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    MPI_Group       comm_group, group;
    MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &comm_group));

    for (size = 0; size <= MAX_SIZE; size = (size ? size * 2 : 1)) {
        allocate_memory(rank, rbuf, size, type, &win);

        if (type == WIN_DYNAMIC) {
            disp = sdisp_remote;
        }

        if (size > LARGE_MESSAGE_SIZE) {
             loop = LOOP_LARGE;
             skip = SKIP_LARGE;
        }
        
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank == 0) {
            destrank = 1;

            MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            for (i = 0; i <  skip +  loop; i++) {
                MPI_CHECK(MPI_Win_start (group, 0, win));
                if (i ==  skip) {
                    t_start = MPI_Wtime ();
                }
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 1, disp, size,
                    MPI_CHAR, MPI_SUM, win));
                MPI_CHECK(MPI_Win_complete(win));
                MPI_CHECK(MPI_Win_post(group, 0, win));
                MPI_CHECK(MPI_Win_wait(win));
            }

            t_end = MPI_Wtime ();
        } else {
            /* rank=1 */
            destrank = 0;

            MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

            for (i = 0; i <  skip +  loop; i++) {
                MPI_CHECK(MPI_Win_post(group, 0, win));
                MPI_CHECK(MPI_Win_wait(win));
                MPI_CHECK(MPI_Win_start(group, 0, win));
                MPI_CHECK(MPI_Get_accumulate(sbuf, size, MPI_CHAR, cbuf, size, MPI_CHAR, 0, disp, size,
                    MPI_CHAR, MPI_SUM, win));
                MPI_CHECK(MPI_Win_complete(win));
            }
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if (rank == 0) {
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, (t_end - t_start) * 1.0e6 /  loop / 2);
            fflush(stdout);
        }

        MPI_CHECK(MPI_Group_free(&group));

        MPI_Win_free(&win);
    }

    MPI_CHECK(MPI_Group_free(&comm_group));
}
/* vi: set sw=4 sts=4 tw=80: */
