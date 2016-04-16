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

#ifdef _ENABLE_CUDA_
CUcontext cuContext;
#endif

char const *win_info[20] = {
    "MPI_Win_create",
#if MPI_VERSION >=3
    "MPI_Win_allocate",
    "MPI_Win_create_dynamic",
#endif
};

char const *sync_info[20] = {
    "MPI_Win_lock/unlock",
    "MPI_Win_post/start/complete/wait",
    "MPI_Win_fence",
#if MPI_VERSION >=3
    "MPI_Win_flush",
    "MPI_Win_flush_local",
    "MPI_Win_lock_all/unlock_all",
#endif
};

MPI_Aint disp_remote;
MPI_Aint disp_local;

int mem_on_dev; 
struct options_t options;

void 
usage (int options_type, char const * name) 
{
    if (CUDA_ENABLED || OPENACC_ENABLED) {
        printf("Usage: %s [options] [RANK0 RANK1] \n", name);
        printf("RANK0 and RANK1 may be `D' or `H' which specifies whether\n"
               "the buffer is allocated on the accelerator device or host\n"
               "memory for each mpi rank\n\n");
    } else { 
        printf("Usage: %s [options] \n", name);
    }

    printf("options:\n");

    printf("  -d <type>       accelerator device buffers can be of <type> "
                            "`cuda' or `openacc'\n");
    printf("\n");

#if MPI_VERSION >= 3
    printf("  -w <win_option>\n");
    printf("            <win_option> can be any of the follows:\n");
    printf("            create            use MPI_Win_create to create an MPI Window object\n");
    if (CUDA_ENABLED || OPENACC_ENABLED) {
        printf("            allocate          use MPI_Win_allocate to create an MPI Window object (not valid when using device memory)\n");
    } else { 
        printf("            allocate          use MPI_Win_allocate to create an MPI Window object\n");
    }
    printf("            dynamic           use MPI_Win_create_dynamic to create an MPI Window object\n");
    printf("\n");
#endif

    printf("  -s <sync_option>\n");
    printf("            <sync_option> can be any of the follows:\n");
    printf("            pscw              use Post/Start/Complete/Wait synchronization calls \n");
    printf("            fence             use MPI_Win_fence synchronization call\n");
    if (options_type == all_sync) { 
        printf("            lock              use MPI_Win_lock/unlock synchronizations calls\n");
#if MPI_VERSION >= 3
        printf("            flush             use MPI_Win_flush synchronization call\n");
        printf("            flush_local       use MPI_Win_flush_local synchronization call\n");
        printf("            lock_all          use MPI_Win_lock_all/unlock_all synchronization calls\n");
#endif
    }
    printf("\n");
    printf("  -x ITER       number of warmup iterations to skip before timing"
            "(default 100)\n");
    printf("  -i ITER       number of iterations for timing (default 10000)\n");

    printf("  -h            print this help message\n");

    fflush(stdout);
}

int
init_accel (void)
{
#if defined(_ENABLE_OPENACC_) || defined(_ENABLE_CUDA_)
     char * str;
     int local_rank, dev_count;
     int dev_id = 0;
#endif
#ifdef _ENABLE_CUDA_
     CUresult curesult = CUDA_SUCCESS;
     CUdevice cuDevice;
#endif

     switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            if ((str = getenv("LOCAL_RANK")) != NULL) {
                cudaGetDeviceCount(&dev_count);
                local_rank = atoi(str);
                dev_id = local_rank % dev_count;
            }

            curesult = cuInit(0);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }

            curesult = cuDeviceGet(&cuDevice, dev_id);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }

            curesult = cuCtxCreate(&cuContext, 0, cuDevice);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            if ((str = getenv("LOCAL_RANK")) != NULL) {
                dev_count = acc_get_num_devices(acc_device_not_host);
                fprintf(stderr, "dev_count : %d \n", dev_count);
                local_rank = atoi(str);
                dev_id = local_rank % dev_count;
            }

            acc_set_device_num (dev_id, acc_device_not_host);
            break;
#endif
        default:
            fprintf(stderr, "Invalid device type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

int
cleanup_accel (void)
{
#ifdef _ENABLE_CUDA_
     CUresult curesult = CUDA_SUCCESS;
#endif

     switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            curesult = cuCtxDestroy(cuContext);

            if (curesult != CUDA_SUCCESS) {
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            acc_shutdown(acc_device_not_host);
            break;
#endif
        default:
            fprintf(stderr, "Invalid accel type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

int 
process_options(int argc, char *argv[], WINDOW *win, SYNC *sync, int options_type) 
{
    extern char *optarg;
    extern int  optind;
    extern int opterr;
    int c;

    /*
     * set default options
     */
    options.rank0 = 'H';
    options.rank1 = 'H';
    options.loop = 10000;
    options.loop_large = 1000;
    options.skip = 100;
    options.skip_large = 10;

    if (CUDA_ENABLED) {
        options.accel = cuda;
    }
    else if (OPENACC_ENABLED) {
        options.accel = openacc;
    }
    else {
        options.accel = none;
    }

#if MPI_VERSION >= 3
    char const * optstring = (CUDA_ENABLED || OPENACC_ENABLED) ? "+d:w:s:h:x:i:" : "+w:s:h:x:i:";
#else
    char const * optstring = (CUDA_ENABLED || OPENACC_ENABLED) ? "+d:s:h:x:i:" : "+s:h:x:i:";
#endif

    while((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
            case 'x':
                options.skip = atoi(optarg);
                break;
            case 'i':                                
                options.loop = atoi(optarg);
                break;
            case 'd':
                /* optarg should contain cuda or openacc */
                if (0 == strncasecmp(optarg, "cuda", 10)) {
                    if (!CUDA_ENABLED) {
                        return po_cuda_not_avail;
                    }
                    options.accel = cuda;
                }
                else if (0 == strncasecmp(optarg, "openacc", 10)) {
                    if (!OPENACC_ENABLED) {
                        return po_openacc_not_avail;
                    }
                    options.accel = openacc;
                }
                else {
                    return po_bad_usage;
                }
                break;
#if MPI_VERSION >= 3
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
#endif
            case 's':
                if (0 == strcasecmp(optarg, "pscw")) {
                    *sync = PSCW;
                }
                else if (0 == strcasecmp(optarg, "fence")) {
                    *sync = FENCE;
                } 
                else if (options_type == all_sync) { 
                    if (0 == strcasecmp(optarg, "lock")) {
                        *sync = LOCK;
                    }
#if MPI_VERSION >= 3
                    else if (0 == strcasecmp(optarg, "flush")) {
                        *sync = FLUSH;
                    }
                    else if (0 == strcasecmp(optarg, "flush_local")) {
                        *sync = FLUSH_LOCAL;
                    }
                    else if (0 == strcasecmp(optarg, "lock_all")) {
                        *sync = LOCK_ALL;
                    }
#endif
                    else { 
                        return po_bad_usage;
                    }
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

    if (CUDA_ENABLED || OPENACC_ENABLED) {
        if ((optind + 2) == argc) {
            options.rank0 = argv[optind][0];
            options.rank1 = argv[optind + 1][0];

            switch (options.rank0) {
                case 'D':
                case 'H':
                    break;
                default:
                    return po_bad_usage;
            }

            switch (options.rank1) {
                case 'D':
                case 'H':
                    break;
                default:
                    return po_bad_usage;
            }
        }
        else if (optind != argc) {
            return po_bad_usage;
        }

#if MPI_VERSION >= 3
        if ((options.rank0 == 'D' || options.rank1 == 'D') 
            && *win == WIN_ALLOCATE) {
            *win = WIN_CREATE;
        }
#endif
    }

    return po_okay;
}

int
allocate_device_buffer (char ** buffer, int size)
{
#ifdef _ENABLE_CUDA_
    cudaError_t cuerr = cudaSuccess;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            cuerr = cudaMalloc((void **)buffer, size);

            if (cudaSuccess != cuerr) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            *buffer = acc_malloc(size);
            if (NULL == *buffer && size != 0) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
        default:
            fprintf(stderr, "Could not allocate device memory\n");
            return 1;
    }

    return 0;
}

int
free_device_buffer (void * buf)
{
    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            cudaFree(buf);
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            acc_free(buf);
            break;
#endif
        default:
            /* unknown device */
            return 1;
    }

    return 0;
}

void *
align_buffer (void * ptr, unsigned long align_size)
{
    return (void *)(((unsigned long)ptr + (align_size - 1)) / align_size *
            align_size);
}

void
set_device_memory (void * ptr, int data, size_t size)
{
#ifdef _ENABLE_OPENACC_
    size_t i;
    char * p = (char *)ptr;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            cudaMemset(ptr, data, size);
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
#pragma acc parallel copyin(size) deviceptr(p)
            for(i = 0; i < size; i++) {
                p[i] = data;
            }
            break;
#endif
        default:
            break;
    }
}

void 
allocate_memory(int rank, char *sbuf_orig, char *rbuf_orig, char **sbuf, char **rbuf, 
            char **win_base, int size, WINDOW type, MPI_Win *win)
{
    int page_size;

    page_size = getpagesize();
    assert(page_size <= MAX_ALIGNMENT);

    if (rank == 0) {
        mem_on_dev = ('D' == options.rank0) ? 1 : 0;
    } else {
        mem_on_dev = ('D' == options.rank1) ? 1 : 0;
    }

    if (mem_on_dev) {
         CHECK(allocate_device_buffer(sbuf, size));
         set_device_memory(*sbuf, 'a', size);
         CHECK(allocate_device_buffer(rbuf, size));
         set_device_memory(*rbuf, 'b', size);
    }
    else {
         *sbuf = (char *)align_buffer((void *)sbuf_orig, page_size);
         memset(*sbuf, 'a', size);
         *rbuf = (char *)align_buffer((void *)rbuf_orig, page_size);
         memset(*rbuf, 'b', size);
    }

#if MPI_VERSION >= 3
    MPI_Status  reqstat;

    switch (type) {
        case WIN_CREATE:
            MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            break;
        case WIN_DYNAMIC:
            MPI_CHECK(MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, win));
            MPI_CHECK(MPI_Win_attach(*win, (void *)*win_base, size));
            MPI_CHECK(MPI_Get_address(*win_base, &disp_local));
            if(rank == 0){
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD, &reqstat));
            }
            else{
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD, &reqstat));
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD));
            }
            break;
        default:
            if (mem_on_dev) {
                MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            } else {
                MPI_CHECK(MPI_Win_allocate(size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, *win_base, win));
            }
            break;
    }
#else
    MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
#endif
}

void 
allocate_atomic_memory(int rank, char *sbuf_orig, char *rbuf_orig, char *tbuf_orig, 
            char *cbuf_orig, char **sbuf, char **rbuf, char **tbuf, 
            char **cbuf, char **win_base, int size, WINDOW type, MPI_Win *win)
{
    int page_size;

    page_size = getpagesize();
    assert(page_size <= MAX_ALIGNMENT);

    if (rank == 0) {
        mem_on_dev = ('D' == options.rank0) ? 1 : 0;
    } else {
        mem_on_dev = ('D' == options.rank1) ? 1 : 0;
    }

    if (mem_on_dev) {
         CHECK(allocate_device_buffer(sbuf, size));
         set_device_memory(*sbuf, 'a', size);
         CHECK(allocate_device_buffer(rbuf, size));
         set_device_memory(*rbuf, 'b', size);
         CHECK(allocate_device_buffer(tbuf, size));
         set_device_memory(*tbuf, 'c', size);
         if (cbuf != NULL) {
             CHECK(allocate_device_buffer(cbuf, size));
             set_device_memory(*cbuf, 'a', size);
         }
    }
    else {
         *sbuf = (char *)align_buffer((void *)sbuf_orig, page_size);
         memset(*sbuf, 'a', size);
         *rbuf = (char *)align_buffer((void *)rbuf_orig, page_size);
         memset(*rbuf, 'b', size);
         *tbuf = (char *)align_buffer((void *)tbuf_orig, page_size);
         memset(*tbuf, 'c', size);
         if (cbuf != NULL) {
             *cbuf = (char *)align_buffer((void *)cbuf_orig, page_size);
             memset(*cbuf, 'a', size);
         }
    }

#if MPI_VERSION >= 3
    MPI_Status  reqstat;

    switch (type) {
        case WIN_CREATE:
            MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            break;
        case WIN_DYNAMIC:
            MPI_CHECK(MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, win));
            MPI_CHECK(MPI_Win_attach(*win, (void *)*win_base, size));
            MPI_CHECK(MPI_Get_address(*win_base, &disp_local));
            if(rank == 0){
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 1, 1, MPI_COMM_WORLD, &reqstat));
            }
            else{
                MPI_CHECK(MPI_Recv(&disp_remote, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD, &reqstat));
                MPI_CHECK(MPI_Send(&disp_local, 1, MPI_AINT, 0, 1, MPI_COMM_WORLD));
            }
            break;
        default:
            if (mem_on_dev) {
                MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
            } else {
                MPI_CHECK(MPI_Win_allocate(size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, *win_base, win));
            }
            break;
    }
#else
    MPI_CHECK(MPI_Win_create(*win_base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, win));
#endif
}

void 
free_atomic_memory (void *sbuf, void *rbuf, void *tbuf, void *cbuf, MPI_Win win, int rank)
{
    MPI_Win_free(&win);

    switch (rank) {
        case 0:
            if ('D' == options.rank0) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
                free_device_buffer(tbuf);
                if (cbuf != NULL)
                    free_device_buffer(cbuf);
            }
            break;
        case 1:
            if ('D' == options.rank1) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
                free_device_buffer(tbuf);
                if (cbuf != NULL)
                    free_device_buffer(cbuf);
            }
            break;
    }
}

void 
free_memory (void *sbuf, void *rbuf, MPI_Win win, int rank)
{
    MPI_Win_free(&win);

    switch (rank) {
        case 0:
            if ('D' == options.rank0) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            }
            break;
        case 1:
            if ('D' == options.rank1) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            }
            break;
    }
}

