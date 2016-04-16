/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *    
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *      
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level directory.
 */

/*
 * HEADER FILES
 */
#include "osu_pt2pt.h"

MPI_Request request[MAX_REQ_NUM];
MPI_Status  reqstat[MAX_REQ_NUM];
MPI_Request send_request[MAX_REQ_NUM];
MPI_Request recv_request[MAX_REQ_NUM];

#ifdef _ENABLE_CUDA_
CUcontext cuContext;
#endif

static char const * benchmark_header = NULL;
static int benchmark_type;
struct options_t options;

void
set_header (const char * header)
{
    benchmark_header = header;
}

static struct {
    char const * message;
    char const * optarg;
    int opt;
} bad_usage;


void
usage (char const * name)
{
    if (CUDA_ENABLED || OPENACC_ENABLED) {
        printf("Usage: %s [options] [RANK0 RANK1]\n\n", name);
        printf("RANK0 and RANK1 may be `D', `H', or 'M' which specifies whether\n"
               "the buffer is allocated on the accelerator device memory, host\n"
               "memory or using CUDA Unified memory respectively for each mpi rank\n\n");
    }

    else {
        printf("Usage: %s [options]\n\n", name);
    }

    printf("options:\n");

    if (CUDA_ENABLED || OPENACC_ENABLED) {
        printf("  -d TYPE       accelerator device buffers can be of TYPE "
                "`cuda' or `openacc'\n");
    }

    printf("  -x ITER       number of warmup iterations to skip before timing"
            "(default %d)\n",
            benchmark_type == BW ? BW_SKIP_SMALL : LAT_SKIP_SMALL);
    printf("  -i ITER       number of iterations for timing (default %d)\n",
            benchmark_type == BW ? BW_LOOP_SMALL : LAT_LOOP_SMALL);
    printf("  -h            print this help message\n");
    fflush(stdout);
}

static int
set_num_iterations (int value)
{
    if (1 > value) {
        return -1;
    }

    options.loop = value;
    options.loop_large = value;

    return 0;
}

static int
set_num_warmup (int value)
{
    if (0 > value) {
        return -1;
    }

    options.skip = value;
    options.skip_large = value;

    return 0;
}

int
process_options (int argc, char *argv[], int type)
{   
    extern char * optarg;
    extern int optind;
    
    char const * optstring = (CUDA_ENABLED || OPENACC_ENABLED) ? "+d:x:i:h" : "+x:i:h";
    int c;
    
    /*
     * set default options
     */
      
    options.src = 'H';
    options.dst = 'H';

    benchmark_type = type;
    switch (type) {
        case BW:
            options.loop = BW_LOOP_SMALL;
            options.skip = BW_SKIP_SMALL;
            options.loop_large = BW_LOOP_LARGE;
            options.skip_large = BW_SKIP_LARGE;
            break;
        case LAT:
            options.loop = LAT_LOOP_SMALL;
            options.skip = LAT_SKIP_SMALL;
            options.loop_large = LAT_LOOP_LARGE;
            options.skip_large = LAT_SKIP_LARGE;
            break;
    }

    if (CUDA_ENABLED) { 
        options.accel = cuda;
    }
    
    else if (OPENACC_ENABLED) {
        options.accel = openacc;
    }
    
    else {
        options.accel = none;
    }
    
    while((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
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
            case 'i':
                if (set_num_iterations(atoi(optarg))) {
                    bad_usage.message = "Invalid Number of Iterations";
                    bad_usage.optarg = optarg;

                    return po_bad_usage;
                }
                break;
            case 'x':
                if (set_num_warmup(atoi(optarg))) {
                    bad_usage.message = "Invalid Number of Warmup Iterations";
                    bad_usage.optarg = optarg;

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
            options.src = argv[optind][0];
            options.dst = argv[optind + 1][0];
            
            switch (options.src) {
                case 'D':
                case 'H':
                case 'M':
                    break;
                default:
                    return po_bad_usage;
            }
            
            switch (options.dst) {
                case 'D':
                case 'H':
                case 'M':
                    break;
                default:
                    return po_bad_usage;
            }
        }
        
        else if (optind != argc) {
            return po_bad_usage;
        }
    }
    
    return po_okay;
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
allocate_managed_buffer (char ** buffer)
{
#ifdef _ENABLE_CUDA_
    cudaError_t cuerr = cudaSuccess;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            cuerr = cudaMallocManaged((void **)buffer, MYBUFSIZE, cudaMemAttachGlobal);

            if (cudaSuccess != cuerr) {
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
allocate_device_buffer (char ** buffer)
{
#ifdef _ENABLE_CUDA_
    cudaError_t cuerr = cudaSuccess;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            cuerr = cudaMalloc((void **)buffer, MYBUFSIZE);

            if (cudaSuccess != cuerr) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            *buffer = acc_malloc(MYBUFSIZE);
            if (NULL == *buffer) {
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
allocate_memory (char ** sbuf, char ** rbuf, int rank)
{
    unsigned long align_size = sysconf(_SC_PAGESIZE);

    switch (rank) {
        case 0:
            if ('D' == options.src) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            }

            else if ('M' == options.src) {
                if (allocate_managed_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }

                if (allocate_managed_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }
            }

            else {
                if (posix_memalign((void**)sbuf, align_size, MYBUFSIZE)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }

                if (posix_memalign((void**)rbuf, align_size, MYBUFSIZE)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }
            }
            break;
        case 1:
            if ('D' == options.dst) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            }

            else if ('M' == options.dst) {
                if (allocate_managed_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }

                if (allocate_managed_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda unified memory\n");
                    return 1;
                }
            }

            else {
                if (posix_memalign((void**)sbuf, align_size, MYBUFSIZE)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }

                if (posix_memalign((void**)rbuf, align_size, MYBUFSIZE)) {
                    fprintf(stderr, "Error allocating host memory\n");
                    return 1;
                }
            }
            break;
    }

    return 0;
}

void
print_header (int rank, int type)
{
    if (0 == rank) {
        switch (options.accel) {
            case cuda:
                printf(benchmark_header, "-CUDA");
                break;
            case openacc:
                printf(benchmark_header, "-OPENACC");
                break;
            default:
                printf(benchmark_header, "");
                break;
        }

        switch (options.accel) {
            case cuda:
            case openacc:
                printf("# Send Buffer on %s and Receive Buffer on %s\n",
                        'M' == options.src ? "MANAGED (M)" : ('D' == options.src ? "DEVICE (D)" : "HOST (H)"),
                        'M' == options.dst ? "MANAGED (M)" : ('D' == options.dst ? "DEVICE (D)" : "HOST (H)"));
            default:
                if (type == BW) {
                    printf("%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Bandwidth (MB/s)");
                } 
                else {
                    printf("%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
                }
                fflush(stdout);
        }
    }
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
#pragma acc parallel loop deviceptr(p)
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
touch_data (void * sbuf, void * rbuf, int rank, size_t size)
{
    if ((0 == rank && 'H' == options.src) ||
            (1 == rank && 'H' == options.dst)) {
        memset(sbuf, 'a', size);
        memset(rbuf, 'b', size);
    } else {
        set_device_memory(sbuf, 'a', size);
        set_device_memory(rbuf, 'b', size);
    }
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

void
free_memory (void * sbuf, void * rbuf, int rank)
{
    switch (rank) {
        case 0:
            if ('D' == options.src || 'M' == options.src) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            }

            else {
                free(sbuf);
                free(rbuf);
            }
            break;
        case 1:
            if ('D' == options.dst || 'M' == options.dst) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            }

            else {
                free(sbuf);
                free(rbuf);
            }
            break;
    }
}

/* vi:set sw=4 sts=4 tw=80: */

