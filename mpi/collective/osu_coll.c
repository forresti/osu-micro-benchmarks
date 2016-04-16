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
#include "osu_coll.h"

#ifdef _ENABLE_OPENACC_
#include <openacc.h>
#endif

/*
 * GLOBAL VARIABLES
 */
#ifdef _ENABLE_CUDA_
static CUcontext cuContext;
#endif

static char const * benchmark_header = NULL;
static char const * benchmark_name = NULL;
static int accel_enabled = 0;
static int kernel_count = 0;
struct options_t options;

/* A is the A in DAXPY for the Compute Kernel */
#define A 2.0
#define DEBUG 0
/* 
 * We are using a 2-D matrix to perform dummy
 * computation in non-blocking collective benchmarks 
 */
#define DIM 25
static float **a, *x, *y;

#ifdef _ENABLE_CUDA_KERNEL_
/* Using new stream for kernels on gpu */
static cudaStream_t stream;

/* Arrays on device for dummy compute */
static float *d_x, *d_y;
#endif

static struct {
    char const * message;
    char const * optarg;
    int opt;
} bad_usage;

static int
set_min_message_size (int value)
{
    int size = 1;
    if (0 >= value) {
        return -1;
    }

    while (size < value) {
        size *= 2;
    }

    options.min_message_size = size;

    return 0;
}

static int
set_max_message_size (int value)
{
    if (0 > value) {
        return -1;
    }

    options.max_message_size = value;

    return 0;
}

static int
set_message_size (char *val_str)
{
    int retval = -1;
    int i, count = 0;
    char *val1, *val2;

    for (i=0; val_str[i]; i++) {
        if (val_str[i] == ':')
            count++;
    }

    if (!count) {
        retval = set_max_message_size(atoi(val_str));
    } else if (count == 1) {
        val1 = strtok(val_str, ":");
        val2 = strtok(NULL, ":");

        if (val1 && val2) {
            retval = set_min_message_size(atoi(val1));
            retval = set_max_message_size(atoi(val2));
        } else if (val1) {
            if (val_str[0] == ':') {
                retval = set_max_message_size(atoi(val1));
            } else {
                retval = set_min_message_size(atoi(val1));
            }
        }
    }

    return retval;
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

static int
set_num_iterations (int value)
{
    if (1 > value) {
        return -1;
    }

    options.iterations = value;
    options.iterations_large = value;

    return 0;
}

static int
set_device_array_size (int value)
{
    if (value < 1 ) {
        return -1;
    }

    options.device_array_size = value;

    return 0;
}

static int
set_num_probes (int value)
{
    if (value < 0 ) {
        return -1;
    }

    options.num_probes = value;

    return 0;
}

static int
set_max_memlimit (int value)
{
    options.max_mem_limit = value;

    if (value < MAX_MEM_LOWER_LIMIT) {
        options.max_mem_limit = MAX_MEM_LOWER_LIMIT; 
        fprintf(stderr,"Requested memory limit too low, using [%d] instead.",
                MAX_MEM_LOWER_LIMIT); 
    }

    return 0;
}

void
set_header (const char * header)
{
    benchmark_header = header;
}

void
set_benchmark_name (const char * name)
{
    benchmark_name = name;
}

void
enable_accel_support (void)
{
    accel_enabled = (CUDA_ENABLED || OPENACC_ENABLED);
}

enum po_ret_type
process_options (int argc, char *argv[])
{
    extern char * optarg;
    extern int optind, optopt;

    char const * optstring = "+:hvfm:i:x:M:t:s:";
    int c;

    if (accel_enabled) {
        optstring = (CUDA_KERNEL_ENABLED) ? "+:d:hvfm:i:x:M:t:r:s:"
            : "+:d:hvfm:i:x:M:t:s:";
    }

    /*
     * SET DEFAULT OPTIONS
     */
    options.accel = none;
    options.show_size = 1;
    options.show_full = 0;
    options.num_probes = 0;
    options.device_array_size = 32; 
    options.target = cpu;
    options.min_message_size = DEFAULT_MIN_MESSAGE_SIZE;
    options.max_message_size = DEFAULT_MAX_MESSAGE_SIZE;
    options.max_mem_limit = MAX_MEM_LIMIT;
    options.iterations = 1000;
    options.iterations_large = 100;
    options.skip = 200;
    options.skip_large = 10;

    while ((c = getopt(argc, argv, optstring)) != -1) {
        bad_usage.opt = c;
        bad_usage.optarg = NULL;
        bad_usage.message = NULL;

        switch (c) {
            case 'h':
                return po_help_message;
            case 'v':
                return po_version_message;
            case 'm':
                if (set_message_size(optarg)) {
                    bad_usage.message = "Invalid Message Size";
                    bad_usage.optarg = optarg;

                    return po_bad_usage;
                }
                break;
            case 't':
                if (set_num_probes(atoi(optarg))){
                    bad_usage.message = "Invalid Number of Probes";
                    bad_usage.optarg = optarg;

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
            case 's':
                if (set_device_array_size(atoi(optarg))){
                    bad_usage.message = "Invalid Device Array Size";
                    bad_usage.optarg = optarg;
                    
                    return po_bad_usage;
                }
                break;
            case 'f':
                options.show_full = 1;
                break;
            case 'M': 
                /*
                 * This function does not error but prints a warning message if
                 * the value is too low.
                 */
                set_max_memlimit(atoll(optarg));
                break; 
            case 'd':
                if (!accel_enabled) {
                    bad_usage.message = "Benchmark Does Not Support "
                        "Accelerator Transfers";
                    bad_usage.optarg = optarg;
                    return po_bad_usage;
                }
                else if (0 == strncasecmp(optarg, "cuda", 10)) {
                    if (CUDA_ENABLED) {
                        options.accel = cuda;
                    }
                    else {
                        bad_usage.message = "CUDA Support Not Enabled\n"
                            "Please recompile benchmark with CUDA support";
                        bad_usage.optarg = optarg;
                        return po_bad_usage;
                    }
                }
                else if (0 == strncasecmp(optarg, "managed", 10)) {
                    if (CUDA_ENABLED) {
                        options.accel = managed;
                    }
                    else {
                        bad_usage.message = "CUDA Managed Memory Support Not Enabled\n"
                            "Please recompile benchmark with CUDA support";
                        bad_usage.optarg = optarg;
                        return po_bad_usage;
                    }
                }
                else if (0 == strncasecmp(optarg, "openacc", 10)) {
                    if (OPENACC_ENABLED) {
                        options.accel = openacc;
                    }
                    else {
                        bad_usage.message = "OpenACC Support Not Enabled\n"
                            "Please recompile benchmark with OpenACC support";
                        bad_usage.optarg = optarg;
                        return po_bad_usage;
                    }
                }
                else {
                    bad_usage.message = "Invalid Accel Type Specified";
                    bad_usage.optarg = optarg;
                    return po_bad_usage;
                }
                break;
            case 'r':
                if (CUDA_KERNEL_ENABLED) {
                    if (0 == strncasecmp(optarg, "cpu", 10)) {
                        options.target = cpu;
                    } else if (0 == strncasecmp(optarg, "gpu", 10)) {
                        options.target = gpu;
                    } else if (0 == strncasecmp(optarg, "both", 10)) {
                        options.target = both;
                    } else {
                        bad_usage.message = "Please use cpu, gpu, or both";
                        bad_usage.optarg = optarg;
                        return po_bad_usage;
                    }
                } else {
                    bad_usage.message = "CUDA Kernel Support Not Enabled\n"
                        "Please recompile benchmark with CUDA Kernel support";
                    bad_usage.optarg = optarg;
                    return po_bad_usage;
                }
                 break;
            case ':':
                bad_usage.message = "Option Missing Required Argument";
                bad_usage.opt = optopt;
                return po_bad_usage;
            default:
                bad_usage.message = "Invalid Option";
                bad_usage.opt = optopt;
                return po_bad_usage;
        }
    }

    return po_okay;
}

void
print_bad_usage_message (int rank)
{
    if (rank) return;

    if (bad_usage.optarg) {
        fprintf(stderr, "%s [-%c %s]\n\n", bad_usage.message,
                (char)bad_usage.opt, bad_usage.optarg);
    }

    else {
        fprintf(stderr, "%s [-%c]\n\n", bad_usage.message,
                (char)bad_usage.opt);
    }

    print_help_message(rank);
}

void
print_help_message (int rank)
{
    if (rank) return;

    printf("Usage: %s [options]\n", benchmark_name);
    printf("options:\n");

    if (accel_enabled) {
        printf("  -d TYPE       use accelerator device buffers which can be of TYPE `cuda' or\n");
        printf("                use accelerator managed device buffers which can be of TYPE `managed' or\n");
        printf("                `openacc' (uses standard host buffers if not specified)\n");
    }

    if (options.show_size) {
        printf("  -m [MIN:]MAX  set the minimum and/or the maximum message size to MIN and/or MAX\n"
               "                bytes respectively. Examples:\n"
               "                -m 128      // min = default, max = 128\n"
               "                -m 2:128    // min = 2, max = 128\n"
               "                -m 2:       // min = 2, max = default\n");
        printf("  -M SIZE       set per process maximum memory consumption to SIZE bytes\n");
        printf("                (default %d)\n", MAX_MEM_LIMIT); 
    }

    printf("  -i ITER       set iterations per message size to ITER (default 1000 for small\n");
    printf("                messages, 100 for large messages)\n");
    printf("  -x ITER       set number of warmup iterations to skip before timing (default 200)\n");

    printf("  -f            print full format listing (MIN/MAX latency and ITERATIONS\n");
    printf("                displayed in addition to AVERAGE latency)\n");
    
    printf("  -t CALLS      set the number of MPI_Test() calls during the dummy computation, \n");
    printf("                set CALLS to 100, 1000, or any number > 0.\n");

    if (CUDA_KERNEL_ENABLED) {
        printf("  -r TARGET     set the compute target for dummy computation\n");
        printf("                set TARGET to cpu (default) to execute \n");
        printf("                on CPU only, set to gpu for executing kernel \n");
        printf("                on the GPU only, and set to both for compute on both.\n");

        printf("  -s SIZE       set the size of arrays to be allocated on device (GPU) \n");
        printf("                for dummy compute on device (GPU) (default 32) \n");    
    }

    printf("  -h            print this help\n");
    printf("  -v            print version info\n");
    printf("\n");
    fflush(stdout);
}

void
print_version_message (int rank)
{
    if (rank) return;

    switch (options.accel) {
        case cuda:
            printf(benchmark_header, "-CUDA");
            break;
        case openacc:
            printf(benchmark_header, "-OPENACC");
            break;
        case managed:
            printf(benchmark_header, "-CUDA MANAGED");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }
 
    fflush(stdout);
}

void 
print_preamble_nbc (int rank) 
{
    if (rank) return;
    
    printf("\n");
    
    switch (options.accel) {
        case cuda:
            printf(benchmark_header, "-CUDA");
            break;
        case openacc:
            printf(benchmark_header, "-OPENACC");
            break;
        case managed:
            printf(benchmark_header, "-MANAGED");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    fprintf(stdout, "# Overall = Coll. Init + Compute + MPI_Test + MPI_Wait\n\n");

    if (options.show_size) {
        fprintf(stdout, "%-*s", 10, "# Size");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Overall(us)");
    }
    else {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Overall(us)");
    }    

    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Coll. Init(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Test(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Wait(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");

    }
    else {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");
    }

    fflush(stdout);
}

void
display_nbc_params()
{
    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Coll. Init(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Test(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "MPI_Wait(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");

    }
    else {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Compute(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Pure Comm.(us)");
        fprintf(stdout, "%*s\n", FIELD_WIDTH, "Overlap(%)");
    }
}

void
print_preamble (int rank)
{
    if (rank) return;

    printf("\n");

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
 
    if (options.show_size) {
        fprintf(stdout, "%-*s", 10, "# Size");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Latency(us)");
    }

    else {
        fprintf(stdout, "# Avg Latency(us)");
    }

    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Min Latency(us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Max Latency(us)");
        fprintf(stdout, "%*s\n", 12, "Iterations");
    }

    else {
        fprintf(stdout, "\n");
    }

    fflush(stdout);
}

void
calculate_and_print_stats(int rank, int size, int numprocs,
                          double timer, double latency,
                          double test_time, double cpu_time, 
                          double wait_time, double init_time)
{
        double test_total   = (test_time * 1e6) / options.iterations;
        double tcomp_total  = (cpu_time * 1e6) / options.iterations;
        double overall_time = (timer * 1e6) / options.iterations;
        double wait_total   = (wait_time * 1e6) / options.iterations;
        double init_total   = (init_time * 1e6) / options.iterations;
        double comm_time   = latency;

        if(rank != 0) {
            MPI_Reduce(&test_total, &test_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(&comm_time, &comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(&overall_time, &overall_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(&tcomp_total, &tcomp_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(&wait_total, &wait_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(&init_total, &init_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(MPI_IN_PLACE, &test_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE, &comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE, &overall_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE, &tcomp_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE, &wait_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE, &init_total, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        /* Overall Time (Overlapped) */
        overall_time = overall_time/numprocs;
        /* Computation Time */
        tcomp_total = tcomp_total/numprocs;
        /* Time taken by MPI_Test calls */
        test_total = test_total/numprocs;
        /* Pure Communication Time */
        comm_time = comm_time/numprocs;
        /* Time for MPI_Wait() call */
        wait_total = wait_total/numprocs;
        /* Time for the NBC call */
        init_total = init_total/numprocs;

        print_stats_nbc(rank, size, overall_time, tcomp_total, comm_time,
                                    wait_total, init_total, test_total);

}

void 
print_stats_nbc (int rank, int size, double overall_time, 
                      double cpu_time, double comm_time,
                      double wait_time, double init_time,
		              double test_time)
{
    if (rank) return;

    double overlap;
 
    /* Note : cpu_time received in this function includes time for
       *      dummy compute as well as test calls so we will subtract
       *      the test_time for overlap calculation as test is an
       *      overhead
       */

    overlap = max(0, 100 - (((overall_time - (cpu_time - test_time)) / comm_time) * 100)); 
    
    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, overall_time);
    }
    else {
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, overall_time);
    }

    if (options.show_full) {
           fprintf(stdout, "%*.*f%*.*f%*.*f%*.*f%*.*f%*.*f\n",
                FIELD_WIDTH, FLOAT_PRECISION, (cpu_time - test_time),
                FIELD_WIDTH, FLOAT_PRECISION, init_time,
                FIELD_WIDTH, FLOAT_PRECISION, test_time, 
                FIELD_WIDTH, FLOAT_PRECISION, wait_time,
                FIELD_WIDTH, FLOAT_PRECISION, comm_time,
                FIELD_WIDTH, FLOAT_PRECISION, overlap);
    }    
    else {
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, (cpu_time - test_time));
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, comm_time);
        fprintf(stdout, "%*.*f\n", FIELD_WIDTH, FLOAT_PRECISION, overlap);
    }

    fflush(stdout);
}

void
print_stats (int rank, int size, double avg_time, double min_time, double
        max_time)
{
    if (rank) return;

    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
    }

    else {
        fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
    }

    if (options.show_full) {
        fprintf(stdout, "%*.*f%*.*f%*lu\n", 
                FIELD_WIDTH, FLOAT_PRECISION, min_time,
                FIELD_WIDTH, FLOAT_PRECISION, max_time,
                12, options.iterations);
    }

    else {
        fprintf(stdout, "\n");
    }

    fflush(stdout);
}

void
set_buffer (void * buffer, enum accel_type type, int data, size_t size)
{
#ifdef _ENABLE_OPENACC_
    size_t i;
    char * p = (char *)buffer;
#endif
    int a;
    switch (type) {
        case none:
            memset(buffer, data, size);
            break;
        case cuda:
        case managed:
#ifdef _ENABLE_CUDA_
            cudaMemset(buffer, data, size);
#endif
            break;
        case openacc:
#ifdef _ENABLE_OPENACC_
#pragma acc parallel loop deviceptr(p)
            for (i = 0; i < size; i++) {
                p[i] = data;
            }
#endif
            break;
    }
}

int
allocate_buffer (void ** buffer, size_t size, enum accel_type type)
{
    size_t alignment = sysconf(_SC_PAGESIZE);
#ifdef _ENABLE_CUDA_
    cudaError_t cuerr = cudaSuccess;
#endif

    switch (type) {
        case none:
            return posix_memalign(buffer, alignment, size);
#ifdef _ENABLE_CUDA_
        case cuda:
            cuerr = cudaMalloc(buffer, size);
            if (cudaSuccess != cuerr) {
                return 1;
            }

            else {
                return 0;
            }
        case managed:
            cuerr = cudaMallocManaged(buffer, size, cudaMemAttachGlobal);
            if (cudaSuccess != cuerr) {
                return 1;
            }

            else {
                return 0;
            }
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            *buffer = acc_malloc(size);
            if (NULL == *buffer) {
                return 1;
            }

            else {
                return 0;
            }
#endif
        default:
            return 1;
    }
}

void
free_buffer (void * buffer, enum accel_type type)
{
    switch (type) {
        case none:
            free(buffer);
            break;
        case managed:
        case cuda:
#ifdef _ENABLE_CUDA_
            cudaFree(buffer);
#endif
            break;
        case openacc:
#ifdef _ENABLE_OPENACC_
            acc_free(buffer);
#endif
            break;
    }
    
    /* Free dummy compute related resources */
    if (is_alloc) {
        if (options.target == cpu) {
            free_host_arrays();
        } 
#ifdef _ENABLE_CUDA_KERNEL_ 
        else if (options.target == gpu || options.target == both) {
            free_host_arrays();
            free_device_arrays();
        }
#endif
    }

    is_alloc = 0;
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
        case managed:
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
cleanup_accel (void)
{
#ifdef _ENABLE_CUDA_
    CUresult curesult = CUDA_SUCCESS;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case managed:
        case cuda:
            curesult = cuCtxDestroy(cuContext);

            if (curesult != CUDA_SUCCESS) {
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            acc_shutdown(acc_device_nvidia);
            break;
#endif
        default:
            fprintf(stderr, "Invalid accel type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

#ifdef _ENABLE_CUDA_KERNEL_
void
free_device_arrays()
{
    cudaError_t cuerr = cudaSuccess;
    cuerr = cudaFree(d_x);
    if (cuerr != cudaSuccess)
        fprintf(stderr, "Failed to free device array\n");
    
    cuerr = cudaFree(d_y);
    if (cuerr != cudaSuccess)
        fprintf(stderr, "Failed to free device array\n");
}
#endif

void 
free_host_arrays()
{
    free(x);
    free(y);
    free(a);
}

double
dummy_compute(double seconds, MPI_Request* request)
{
    double test_time = 0.0;

    test_time = do_compute_and_probe(seconds, request);

    return test_time;
}

#ifdef _ENABLE_CUDA_KERNEL_
void
do_compute_gpu(double seconds)
{
    int i,j;
    double time_elapsed = 0.0, t1 = 0.0, t2 = 0.0;

    {
        t1 = MPI_Wtime();

        /* Execute Dummy Kernel on GPU if set by user */
        if (options.target == both || options.target == gpu) {
            {
                cudaStreamCreate(&stream);
                call_kernel(A, d_x, d_y, options.device_array_size, &stream);
            }
        }

        t2 = MPI_Wtime();
        time_elapsed += (t2-t1);
    }
}
#endif

void
compute_on_host()
{
    int i = 0, j = 0;
    for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++)
            x[i] = x[i] + a[i][j]*a[j][i] + y[j];
}


static inline void 
do_compute_cpu(double target_seconds)
{
    double t1 = 0.0, t2 = 0.0;
    double time_elapsed = 0.0;
    while (time_elapsed < target_seconds) {
        t1 = MPI_Wtime();
        compute_on_host();
        t2 = MPI_Wtime();
        time_elapsed += (t2-t1);
    }
    if (DEBUG) fprintf(stderr, "time elapsed = %f\n", (time_elapsed * 1e6));
}

double
do_compute_and_probe(double seconds, MPI_Request* request)
{
    double t1 = 0.0, t2 = 0.0;
    double test_time = 0.0;
    int num_tests = 0;
    double target_seconds_for_compute = 0.0;
    int flag = 0;
    MPI_Status status;

    if (options.num_probes) {
        target_seconds_for_compute = (double) seconds/options.num_probes;
        if (DEBUG) fprintf(stderr, "setting target seconds to %f\n", (target_seconds_for_compute * 1e6 ));
    } 
    else {
        target_seconds_for_compute = seconds;
        if (DEBUG) fprintf(stderr, "setting target seconds to %f\n", (target_seconds_for_compute * 1e6 ));
    }

#ifdef _ENABLE_CUDA_KERNEL_
    if (options.target == gpu) {
        if (options.num_probes) {
            /* Do the dummy compute on GPU only */
            do_compute_gpu(target_seconds_for_compute);
            num_tests = 0;
            while (num_tests < options.num_probes) {
                t1 = MPI_Wtime();
                MPI_Test(request, &flag, &status);
                t2 = MPI_Wtime();
                test_time += (t2-t1);
                num_tests++;
            }
        }
        else {
            do_compute_gpu(target_seconds_for_compute);
        }
    }
    else if (options.target == both) {
        if (options.num_probes) {
            /* Do the dummy compute on GPU and CPU*/
            do_compute_gpu(target_seconds_for_compute);
            num_tests = 0;
            while (num_tests < options.num_probes) {
                t1 = MPI_Wtime();
                MPI_Test(request, &flag, &status);
                t2 = MPI_Wtime();
                test_time += (t2-t1);
                num_tests++;
                do_compute_cpu(target_seconds_for_compute);
            }
        } 
        else {
            do_compute_gpu(target_seconds_for_compute);
            do_compute_cpu(target_seconds_for_compute);
        }        
    }
    else
#endif
    if (options.target == cpu) {
        if (options.num_probes) {
            num_tests = 0;
            while (num_tests < options.num_probes) {
                do_compute_cpu(target_seconds_for_compute);
                t1 = MPI_Wtime();
                MPI_Test(request, &flag, &status);
                t2 = MPI_Wtime();
                test_time += (t2-t1);
                num_tests++;
            }
        }
        else {
            do_compute_cpu(target_seconds_for_compute);
        }
    }

#ifdef _ENABLE_CUDA_KERNEL_
    if (options.target == gpu || options.target == both) {
        cudaDeviceSynchronize();    
        cudaStreamDestroy(stream);
    }
#endif
    
    return test_time;
}

void 
init_arrays(double target_time) 
{
    
    if (DEBUG) fprintf(stderr, "called init_arrays with target_time = %f \n", (target_time * 1e6));
    int i = 0, j = 0;
    
    a = (float **)malloc(DIM * sizeof(float *));
    
    for (i = 0; i < DIM; i++) {
        a[i] = (float *)malloc(DIM * sizeof(float));
    }
    
    x = (float *)malloc(DIM * sizeof(float));
    y = (float *)malloc(DIM * sizeof(float));

    for (i = 0; i < DIM; i++) {
        x[i] = y[i] = 1.0f;
        for (j = 0; j < DIM; j++) {
            a[i][j] = 2.0f;
        }
    }

#ifdef _ENABLE_CUDA_KERNEL_
    if (options.target == gpu || options.target == both) {
    /* Setting size of arrays for Dummy Compute */
    int N = options.device_array_size;

    /* Device Arrays for Dummy Compute */
    allocate_device_arrays(N);
    
    double time_elapsed = 0.0;
    double t1 = 0.0, t2 = 0.0;
    
    while (1) {
        t1 = MPI_Wtime();
        
        if (options.target == gpu || options.target == both) {
            cudaStreamCreate(&stream);
            call_kernel(A, d_x, d_y, N, &stream);
            
            cudaDeviceSynchronize();
            cudaStreamDestroy(stream);
        }

        t2 = MPI_Wtime();
        if ((t2-t1) < target_time)
        {  
            N += 32;

            /* First free the old arrays */
            free_device_arrays();

            /* Now allocate arrays of size N */
            allocate_device_arrays(N);
        }
        else {
            break;
        }
    }
    
    /* we reach here with desired N so save it and pass it to options */
    options.device_array_size = N;
    if (DEBUG) fprintf(stderr, "correct N = %d\n", N);
    }
#endif

}
#ifdef _ENABLE_CUDA_KENEL_
void
allocate_device_arrays(int n)
{
    cudaError_t cuerr = cudaSuccess;
    
    /* Allocate Device Arrays for Dummy Compute */
    cuerr = cudaMalloc((void**)&d_x, n * sizeof(float));
    if (cuerr != cudaSuccess)
        fprintf(stderr, "Failed to free device array");
    
    cuerr = cudaMalloc((void**)&d_y, n * sizeof(float));
    if (cuerr != cudaSuccess)
        fprintf(stderr, "Failed to free device array");

    cudaMemset(d_x, 1.0f, n);
    cudaMemset(d_y, 2.0f, n);
}
#endif
/* vi:set sw=4 sts=4 tw=80: */
