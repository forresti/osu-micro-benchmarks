#define BENCHMARK "OSU MPI Multi-threaded Latency Test"
/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <osu_pt2pt.h>
#include <pthread.h>

#define THREADS 2

pthread_mutex_t finished_size_mutex;
pthread_cond_t  finished_size_cond;

int finished_size;

typedef struct thread_tag  {
        int id;
} thread_tag_t;

void * send_thread(void *arg);
void * recv_thread(void *arg);

int main(int argc, char *argv[])
{
    int numprocs, provided, myid, err;
    int i = 0;
    pthread_t sr_threads[THREADS];
    thread_tag_t tags[THREADS];

    pthread_mutex_init(&finished_size_mutex, NULL);
    pthread_cond_init(&finished_size_cond, NULL);

    set_header(HEADER);

    int po_ret = process_options(argc, argv, LAT);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if(err != MPI_SUCCESS) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (0 == myid) {
        switch (po_ret) {
            case po_cuda_not_avail:
                fprintf(stderr, "CUDA support not available.\n");
                break;
            case po_openacc_not_avail:
                fprintf(stderr, "OPENACC support not available.\n");
                break;
            case po_bad_usage:
            case po_help_message:
                usage("osu_latency_mt");
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

    if(numprocs != 2) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_Finalize();

        return EXIT_FAILURE;
    }

    /* Check to make sure we actually have a thread-safe
     * implementation 
     */

    finished_size = 1;

    if(provided != MPI_THREAD_MULTIPLE) {
        if(myid == 0) {
            fprintf(stderr,
                "MPI_Init_thread must return MPI_THREAD_MULTIPLE!\n");
        }

        MPI_Finalize();

        return EXIT_FAILURE;
    }

    if(myid == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
        fflush(stdout);

        tags[i].id = i;
        pthread_create(&sr_threads[i], NULL, send_thread, &tags[i]);
        pthread_join(sr_threads[i], NULL);

    }

    else {
        for(i = 0; i < THREADS; i++) {
            tags[i].id = i;
            pthread_create(&sr_threads[i], NULL, recv_thread, &tags[i]);
        }

        for(i = 0; i < THREADS; i++) {
            pthread_join(sr_threads[i], NULL);
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

void * recv_thread(void *arg) {
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    int size, i, val;
    int iter;
    char * ret = NULL;
    char *s_buf, *r_buf;
    thread_tag_t *thread_id;

    thread_id = (thread_tag_t *)arg;
    val = thread_id->id;

    if (posix_memalign((void**)&s_buf, align_size, MYBUFSIZE)) {
        fprintf(stderr, "Error allocating host memory\n");
        *ret = '1';
        return ret;
    }

    if (posix_memalign((void**)&r_buf, align_size, MYBUFSIZE)) {
        fprintf(stderr, "Error allocating host memory\n");
        *ret = '1';
        return ret;
    }

    for(size = 0, iter = 0; size <= MAX_MSG_SIZE; size = (size ? size * 2 : 1)) {
        pthread_mutex_lock(&finished_size_mutex);

        if(finished_size == THREADS) {
            MPI_Barrier(MPI_COMM_WORLD);

            finished_size = 1;

            pthread_mutex_unlock(&finished_size_mutex);
            pthread_cond_broadcast(&finished_size_cond);
        }

        else {
            finished_size++;

            pthread_cond_wait(&finished_size_cond, &finished_size_mutex);
            pthread_mutex_unlock(&finished_size_mutex);
        }

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = options.loop_large;
            options.skip = options.skip_large;
        }  

        /* touch the data */
        for(i = 0; i < size; i++) {
            s_buf[i] = 'a';
            r_buf[i] = 'b';
        }

        for(i = val; i < (options.loop + options.skip); i += THREADS) {
            MPI_Recv (r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD,
                    &reqstat[val]);
            MPI_Send (s_buf, size, MPI_CHAR, 0, 2, MPI_COMM_WORLD);
        }

        iter++;
    }

    free(r_buf);
    free(s_buf);

    sleep(1);

    return 0;
}


void * send_thread(void *arg) {
    unsigned long align_size = sysconf(_SC_PAGESIZE);
    int size, i, val, iter;
    char *s_buf, *r_buf;
    double t_start = 0, t_end = 0, t = 0, latency;
    thread_tag_t *thread_id = (thread_tag_t *)arg;
    char *ret = NULL;

    val = thread_id->id;

    if (posix_memalign((void**)&s_buf, align_size, MYBUFSIZE)) {
        fprintf(stderr, "Error allocating host memory\n");
        *ret = '1';
        return ret;
    }

    if (posix_memalign((void**)&r_buf, align_size, MYBUFSIZE)) {
        fprintf(stderr, "Error allocating host memory\n");
        *ret = '1';
        return ret;
    }

    for(size = 0, iter = 0; size <= MAX_MSG_SIZE; size = (size ? size * 2 : 1)) {
        MPI_Barrier(MPI_COMM_WORLD);

        if(size > LARGE_MESSAGE_SIZE) {
            options.loop = options.loop_large;
            options.skip = options.skip_large;
        }  

        /* touch the data */
        for(i = 0; i < size; i++) {
            s_buf[i] = 'a';
            r_buf[i] = 'b';
        }

        for(i = 0; i < options.loop + options.skip; i++) {
            if(i == options.skip) {
                t_start = MPI_Wtime();
            }

            MPI_Send(s_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
            MPI_Recv(r_buf, size, MPI_CHAR, 1, 2, MPI_COMM_WORLD,
                    &reqstat[val]);
        }

        t_end = MPI_Wtime ();
        t = t_end - t_start;

        latency = (t) * 1.0e6 / (2.0 * options.loop);
        fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH, FLOAT_PRECISION,
                latency);
        fflush(stdout);
        iter++;
    }

    free(r_buf);
    free(s_buf);

    return 0;
}

/* vi: set sw=4 sts=4 tw=80: */
