NVCC = nvcc
NVCFLAGS = -cuda -maxrregcount 32
SUFFIXES = .cu .cpp
.cu.cpp:
	$(NVCC) $(NVCFLAGS) $(INCLUDES) $(CPPFLAGS) --output-file $@ $<

collectivedir = $(pkglibexecdir)/mpi/collective
collective_PROGRAMS = osu_alltoallv osu_allgatherv osu_scatterv osu_gatherv osu_reduce_scatter osu_barrier osu_reduce osu_allreduce osu_alltoall osu_bcast osu_gather osu_allgather osu_scatter osu_iallgather osu_ibcast  osu_ialltoall osu_ibarrier osu_igather osu_iscatter osu_iscatterv osu_igatherv osu_iallgatherv osu_ialltoallv osu_ialltoallw

osu_alltoallv_SOURCES = osu_alltoallv.c osu_coll.c osu_coll.h
osu_allgatherv_SOURCES = osu_allgatherv.c osu_coll.c osu_coll.h
osu_scatterv_SOURCES = osu_scatterv.c osu_coll.c osu_coll.h
osu_gather_SOURCES = osu_gather.c osu_coll.c osu_coll.h
osu_gatherv_SOURCES = osu_gatherv.c osu_coll.c osu_coll.h
osu_reduce_scatter_SOURCES = osu_reduce_scatter.c osu_coll.c osu_coll.h  
osu_barrier_SOURCES = osu_barrier.c osu_coll.c osu_coll.h  
osu_reduce_SOURCES = osu_reduce.c osu_coll.c osu_coll.h  
osu_allreduce_SOURCES = osu_allreduce.c osu_coll.c osu_coll.h  
osu_bcast_SOURCES = osu_bcast.c osu_coll.c osu_coll.h  
osu_alltoall_SOURCES = osu_alltoall.c osu_coll.c osu_coll.h  
osu_ialltoall_SOURCES = osu_ialltoall.c osu_coll.c osu_coll.h  
osu_ialltoallv_SOURCES = osu_ialltoallv.c osu_coll.c osu_coll.h  
osu_ialltoallw_SOURCES = osu_ialltoallw.c osu_coll.c osu_coll.h  
osu_ibarrier_SOURCES = osu_ibarrier.c osu_coll.c osu_coll.h  
osu_ibcast_SOURCES = osu_ibcast.c osu_coll.c osu_coll.h  
osu_igather_SOURCES = osu_igather.c osu_coll.c osu_coll.h  
osu_igatherv_SOURCES = osu_igatherv.c osu_coll.c osu_coll.h  
osu_allgather_SOURCES = osu_allgather.c osu_coll.c osu_coll.h  
osu_iallgather_SOURCES = osu_iallgather.c osu_coll.c osu_coll.h  
osu_iallgatherv_SOURCES = osu_iallgatherv.c osu_coll.c osu_coll.h  
osu_scatter_SOURCES = osu_scatter.c osu_coll.c osu_coll.h  
osu_iscatter_SOURCES = osu_iscatter.c osu_coll.c osu_coll.h  
osu_iscatterv_SOURCES = osu_iscatterv.c osu_coll.c osu_coll.h  

if CUDA_KERNELS
osu_alltoall_SOURCES += kernel.cu
osu_alltoallv_SOURCES += kernel.cu
osu_allgather_SOURCES += kernel.cu
osu_allgatherv_SOURCES += kernel.cu
osu_barrier_SOURCES += kernel.cu
osu_bcast_SOURCES += kernel.cu
osu_scatter_SOURCES += kernel.cu
osu_scatterv_SOURCES += kernel.cu
osu_gather_SOURCES += kernel.cu
osu_gatherv_SOURCES += kernel.cu
osu_allreduce_SOURCES += kernel.cu
osu_reduce_SOURCES += kernel.cu
osu_reduce_scatter_SOURCES += kernel.cu
osu_ialltoall_SOURCES += kernel.cu
osu_ialltoallv_SOURCES += kernel.cu
osu_ialltoallw_SOURCES += kernel.cu
osu_iallgather_SOURCES += kernel.cu
osu_iallgatherv_SOURCES += kernel.cu
osu_ibarrier_SOURCES += kernel.cu
osu_ibcast_SOURCES += kernel.cu
osu_iscatter_SOURCES += kernel.cu
osu_iscatterv_SOURCES += kernel.cu
osu_igather_SOURCES += kernel.cu
osu_igatherv_SOURCES += kernel.cu
endif

if EMBEDDED_BUILD
    AM_LDFLAGS =
    AM_CPPFLAGS = -I$(top_builddir)/../src/include \
		  -I${top_srcdir}/../src/include
if BUILD_PROFILING_LIB
    AM_LDFLAGS += $(top_builddir)/../lib/lib@PMPILIBNAME@.la
endif
    AM_LDFLAGS += $(top_builddir)/../lib/lib@MPILIBNAME@.la
endif

if OPENACC
    AM_CFLAGS = -acc
endif
