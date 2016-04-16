#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.59])
AC_INIT([OSU-Micro-Benchmarks], [5.3], [mvapich-discuss@cse.ohio-state.edu])
AC_CONFIG_SRCDIR([mpi/pt2pt/osu_latency.c])

AM_INIT_AUTOMAKE([foreign])
LT_INIT

AC_ARG_ENABLE([openacc],
              [AS_HELP_STRING([--enable-openacc],
                              [Enable OpenACC benchmarks])
              ],
              [],
              [enable_openacc=no])

AC_ARG_ENABLE([cuda],
              [AS_HELP_STRING([--enable-cuda],
			      [Enable CUDA benchmarks (default is no).  Specify
			       --enable-cuda=basic to enable basic cuda support
			       without using cuda kernel support for
			       non-blocking collectives])
              ],
              [],
              [enable_cuda=no])

AC_ARG_WITH([cuda],
            [AS_HELP_STRING([--with-cuda=@<:@CUDA installation path@:>@],
                            [Provide path to CUDA installation])
            ],
            [AS_CASE([$with_cuda],
                     [yes|no], [],
                     [CPPFLAGS="-I$with_cuda/include $CPPFLAGS"
                      LDFLAGS="-L$with_cuda/lib64 -Wl,-rpath=$with_cuda/lib64 -L$with_cuda/lib -Wl,-rpath=$with_cuda/lib $LDFLAGS"])
            ])

AC_ARG_WITH([cuda-include],
            [AS_HELP_STRING([--with-cuda-include=@<:@CUDA include path@>:@],
                            [Provide path to CUDA include files])
            ],
            [AS_CASE([$with_cuda_include],
                     [yes|no], [],
                     [CPPFLAGS="-I$with_cuda_include $CPPFLAGS"])
            ])

AC_ARG_WITH([cuda-libpath],
            [AS_HELP_STRING([--with-cuda-libpath=@<:@CUDA library path@>:@],
                            [Provide path to CUDA library files])
            ],
            [AS_CASE([$with_cuda_libpath],
                     [yes|no], [],
                     [LDFLAGS="-L$with_cuda_libpath -Wl,-rpath=$with_cuda_libpath $LDFLAGS"])
            ])

# Checks for programs.
AC_PROG_CC([mpicc oshcc upcc upc++])

# Checks for mpicxx used for compiling kernel.cu in nbc benchmarks and/or the
# upc++ compiler for upcxx benchmarks
AC_PROG_CXX([mpicxx upc++])

# Checks for libraries.
AC_SEARCH_LIBS([sqrt], [m])
AC_SEARCH_LIBS([pthread_create], [pthread])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE

# Checks for library functions.
AC_CHECK_FUNCS([getpagesize gettimeofday memset sqrt])

AS_IF([test "x$enable_embedded" = xyes], [
       AS_IF([test x"$enable_mpi3" = xyes], [mpi3_library=true])
       AS_IF([test x"$enable_mpi2" = xyes], [mpi2_library=true])
       AS_IF([test x"$enable_mpi" = xyes], [mpi_library=true])
       AS_IF([test x"$enable_oshm" = xyes], [oshm_library=true])
       AS_IF([test x"$enable_upc" = xyes], [upc_compiler=true])
       AS_IF([test x"$enable_upcxx" = xyes], [upcxx_compiler=true])
      ], [
       AC_CHECK_FUNC([MPI_Init], [mpi_library=true])
       AC_CHECK_FUNC([MPI_Accumulate], [mpi2_library=true])
       AC_CHECK_FUNC([MPI_Get_accumulate], [mpi3_library=true])
       AC_CHECK_FUNC([shmem_barrier_all], [oshm_library=true])
       AC_CHECK_FUNC([upc_memput], [upc_compiler=true])
       AC_CHECK_DECL([upcxx_alltoall], [upcxx_compiler=true], [],
                     [#include <upcxx.h>])
      ])

AM_CONDITIONAL([EMBEDDED_BUILD], [test x"$enable_embedded" = xyes])
AM_CONDITIONAL([BUILD_PROFILING_LIB], [test x"$with_plib" = xyes])
AC_SUBST([PMPILIBNAME], [$PMILIBNAME])
AC_SUBST([MPILIBNAME], [$MPILIBNAME])

AS_IF([test "x$enable_openacc" = xyes], [
       AC_CHECK_HEADERS([openacc.h], [],
                        [AC_MSG_ERROR([cannot include openacc.h])])
       AC_DEFINE([_ENABLE_OPENACC_], [1], [Enable OpenACC])
       ])

AS_CASE([$enable_cuda],
        [yes], [build_cuda_kernels=yes; build_cuda=yes],
        [basic], [build_cuda=yes])

AS_IF([test "x$build_cuda" = xyes], [
       AC_SEARCH_LIBS([cuPointerGetAttribute], [cuda], [],
                      [AC_MSG_ERROR([cannot link with -lcuda])])
       AC_SEARCH_LIBS([cudaFree], [cudart], [],
                      [AC_MSG_ERROR([cannot link with -lcudart])])
       AC_CHECK_HEADERS([cuda.h], [],
                        [AC_MSG_ERROR([cannot include cuda.h])])
       AC_DEFINE([_ENABLE_CUDA_], [1], [Enable CUDA])
       ])

AS_IF([test "xbuild_cuda_kernels" = xyes], [
       AC_DEFINE([_ENABLE_CUDA_KERNEL_], [1], [Enable CUDA Kernel])
       ])

AM_CONDITIONAL([MPI2_LIBRARY], [test x$mpi2_library = xtrue])
AM_CONDITIONAL([MPI3_LIBRARY], [test x$mpi3_library = xtrue])
AM_CONDITIONAL([CUDA], [test x$build_cuda = xyes])
AM_CONDITIONAL([CUDA_KERNELS], [test x$build_cuda_kernels = xyes])
AM_CONDITIONAL([OPENACC], [test x$enable_openacc = xyes])
AM_CONDITIONAL([OSHM], [test x$oshm_library = xtrue])
AM_CONDITIONAL([MPI], [test x$mpi_library = xtrue])
AM_CONDITIONAL([UPC], [test x$upc_compiler = xtrue])
AM_CONDITIONAL([UPCXX], [test x$upcxx_compiler = xtrue])

AC_DEFINE([FIELD_WIDTH], [18], [Width of field used to report numbers])
AC_DEFINE([FLOAT_PRECISION], [2], [Precision of reported numbers])

AC_CONFIG_FILES([Makefile mpi/Makefile mpi/pt2pt/Makefile mpi/startup/Makefile
                 mpi/one-sided/Makefile mpi/collective/Makefile
                 openshmem/Makefile upc/Makefile upcxx/Makefile])
AC_OUTPUT
