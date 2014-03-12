# Library to provide the weak definition of slarfp_ needed to determine if
# LAPACK 3.2 or later is installed.  Needs to be in a separate library so
# that the strong version from lapack can be picked up if it exists.
$(eval $(call library,lapack_detect,lapack_detect.cc,))

LIBALGEBRA_SOURCES := \
        least_squares.cc \
        irls.cc \
        lapack.cc \
	ilaenv.f \
        svd.cc \
        matrix_ops.cc

$(eval $(call add_sources,$(LIBALGEBRA_SOURCES)))

LIBALGEBRA_LINK :=	utils lapack blas gfortran db lapack_detect ACE worker_task

$(eval $(call library,algebra,$(LIBALGEBRA_SOURCES),$(LIBALGEBRA_LINK)))

$(eval $(call include_sub_make,algebra_testing,testing))
