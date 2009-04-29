LIBALGEBRA_SOURCES := \
        least_squares.cc \
        irls.cc \
        lapack.cc \
	ilaenv.f \
        svd.cc \
        matrix_ops.cc

$(eval $(call add_sources,$(LIBALGEBRA_SOURCES)))

LIBALGEBRA_LINK :=	utils lapack blas gfortran db

$(eval $(call library,algebra,$(LIBALGEBRA_SOURCES),$(LIBALGEBRA_LINK)))

$(eval $(call include_sub_make,algebra_testing,testing))
