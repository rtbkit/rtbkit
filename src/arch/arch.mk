LIBARCH_SOURCES := \
        simd_vector.cc \
        demangle.cc \
	tick_counter.cc \
	cpuid.cc \
	simd.cc \
	exception.cc \
	backtrace.cc \
        format.cc \
	exception_handler.cc \
	gpgpu.cc \
	environment_static.cc

$(eval $(call add_sources,$(LIBARCH_SOURCES)))
$(eval $(call add_sources,exception_hook.cc))

LIBARCH_LINK :=	ACE

$(eval $(call library,arch,$(LIBARCH_SOURCES),$(LIBARCH_LINK)))

ifeq ($(CUDA_ENABLED),1)

LIBARCH_CUDA_SOURCES 	:= cuda.cc
LIBARCH_CUDA_LINK 	:= arch cuda cudart

$(eval $(call library,arch_cuda,$(LIBARCH_CUDA_SOURCES),$(LIBARCH_CUDA_LINK)))

endif # CUDA_ENABLED


ifeq ($(CAL_ENABLED),1)

LIBARCH_CAL_SOURCES 	:= cal.cc
LIBARCH_CAL_LINK 	:= arch amd

$(eval $(call library,arch_cal,$(LIBARCH_CAL_SOURCES),$(LIBARCH_CAL_LINK)))

$(eval $(call include_sub_make,arch_testing,testing))

endif # CAL_ENABLED
