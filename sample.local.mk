# to speed up repeated builds, use ccache
CXX := ccache g++
CC := ccache gcc
FC := ccache gfortran

# to get colorized output with ccache, put colorccache on your path and use
# CXX := colorccache g++
# CC := colorccache gcc
# FC := colorccache gfortran

BOOST_VERSION := 52
TCMALLOC_ENABLED := 1
