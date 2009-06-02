CXX := colorccache distcc g++
CXXFLAGS := -I. -pipe -Wall -Werror -Wno-sign-compare -O3 -march=i686 -m32 -g -fPIC
CXXLINKFLAGS := -shared -L$(BIN) -Wl,--rpath,$(BIN)
CXXEXEFLAGS :=	-L$(BIN) -Wl,--rpath,$(BIN)
CXXDEBUGFLAGS := -O0 -g

FC := colorccache f77
FFLAGS := -I. -fPIC

