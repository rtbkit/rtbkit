%module "api"
%{ 
    #define SWIG_FILE_WITH_INIT
    #include "api/bidder.h"
%}
%include "bidder.h"
