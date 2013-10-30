%module "lwrtb"
%{ 
    #define SWIG_FILE_WITH_INIT
    #include "lwrtb/bidder.h"
%}
%include "bidder.h"
