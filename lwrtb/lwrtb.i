%module(directors="1") lwrtb
%include "std_string.i"
%include "std_vector.i"
%{
   #include "lwrtb/bidder.h"
%}
%feature("director")   BidRequestCb;
%feature("director")   DeliveryCb;
%feature("director")   BidResultCb;
%feature("director")   ErrorCb;
%feature("nodirector") Bidder;

%include "bidder.h"
