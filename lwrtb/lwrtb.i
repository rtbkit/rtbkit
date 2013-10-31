%module(directors="1") lwrtb
%include "std_string.i"
%include "std_vector.i"
namespace std {
%template(StringVector) vector<string>;
}

%{
   /* #define SWIG_FILE_WITH_INIT*/
   #include "lwrtb/bidder.h"
%}

%feature("director")   BidRequestCb;
%feature("director")   DeliveryCb;
%feature("director")   BidResultCb;
%feature("director")   ErrorCb;
%feature("nodirector") Bidder;
%feature("pythonprepend") lwrtb::Bidder::setBidRequestCb(BidRequestCb&) %{
   if len(args) == 1 and (not isinstance(args[0], BidRequestCb) and callable(args[0])):
      class CallableWrapper(BidRequestCb):
         def __init__(self, f):
            super(CallableWrapper, self).__init__()
            self.f_ = f
         def call(self, obj, *args):
            self.f_(obj, *args)

      args = tuple([CallableWrapper(args[0])])
      args[0].__disown__()
   elif len(args) == 1 and isinstance(args[0], BidRequestCb):
      args[0].__disown__()
%}
%feature("pythonprepend") lwrtb::Bidder::setDeliveryCb(DeliveryCb&) %{
   if len(args) == 1 and (not isinstance(args[0], DeliveryCb) and callable(args[0])):
      class CallableWrapper(DeliveryCb):
         def __init__(self, f):
            super(CallableWrapper, self).__init__()
            self.f_ = f
         def call(self, obj, *args):
            self.f_(obj, *args)

      args = tuple([CallableWrapper(args[0])])
      args[0].__disown__()
   elif len(args) == 1 and isinstance(args[0], DeliveryCb):
      args[0].__disown__()
%}
%feature("pythonprepend") lwrtb::Bidder::setBidResultCb(BidResultCb&) %{
   if len(args) == 1 and (not isinstance(args[0], BidResultCb) and callable(args[0])):
      class CallableWrapper(BidResultCb):
         def __init__(self, f):
            super(CallableWrapper, self).__init__()
            self.f_ = f
         def call(self, obj, *args):
            self.f_(obj, *args)

      args = tuple([CallableWrapper(args[0])])
      args[0].__disown__()
   elif len(args) == 1 and isinstance(args[0], BidResultCb):
      args[0].__disown__()
%}
%feature("pythonprepend") lwrtb::Bidder::setErrorCb(ErrorCb&) %{
   if len(args) == 1 and (not isinstance(args[0], ErrorCb) and callable(args[0])):
      class CallableWrapper(ErrorCb):
         def __init__(self, f):
            super(CallableWrapper, self).__init__()
            self.f_ = f
         def call(self, obj, *args):
            self.f_(obj, *args)

      args = tuple([CallableWrapper(args[0])])
      args[0].__disown__()
   elif len(args) == 1 and isinstance(args[0], ErrorCb):
      args[0].__disown__()
%}

%include "bidder.h"
