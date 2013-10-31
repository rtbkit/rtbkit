%module(directors="1") lwrtb
%include "std_string.i"
%include "std_vector.i"
namespace std {
%template(StringVector) vector<string>;
}
%{
   #define SWIG_FILE_WITH_INIT
   #include "lwrt/bidder.h"
%}
%feature("director")   lwrtb::BidRequestCb;
%feature("director")   lwrtb::DeliveryCb;
%feature("director")   lwrtb::BidResultCb;
%feature("director")   lwrtb::ErrorCb;
%feature("nodirector") lwrtb::Bidder;
%feature("pythonprepend") lwrtb::Bidder::setBidRequestCb(lwrtb::BidRequestCb&) %{
   if len(args) == 1 and (not isinstance(args[0], Callback) and callable(args[0])):
      class CallableWrapper(Callback):
         def __init__(self, f):
            super(CallableWrapper, self).__init__()
            self.f_ = f
         def call(self, obj, *args):
            self.f_(obj, *args)

      args = tuple([CallableWrapper(args[0])])
      args[0].__disown__()
   elif len(args) == 1 and isinstance(args[0], Callback):
      args[0].__disown__()
%}
%feature("pythonprepend") lwrtb::Bidder::setDeliveryCb(lwrtb::DeliveryCb&) %{
   if len(args) == 1 and (not isinstance(args[0], Callback) and callable(args[0])):
      class CallableWrapper(Callback):
         def __init__(self, f):
            super(CallableWrapper, self).__init__()
            self.f_ = f
         def call(self, obj, *args):
            self.f_(obj, *args)

      args = tuple([CallableWrapper(args[0])])
      args[0].__disown__()
   elif len(args) == 1 and isinstance(args[0], Callback):
      args[0].__disown__()
%}
%feature("pythonprepend") lwrtb::Bidder::setBidResultCb(lwrtb::BidResultCb&) %{
   if len(args) == 1 and (not isinstance(args[0], Callback) and callable(args[0])):
      class CallableWrapper(Callback):
         def __init__(self, f):
            super(CallableWrapper, self).__init__()
            self.f_ = f
         def call(self, obj, *args):
            self.f_(obj, *args)

      args = tuple([CallableWrapper(args[0])])
      args[0].__disown__()
   elif len(args) == 1 and isinstance(args[0], Callback):
      args[0].__disown__()
%}
%feature("pythonprepend") lwrtb::Bidder::setErrorCb(lwrtb::ErrorCb&) %{
   if len(args) == 1 and (not isinstance(args[0], Callback) and callable(args[0])):
      class CallableWrapper(Callback):
         def __init__(self, f):
            super(CallableWrapper, self).__init__()
            self.f_ = f
         def call(self, obj, *args):
            self.f_(obj, *args)

      args = tuple([CallableWrapper(args[0])])
      args[0].__disown__()
   elif len(args) == 1 and isinstance(args[0], Callback):
      args[0].__disown__()
%}

%include "bidder.h"
