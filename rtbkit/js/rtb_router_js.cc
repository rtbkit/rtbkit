/* rtb_router_js.cc
   Jeremy Barnes, 5 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   JS bindings for RTB router.
*/

#include "rtbkit/core/router/router_stack.h"
#include "rtb_js.h"
#include "auction_js.h"
#include "banker_js.h"
#include "v8.h"
#include "node.h"
#include "soa/js/js_value.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_wrapped.h"
#include "soa/js/js_call.h"
#include "soa/js/js_registry.h"
#include "jml/arch/timers.h"
#include "soa/sigslot/slot.h"
#include "jml/utils/guard.h"
#include <boost/make_shared.hpp>
#include "rtbkit/plugins/bidding_agent/bidding_agent.h"
#include "node/uv.h"
#include "soa/types/js/id_js.h"
#include "bid_request_js.h"
#include "currency_js.h"

using namespace std;
using namespace v8;
using namespace node;
using namespace ML;
using namespace RTBKIT;

namespace Datacratic {
namespace JS {

extern Registry registry;


/*****************************************************************************/
/* RTB ROUTER JS                                                             */
/*****************************************************************************/

/** A combination of enough of a router to run a simulation:
    - Router
    - Post auction service
    - Banker
*/

const char * RTBRouterStackName = "Router";

struct RTBRouterStackJS
    : public JSWrapped2<RTBKIT::RouterStack, RTBRouterStackJS, RTBRouterStackName,
                        rtbModule, false> {

    RTBRouterStackJS(const v8::Handle<v8::Object> & This,
                      const std::shared_ptr<RTBKIT::RouterStack> & router
                          = std::shared_ptr<RTBKIT::RouterStack>())
    {
        wrap(This, router);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new RTBRouterStackJS
                (args.This(),
                 std::shared_ptr<RTBKIT::RouterStack>
                 (new RTBKIT::RouterStack(std::make_shared<ServiceProxies>(),
                                  getArg(args, 2, "router", "routerName"),
                                  getArg(args, 0, 2.0,
                                         "secondsUntilLossAssumed"))));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        //NODE_SET_PROTOTYPE_METHOD(t, "initEndpoints", initEndpoints);
        NODE_SET_PROTOTYPE_METHOD(t, "createProxy", createProxy);
        NODE_SET_PROTOTYPE_METHOD(t, "bindAgents", bindAgents);
        NODE_SET_PROTOTYPE_METHOD(t, "start", start);
        NODE_SET_PROTOTYPE_METHOD(t, "sleepUntilIdle", sleepUntilIdle);
        NODE_SET_PROTOTYPE_METHOD(t, "shutdown", shutdown);
        NODE_SET_PROTOTYPE_METHOD(t, "injectAuction", injectAuction);
        NODE_SET_PROTOTYPE_METHOD(t, "injectWin", injectWin);
        NODE_SET_PROTOTYPE_METHOD(t, "injectLoss", injectLoss);
        NODE_SET_PROTOTYPE_METHOD(t, "injectCampaignEvent", injectCampaignEvent);
        NODE_SET_PROTOTYPE_METHOD(t, "numAuctionsInProgress",
                                  numAuctionsInProgress);
        NODE_SET_PROTOTYPE_METHOD(t, "getStats", getStats);

#if 0

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("banker"),
                          bankerGetter, bankerSetter);
#endif

#if 0
        registerRWPropertyGetterSetter(&RTBKIT::RouterStack::getBanker,
                                       &RTBKIT::RouterStack::setBanker,
                                       "banker", v8::DontDelete);
#endif
        registerMemberFn(&RTBKIT::RouterStack::notifyFinishedAuction,
                         "notifyFinishedAuction");
        registerMemberFn(&RTBKIT::RouterStack::init, "init");
    }

#if 0
    static Handle<v8::Value>
    initEndpoints(const Arguments & args)
    {
        try {
            getShared(args)
                ->initEndpoints(getArg(args, 0, -1, "bidPort"),
                                //getArg(args, 1, -1, "backchannelPort"),
                                getArg(args, 2, "localhost", "hostname"),
                                getArg(args, 3,  1, "nThreads"));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
#endif

    static Handle<v8::Value>
    createProxy(const Arguments & args)
    {
        try {

            return JS::toJS
                (std::make_shared<RTBKIT::BiddingAgent>
                 (getShared(args)->getServices(), "bidding_agent"));
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    bindAgents(const Arguments & args)
    {
        try {
            getShared(args)
                ->router.bindAgents(getArg(args, 0, "agentsUri"));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    start(const Arguments & args)
    {
        try {
            // Make sure Node doesn't exit and we don't get GCd when the
            // event loop is running.

            ev_ref(ev_default_loop());
            v8::Persistent<v8::Object> phandle
                = v8::Persistent<v8::Object>::New(args.This());

            auto cleanup = [=] ()
                {
                    //cerr << "calling cleanup" << endl;

                    v8::Persistent<v8::Object> handle = phandle;
                    ev_unref(ev_default_loop());

                    //cerr << "done loop unref" << endl;
                    //cerr << "depth " << ev_depth(loop) << endl;
                    //cerr << "iter " << ev_iteration(loop) << endl;
                    //cerr << "watching " << ev_pending_count(loop) << endl;
                    //ev_break(loop, EVBREAK_ONE);

                    handle.Clear();
                    handle.Dispose();

                };

            getShared(args)->start(cleanup);

            // TODO: add something to node to stop it from returning?

            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    shutdown(const Arguments & args)
    {
        try {
            getShared(args)->shutdown();
            //cerr << "finished shutdown" << endl;
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    sleepUntilIdle(const Arguments & args)
    {
        try {
            getShared(args)->sleepUntilIdle();
            //cerr << "finished shutdown" << endl;
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    struct CallbackData {
        Slot onAuctionFinished;
        std::shared_ptr<Auction> auction;
        v8::Persistent<v8::Object> This;

        ~CallbackData()
        {
            if (!This.IsEmpty()) {
                This.Dispose();
                This.Clear();
            }
        }
    };

    static void doNothing(eio_req * req)
    {
        // TODO: don't do this; find how to use libeio properly
    }

    static int finishedCallback(eio_req * req)
    {
        HandleScope scope;

        auto_ptr<CallbackData> data((CallbackData *)req->data);

        TryCatch try_catch;

        try {
            const int argc = 1;
            v8::Handle<v8::Value> argv[argc] = { JS::toJS(data->auction) };
            data->onAuctionFinished.call(data->This, argc, argv);
        } catch (const JSPassException & exc) {
            // Corner case... but probably shouldn't happen
        } catch (...) {
            v8::Handle<v8::Value> result = translateCurrentException();
            v8::ThrowException(result);
        }

        if (try_catch.HasCaught())
            FatalException(try_catch);

        return 0;
    }


    static Handle<v8::Value>
    injectAuction(const Arguments & args)
    {
        try {
            Slot onAuctionFinished = getArg(args, 0, "onAuctionFinished");
            double startTime       = getArg(args, 2,
                                            Date::now().secondsSinceEpoch(),
                                            "startTime");
            double expiryTime      = getArg(args, 3, startTime + 0.03,
                                            "expiryTime");
            double lossTime        = getArg(args, 4,
                                            Date::positiveInfinity()
                                                .secondsSinceEpoch(),
                                            "lossTime");

            std::shared_ptr<BidRequest> request;
            std::string requestStr;

            v8::Handle<v8::Value> requestArg = args[1];
            std::string requestFormat;
            if (requestArg->IsString()) {
                requestStr = getArg<string>(args, 1, "requestStr");
                requestFormat = getArg<string>(args, 5, "datacratic",
                                               "requestFormat");
                request.reset(BidRequest::parse(requestFormat, requestStr));
            }
            else if (request = getBidRequestSharedPointer(requestArg)) {
                requestStr = request->toJsonStr();
                requestFormat = "datacratic";
            }
            else throw ML::Exception("don't know how to turn " + cstr(requestArg)
                                     + " into a bidRequest for injectAuction");

            CallbackData * data = new CallbackData();
            Call_Guard deleteDataGuard([&] () { delete data; });

            data->onAuctionFinished = onAuctionFinished;
            data->This = v8::Persistent<v8::Object>::New(args.This());

            /* Create an asynchronous, libeioified wrapper that will call
               back to the JS callback from the correct callback thread.
            */
            auto onAuctionFinished2 = [=] (std::shared_ptr<Auction> auction)
                {
                    data->auction = auction;

                    eio_custom(doNothing, EIO_PRI_DEFAULT, finishedCallback,
                               data);

                    return true;
                };

            getShared(args)
                ->router.injectAuction(onAuctionFinished2,
                                       request, requestStr, requestFormat,
                                       startTime, expiryTime, lossTime);

            deleteDataGuard.clear();

            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    injectWin(const Arguments & args)
    {
        try {
            Id auctionId = getArg(args, 0, "auctionId");
            Id adSpot = getArg(args, 1, "adspot");
            Amount winAmount = getArg(args, 2, "winPrice");
            Date timestamp
                = Date::fromSecondsSinceEpoch
                    (getArg<double>(args, 3, "timestamp"));
            Json::Value meta = getArg(args, 4, Json::Value(), "meta");
            UserIds uids = getArg(args, 5, UserIds(), "userIds");
            AccountKey account = getArg(args, 6, AccountKey(), "account");
            Date bidTimestamp
                = Date::fromSecondsSinceEpoch
                    (getArg<double>(args, 7, -1, "bidTimestamp"));
            getShared(args)->injectWin(auctionId, adSpot, winAmount,
                                       timestamp, meta, uids,
                                       account, bidTimestamp);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    injectLoss(const Arguments & args)
    {
        try {
            Id auctionId = getArg(args, 0, "auctionId");
            Id adSpot = getArg(args, 1, "adspot");
            Date timestamp
                = Date::fromSecondsSinceEpoch
                    (getArg<double>(args, 2, "timestamp"));
            Json::Value meta = getArg(args, 3, Json::Value(), "meta");
            AccountKey account = getArg(args, 4, AccountKey(), "account");
            Date bidTimestamp
                = Date::fromSecondsSinceEpoch
                    (getArg<double>(args, 5, -1, "bidTimestamp"));
            getShared(args)->injectLoss(auctionId, adSpot, timestamp,
                                        meta, account, bidTimestamp);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    injectCampaignEvent(const Arguments & args)
    {
        try {
            string label = getArg(args, 0, "label");
            Id auctionId = getArg(args, 1, "auctionId");
            Id adSpot = getArg(args, 2, "adspot");
            Date timestamp
                = Date::fromSecondsSinceEpoch
                    (getArg<double>(args, 3, "timestamp"));
            Json::Value meta = getArg(args, 4, Json::Value(), "meta");
            UserIds uids = getArg(args, 5, UserIds(), "userIds");
            getShared(args)->injectCampaignEvent(label, auctionId, adSpot,
                                                 timestamp, meta, uids);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    numAuctionsInProgress(const Arguments & args)
    {
        try {
            return JS::toJS(getShared(args)->numAuctionsInProgress());
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    getStats(const Arguments & args)
    {
        try {
            return JS::toJS(getShared(args)->getStats());
        } HANDLE_JS_EXCEPTIONS;
    }

#if 0
    static Handle<v8::Value>
    addBudget(const Arguments & args)
    {
        try {
            return JS::toJS(getShared(args)
                            ->addBudget(getArg(args, 0, "account"),
                                        getArg(args, 1, "amount")));
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    setBudget(const Arguments & args)
    {
        try {
            return JS::toJS(getShared(args)
                            ->setBudget(getArg(args, 0, "account"),
                                        getArg(args, 1, "amount")));
        } HANDLE_JS_EXCEPTIONS;
    }
#endif

#if 0
    static v8::Handle<v8::Value>
    bankerGetter(v8::Local<v8::String> property,
                 const AccessorInfo & info)
    {
        try {
            return JS::toJS(getShared(info.This())->getBanker());
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    bankerSetter(v8::Local<v8::String> property,
                 v8::Local<v8::Value> value,
                 const AccessorInfo & info)
    {
        try {
            getShared(info.This())->setBanker(JS::fromJS(value));
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }
#endif

};

RegisterJsOps<void (std::shared_ptr<Auction>)> reg_handleAuction;

} // namespace JS
} // namespace Datacratic
