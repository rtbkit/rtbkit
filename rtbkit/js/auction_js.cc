/* auction_js.cc
   Jeremy Barnes, 6 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Auctions.
*/


#include "auction_js.h"
#include "bid_request_js.h"
#include "soa/js/js_wrapped.h"
#include "jml/utils/smart_ptr_utils.h"
#include "soa/types/js/id_js.h"

using namespace std;
using namespace v8;
using namespace node;

namespace Datacratic {
namespace JS {

/*****************************************************************************/
/* AUCTION JS                                                                */
/*****************************************************************************/

const char * AuctionName = "Auction";

struct AuctionJS
    : public JSWrapped2<Auction, AuctionJS, AuctionName,
                        rtbModule> {
    
    AuctionJS(v8::Handle<v8::Object> This,
              const std::shared_ptr<Auction> & bid
              = std::shared_ptr<Auction>())
    {
        HandleScope scope;
        wrap(This, bid);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new AuctionJS(args.This(), ML::make_std_sp(new Auction()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
        NODE_SET_PROTOTYPE_METHOD(t, "setRequest", setRequest);
        //NODE_SET_PROTOTYPE_METHOD(t, "getResponse", getResponse);
        NODE_SET_PROTOTYPE_METHOD(t, "notifyNoMoreBids", finish);
        NODE_SET_PROTOTYPE_METHOD(t, "finish", finish);

        RWPropertyHandler<AuctionJS, Auction, Date, &Auction::start>
            handleStart(t, "start");
        RWPropertyHandler<AuctionJS, Auction, Date, &Auction::expiry>
            handleExpiry(t, "expiry");
        RWPropertyHandler<AuctionJS, Auction, Id, &Auction::id>
            handleId(t, "id");

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("request"), requestGetter,
                          requestSetter, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(DontDelete));

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("requestStr"), requestStrGetter,
                          requestStrSetter, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(DontDelete));

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("tooLate"), tooLateGetter,
                          0, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(ReadOnly | DontDelete));

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("responses"), responseGetter,
                          0, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(ReadOnly | DontDelete));
    }

    static v8::Handle<v8::Value>
    requestGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            JSValue result;
            to_js(result, getShared(info.This())->request);
            return result;
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    requestSetter(v8::Local<v8::String> property,
                  v8::Local<v8::Value> value,
                  const v8::AccessorInfo & info)
    {
        try {
            std::shared_ptr<BidRequest> br
                = getBidRequestSharedPointer(value);
            if (br) {
                getShared(info.This())->request = br;
            }
            else if (value->IsString()) {

                string s = cstr(value);
                getShared(info.This())->request
                    .reset(BidRequest::parse("datacratic", s));
                getShared(info.This())->requestStr = s;
            }
            else {
                Json::Value request = JS::fromJS(value);

                string s = request.toString();
                getShared(info.This())->request
                    .reset(BidRequest::parse("datacratic", s));
                getShared(info.This())->requestStr = s;

            }
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }

    static v8::Handle<v8::Value>
    requestStrGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            return JS::toJS(getShared(info.This())->requestStr);
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    requestStrSetter(v8::Local<v8::String> property,
                  v8::Local<v8::Value> value,
                  const v8::AccessorInfo & info)
    {
        try {
            Json::Value request = JS::fromJS(value);
            string s = request.toString();
            getShared(info.This())->request
                .reset(BidRequest::parse("datacratic", s));
            getShared(info.This())->requestStr = s;

        } HANDLE_JS_EXCEPTIONS_SETTER;
    }

    static v8::Handle<v8::Value>
    tooLateGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            return v8::Boolean::New(getShared(info.This())->tooLate());
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    responseGetter(v8::Local<v8::String> property,
                   const v8::AccessorInfo & info)
    {
        HandleScope scope;
        try {
            const auto & auction = *getShared(info.This());
            vector<vector<Auction::Response> > responses
                = auction.getResponses();

            vector<vector<Json::Value> > respjson(responses.size());
            for (unsigned spotNum = 0;  spotNum < responses.size();
                 ++spotNum) {
                for (auto it = responses[spotNum].begin(),
                         end = responses[spotNum].end();
                     it != end;  ++it)
                    respjson[spotNum].push_back(it->toJson());
            }
            return scope.Close(JS::toJS(respjson));
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    setRequest(const Arguments & args)
    {
        try {
            Json::Value request = getArg(args, 0, "request");
            string s = request.toString();
            std::shared_ptr<BidRequest> newRequest
                (BidRequest::parse("datacratic", s));
            getShared(args)->request = newRequest;
            getShared(args)->requestStr = s;
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    getResponse(const Arguments & args)
    {
        try {
            return JS::toJS(getShared(args)->getResponseJson(getArg(args, 0, "slotNum")));
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    finish(const Arguments & args)
    {
        try {
            getShared(args)->finish();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
};


std::shared_ptr<Auction>
from_js(const JSValue & value, std::shared_ptr<Auction> *)
{
    return AuctionJS::fromJS(value);
}

Auction *
from_js(const JSValue & value, Auction **)
{
    return AuctionJS::fromJS(value).get();
}


} // namespace JS
} // namespace Datacratic
