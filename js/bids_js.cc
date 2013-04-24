/** bids_js.cc                                 -*- C++ -*-
    Rémi Attab, 01 Mar 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    JS Wrappers for the bids classes.
*/

#include "currency_js.h"
#include "bids_js.h"
#include "soa/js/js_wrapped.h"
#include "soa/js/js_call.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_registry.h"
#include "soa/sigslot/slot.h"
#include "jml/utils/smart_ptr_utils.h"


using namespace std;
using namespace v8;
using namespace node;
using namespace RTBKIT;


namespace Datacratic {
namespace JS {


/******************************************************************************/
/* BID                                                                        */
/******************************************************************************/

extern const char* const BidName;
const char * const BidName = "Bid";

struct BidJS : public JSWrapped2<Bid, BidJS, BidName, rtbModule>
{
    BidJS() {}

    BidJS(  v8::Handle<v8::Object> This,
            const std::shared_ptr<Bid>& bid = std::shared_ptr<Bid>())
    {
        HandleScope scope;
        wrap(This, bid);
    }

    static Handle<v8::Value>
    New(const Arguments& args)
    {
        try {
            new BidJS(args.This(), make_shared<Bid>());
            return args.This();
        }
        HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        registerROProperty(&Bid::spotIndex, "spotIndex");
        registerROProperty(&Bid::availableCreatives, "availableCreatives");
        registerROProperty(&Bid::price, "price");
        registerROProperty(&Bid::creativeIndex, "creativeIndex");
        registerROProperty(&Bid::priority, "priority");
        registerROProperty(&Bid::account, "account");

    }
};

Bid* from_js(const JSValue& value, Bid**)
{
    return BidJS::fromJS(value).get();
}

Bid from_js(const JSValue& value, Bid*)
{
    return *BidJS::fromJS(value);
}

void to_js(JS::JSValue& value, const Bid& bid)
{
    value = BidJS::toJS(make_shared<Bid>(bid));
}


/******************************************************************************/
/* BIDS                                                                       */
/******************************************************************************/

extern const char* const BidsName;
const char * const BidsName = "Bids";

struct BidsJS : public JSWrapped2<Bids, BidsJS, BidsName, rtbModule>
{
    BidsJS() {}

    BidsJS( v8::Handle<v8::Object> This,
            const std::shared_ptr<Bids>& bids = std::shared_ptr<Bids>())
    {
        HandleScope scope;
        wrap(This, bids);
    }

    static Handle<v8::Value>
    New(const Arguments& args)
    {
        try {
            new BidsJS(args.This(), make_shared<Bids>());
            return args.This();
        }
        HANDLE_JS_EXCEPTIONS;
    }


    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        NODE_SET_PROTOTYPE_METHOD(t, "bid", bid);
        NODE_SET_PROTOTYPE_METHOD(t, "addSource", addSource);
        registerMemberFn(&RTBKIT::Bids::bidForSpot, "bidForSpot");

        registerROProperty(&Bids::size, "length");
        t->InstanceTemplate()->SetIndexedPropertyHandler(at, 0, check, 0, list);
    }

    /** Ideally this would be in the BidJS class but, because bids is an array
        of objects and not shared_ptr, the Bid objects get copied everytime they
        are accessed which means that any modifications will affect the copy and
        not the original. Annoying.
     */
    static Handle<Value>
    bid(const Arguments& args)
    {
        try {
            unsigned spot     = getArg<unsigned> (args, 0, "spot");
            int creativeIndex = getArg<int>      (args, 1, "creativeIndex");
            Amount price      = getArg<Amount>   (args, 2, "price");
            double priority   = getArg<double>   (args, 3, 0.0, "priority");

            auto obj = getShared(args);
            (*obj)[spot].bid(creativeIndex, price, priority);

            return Handle<Value>();
        }
        HANDLE_JS_EXCEPTIONS;
    }

    /** Again, this should be in BidJS but the copies make this annoyingly hard.
     */
    static Handle<Value>
    addSource(const Arguments& args)
    {
        try {
            string source = getArg<string>(args, 0, "source");
            getShared(args)->dataSources.insert(source);
            return Handle<Value>();
        }
        HANDLE_JS_EXCEPTIONS;
    }


    static Handle<Value>
    at(uint32_t index, const v8::AccessorInfo & info)
    {
        try {
            auto data = getSharedPtr(info.This());

            if (index >= data->size())
                return Handle<Value>();

            return Datacratic::JS::toJS((*data)[index]);
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<Integer>
    check(uint32_t index, const v8::AccessorInfo & info)
    {
        try {
            auto data = getSharedPtr(info.This());
            return Integer::New(index >= data->size());
        }
        catch (...) {
            translateCurrentException();
            return Handle<Integer>();
        }
    }

    static Handle<Array>
    list(const v8::AccessorInfo & info)
    {
        try {
            auto data = getSharedPtr(info.This());

            auto array = Array::New(data->size());
            for (size_t i = 0; i < data->size(); ++i)
                array->Set(i, Integer::New(i));

            return array;
        }
        catch (...) {
            translateCurrentException();
            return Handle<Array>();
        }
    }

};

Bids from_js(const JSValue& value, Bids*)
{
    return *BidsJS::fromJS(value);
}

Bids* from_js(const JSValue& value, Bids**)
{
    return BidsJS::fromJS(value).get();
}

void to_js(JS::JSValue& value, const Bids& bids)
{
    value = BidsJS::toJS(std::make_shared<Bids>(bids));
}

} // namepsace JS
} // namespace Datacratic
