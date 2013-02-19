/* bid_request_js.cc
   Jeremy Barnes, 6 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Bid Request.
*/


#include "bid_request_js.h"
#include "soa/js/js_wrapped.h"
#include "jml/utils/smart_ptr_utils.h"
#include "soa/types/js/id_js.h"
#include "soa/types/js/url_js.h"
#include <boost/make_shared.hpp>
#include <boost/algorithm/string/trim.hpp>

using namespace std;
using namespace v8;
using namespace node;

namespace Datacratic {
namespace JS {


extern const char * const bidRequestModule;
const char * const bidRequestModule = "bid_request";
//so we can do require("standalone_demo")


/*****************************************************************************/
/* SEGMENT LIST JS                                                           */
/*****************************************************************************/

const char * SegmentListName = "SegmentList";

struct SegmentListJS
    : public JSWrapped2<SegmentList, SegmentListJS, SegmentListName,
                        bidRequestModule> {

    SegmentListJS(v8::Handle<v8::Object> This,
              const std::shared_ptr<SegmentList> & list
                  = std::shared_ptr<SegmentList>())
    {
        HandleScope scope;
        wrap(This, list);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            if (args.Length() == 0) {
                new SegmentListJS(args.This(),
                                  std::make_shared<SegmentList>());
                return args.This();
            }
            else {
                if (SegmentListJS::tmpl->HasInstance(args[0])) {
                    throw ML::Exception("segment list from segment list");
                    //new SegmentListJS(args.This(),
                    //                  std::make_shared<SegmentList>
                    //                  (*SegmentListJS::fromJS(args[0])));
                    return args.This();
                }
                
                Json::Value valInJson = JS::fromJS(args[0]);
                new SegmentListJS(args.This(),
                                  std::make_shared<SegmentList>
                                  (SegmentList::createFromJson(valInJson)));
                return args.This();
            }
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
        NODE_SET_PROTOTYPE_METHOD(t, "forEach", forEach);
        NODE_SET_PROTOTYPE_METHOD(t, "add", add);
        NODE_SET_PROTOTYPE_METHOD(t, "toArray", toArray);
        NODE_SET_PROTOTYPE_METHOD(t, "toString", toString);
        NODE_SET_PROTOTYPE_METHOD(t, "inspect", toString);

        registerMemberFn(&SegmentList::toJson, "toJSON");

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("length"), lengthGetter,
                          0, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(ReadOnly | DontEnum | DontDelete));
                          
        t->InstanceTemplate()
            ->SetIndexedPropertyHandler(getIndexed, setIndexed, queryIndexed,
                                        deleteIndexed, listIndexed);
    }

    static v8::Handle<v8::Value>
    add(const Arguments & args)
    {
        try {
            auto segs = getShared(args.This());
            segs->add(getArg<string>(args, 0, "segment"));
            segs->sort();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    forEach(const Arguments & args)
    {
        HandleScope scope;
        try {
            auto segs = getShared(args.This());

            v8::Local<v8::Function> fn = getArg(args, 0, "iterFunction");

            for (unsigned index = 0;  index < segs->size();  ++index) {
                HandleScope scope;
                JSValue value;
                
                if (index < segs->ints.size())
                    value = JS::toJS(segs->ints[index]);
                else if (index - segs->ints.size() < segs->strings.size())
                    value = JS::toJS(segs->strings[index - segs->ints.size()]);
                else throw ML::Exception("logic error in forEach");

                // Argument 1: value
                // Argument 2: index number
                int argc = 2;
                v8::Handle<v8::Value> argv[argc];
            
                argv[0] = value;
                argv[1] = v8::Uint32::New(index);
                
                v8::Handle<v8::Value> result
                    = fn->Call(args.This(), argc, argv);
                
                // Exception?
                if (result.IsEmpty())
                    return scope.Close(result);

                if (index == segs->size() - 1 && !result->IsUndefined())
                    return scope.Close(result);
            }

            return v8::Undefined();
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    getIndexed(uint32_t index, const v8::AccessorInfo & info)
    {
        try {
            auto segs = getShared(info.This());
            
            if (index < segs->ints.size())
                return JS::toJS(segs->ints[index]);
            else if (index - segs->ints.size() < segs->strings.size())
                return JS::toJS(segs->strings[index - segs->ints.size()]);
            else return v8::Undefined();
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    setIndexed(uint32_t index,
               v8::Local<v8::Value> value,
               const v8::AccessorInfo & info)
    {
        try {
            throw ML::Exception("can't modify segments argument");
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Integer>
    queryIndexed(uint32_t index,
                 const v8::AccessorInfo & info)
    {
        auto segs = getShared(info.This());
        int sz = segs->size();

        if (index <= sz)
            return v8::Integer::New(ReadOnly | DontDelete);
        
        return NULL_HANDLE;
    }

    static v8::Handle<v8::Array>
    listIndexed(const v8::AccessorInfo & info)
    {
        v8::HandleScope scope;
        auto segs = getShared(info.This());
        int sz = segs->size();

        v8::Handle<v8::Array> result(v8::Array::New(sz));

        for (unsigned i = 0;  i < sz;  ++i) {
            result->Set(v8::Uint32::New(i),
                        v8::Uint32::New(i));
        }
        
        return scope.Close(result);
    }

    static v8::Handle<v8::Boolean>
    deleteIndexed(uint32_t index,
                  const v8::AccessorInfo & info)
    {
        return NULL_HANDLE;
    }

    static v8::Handle<v8::Value>
    toArray(const v8::Arguments & args)
    {
        try {
            v8::HandleScope scope;
            auto segs = getShared(args.This());
            int sz = segs->size();

            v8::Handle<v8::Array> result(v8::Array::New(sz));

            for (unsigned i = 0;  i < segs->ints.size();  ++i) {
                result->Set(v8::Uint32::New(i),
                            v8::Uint32::New(segs->ints[i]));
            }
            for (unsigned i = 0;  i < segs->strings.size();  ++i) {
                result->Set(v8::Uint32::New(i + segs->ints.size()),
                            JS::toJS(segs->strings[i]));
            }

            return scope.Close(result);
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    toString(const v8::Arguments & args)
    {
        try {
            auto segs = getShared(args.This());
            return JS::toJS(boost::trim_copy(segs->toJson().toString()));
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    lengthGetter(v8::Local<v8::String> property,
                 const AccessorInfo & info)
    {
        try {
            return v8::Integer::New(getShared(info.This())->size());
        } HANDLE_JS_EXCEPTIONS;
    }
};

std::shared_ptr<SegmentList>
from_js(const JSValue & value, std::shared_ptr<SegmentList> *)
{
    if (SegmentListJS::tmpl->HasInstance(value))
        return SegmentListJS::fromJS(value);

    vector<string> values;
    JS::from_js(value, &values);
    return std::make_shared<SegmentList>(values);
}

SegmentList *
from_js(const JSValue & value, SegmentList **)
{
    return SegmentListJS::fromJS(value).get();
}

std::shared_ptr<SegmentList>
from_js_ref(const JSValue & value, std::shared_ptr<SegmentList> *)
{
    return SegmentListJS::fromJS(value);
}

void to_js(JS::JSValue & value, const std::shared_ptr<SegmentList> & br)
{
    value = SegmentListJS::toJS(br);
}

SegmentList
from_js(const JSValue & value, SegmentList *)
{
    if (SegmentListJS::tmpl->HasInstance(value))
        return *SegmentListJS::fromJS(value);
    Json::Value valInJson = JS::fromJS(value);
    SegmentList result = SegmentList::createFromJson(valInJson);
    return result;
}

void to_js(JS::JSValue & value, const UserIds & uids)
{
    to_js(value, static_cast<const std::map<std::string, Id> &>(uids));
}

UserIds 
from_js(const JSValue & value, UserIds *)
{
    UserIds result;
    static_cast<std::map<std::string, Id> &>(result)
        = from_js(value, (std::map<std::string, Id> *)0);
    return result;
}


/*****************************************************************************/
/* SEGMENTS BY SOURCE JS                                                     */
/*****************************************************************************/

const char * SegmentsBySourceName = "SegmentsBySource";

struct SegmentsBySourceJS
    : public JSWrapped2<SegmentsBySource, SegmentsBySourceJS,
                        SegmentsBySourceName,
                        bidRequestModule> {

    SegmentsBySourceJS(v8::Handle<v8::Object> This,
                       const std::shared_ptr<SegmentsBySource> & list
                           = std::shared_ptr<SegmentsBySource>())
    {
        HandleScope scope;
        wrap(This, list);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new SegmentsBySourceJS(args.This(),
                                   std::make_shared<SegmentsBySource>());
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        t->InstanceTemplate()
            ->SetNamedPropertyHandler(getNamed, setNamed, queryNamed,
                                      deleteNamed, listNamed);

    }

    static v8::Handle<v8::Value>
    getNamed(v8::Local<v8::String> property,
             const v8::AccessorInfo & info)
    {
        HandleScope scope;
        try {
            Local<v8::Value> object_prop
                = info.This()->GetRealNamedProperty(property);
            if (!object_prop.IsEmpty())
                return scope.Close(object_prop);
            
            // Is it a column name?
            string name = cstr(property);
            
            SegmentsBySource * segs = getShared(info.This());
            
            if (!segs->count(name))
                return NULL_HANDLE;

            return scope.Close(JS::toJS(segs->find(name)->second));
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    setNamed(v8::Local<v8::String> property,
             v8::Local<v8::Value> value,
             const v8::AccessorInfo & info)
    {
        try {
            if (info.This()->HasRealNamedProperty(property)) {
                if (info.This()->Set(property, value))
                    return value;
            }
            
            // Is it a column name?
            string name = cstr(property);
            SegmentsBySource * segs = getShared(info.This());
            
            // Is the value sensible?
            if (value->IsNull() || value->IsUndefined()) {
                throw ML::Exception("can't set named to undefined");
            }
            
            std::shared_ptr<SegmentList> segs2
                = from_js(value, &segs2);

            if (!segs2)
                throw ML::Exception("can't set to null segments");

            (*segs)[name] = segs2;
            
            return v8::Undefined();
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Integer>
    queryNamed(v8::Local<v8::String> property,
               const v8::AccessorInfo & info)
    {
        if (property.IsEmpty() || property->IsNull()
            || property->IsUndefined())
            throw ML::Exception("queryNamed: invalid property");

        string name = cstr(property);

        SegmentsBySource * segs = getShared(info.This());
        
        if (segs->count(name))
            return v8::Integer::New(DontDelete);
        
        return NULL_HANDLE;
    }

    static v8::Handle<v8::Boolean>
    deleteNamed(v8::Local<v8::String> property,
                const v8::AccessorInfo & info)

    {
        if (property.IsEmpty() || property->IsNull()
            || property->IsUndefined())
            throw ML::Exception("queryNamed: invalid property");

        string name = cstr(property);

        SegmentsBySource * segs = getShared(info.This());

        return v8::Boolean::New(segs->erase(name));
    }

    static v8::Handle<v8::Array>
    listNamed(const v8::AccessorInfo & info)
    {
        //cerr << "listNamed" << endl;
        HandleScope scope;
        try {
            SegmentsBySource * segs = getShared(info.This());

            int n = segs->size();

            v8::Handle<v8::Array> result = v8::Array::New(n);

            //cerr << "listNamed: " << ncol << " columns" << endl;

            unsigned i = 0;
            for (auto it = segs->begin(), end = segs->end();
                 it != end;  ++it,++i) {
                v8::Local<Integer> key = v8::Integer::New(i);
                v8::Handle<Value>  val = JS::toJS(it->first);
                result->Set(key, val);
            }
            
            return scope.Close(result);
        } catch (...) {
            cerr << "got exception in listNamed" << endl;
            return NULL_HANDLE;
        }
    }
};


/*****************************************************************************/
/* USER IDS JS                                                               */
/*****************************************************************************/

const char * UserIdsName = "UserIds";

struct UserIdsJS
    : public JSWrapped2<UserIds, UserIdsJS,
                        UserIdsName,
                        bidRequestModule> {

    UserIdsJS(v8::Handle<v8::Object> This,
                       const std::shared_ptr<UserIds> & list
                           = std::shared_ptr<UserIds>())
    {
        HandleScope scope;
        wrap(This, list);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new UserIdsJS(args.This(),
                                   std::make_shared<UserIds>());
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        t->InstanceTemplate()
            ->SetNamedPropertyHandler(getNamed, setNamed, queryNamed,
                                      deleteNamed, listNamed);
    }

    static v8::Handle<v8::Value>
    getNamed(v8::Local<v8::String> property,
             const v8::AccessorInfo & info)
    {
        HandleScope scope;
        try {
            Local<v8::Value> object_prop
                = info.This()->GetRealNamedProperty(property);
            if (!object_prop.IsEmpty())
                return scope.Close(object_prop);
            
            // Is it a column name?
            string name = cstr(property);
            
            UserIds * ids = getShared(info.This());
            
            if (!ids->count(name))
                return NULL_HANDLE;

            return scope.Close(JS::toJS(ids->find(name)->second));
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    setNamed(v8::Local<v8::String> property,
             v8::Local<v8::Value> value,
             const v8::AccessorInfo & info)
    {
        try {
            if (info.This()->HasRealNamedProperty(property)) {
                if (info.This()->Set(property, value))
                    return value;
            }
            
            // Is it a column name?
            string name = cstr(property);
            UserIds * ids = getShared(info.This());
            
            // Is the value sensible?
            if (value->IsNull() || value->IsUndefined()) {
                throw ML::Exception("can't set named to undefined");
            }
            
            Id id = from_js(value, &id);

            if (!id)
                throw ML::Exception("can't set to null ID");
            
            ids->set(id, name);
            
            return v8::Undefined();
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Integer>
    queryNamed(v8::Local<v8::String> property,
               const v8::AccessorInfo & info)
    {
        if (property.IsEmpty() || property->IsNull()
            || property->IsUndefined())
            throw ML::Exception("queryNamed: invalid property");

        string name = cstr(property);

        UserIds * ids = getShared(info.This());
        
        if (ids->count(name))
            return v8::Integer::New(DontDelete);
        
        return NULL_HANDLE;
    }

    static v8::Handle<v8::Boolean>
    deleteNamed(v8::Local<v8::String> property,
                const v8::AccessorInfo & info)

    {
        if (property.IsEmpty() || property->IsNull()
            || property->IsUndefined())
            throw ML::Exception("queryNamed: invalid property");

        string name = cstr(property);

        UserIds * ids = getShared(info.This());

        return v8::Boolean::New(ids->erase(name));
    }

    static v8::Handle<v8::Array>
    listNamed(const v8::AccessorInfo & info)
    {
        //cerr << "listNamed" << endl;
        HandleScope scope;
        try {
            UserIds * ids = getShared(info.This());

            int n = ids->size();

            v8::Handle<v8::Array> result = v8::Array::New(n);

            //cerr << "listNamed: " << ncol << " columns" << endl;

            unsigned i = 0;
            for (auto it = ids->begin(), end = ids->end();
                 it != end;  ++it,++i) {
                v8::Local<Integer> key = v8::Integer::New(i);
                v8::Handle<Value>  val = JS::toJS(it->first);
                result->Set(key, val);
            }
            
            return scope.Close(result);
        } catch (...) {
            cerr << "got exception in listNamed" << endl;
            return NULL_HANDLE;
        }
    }
};


/*****************************************************************************/
/* LOCATION JS                                                               */
/*****************************************************************************/

const char * LocationName = "Location";

struct LocationJS
    : public JSWrapped2<Location, LocationJS,
                        LocationName,
                        bidRequestModule> {

    LocationJS(v8::Handle<v8::Object> This,
                       const std::shared_ptr<Location> & list
                           = std::shared_ptr<Location>())
    {
        HandleScope scope;
        wrap(This, list);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new LocationJS(args.This(),
                           std::make_shared<Location>());
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
        registerRWProperty(&Location::countryCode, "countryCode");
        registerRWProperty(&Location::regionCode, "regionCode");
        registerRWProperty(&Location::cityName, "cityName");
        registerRWProperty(&Location::dma, "dma");
        registerRWProperty(&Location::timezoneOffsetMinutes,
                           "timezoneOffsetMinutes");
        registerMemberFn(&Location::toJson, "toJSON");
        registerMemberFn(&Location::fullLocationString,
                         "fullLocationString");
    }
};


/*****************************************************************************/
/* AD SPOT JS                                                                */
/*****************************************************************************/

const char * AdSpotName = "AdSpot";

struct AdSpotJS
    : public JSWrapped2<AdSpot, AdSpotJS, AdSpotName,
                        bidRequestModule> {

        AdSpotJS(v8::Handle<v8::Object> This,
              const std::shared_ptr<AdSpot> & as
                  = std::shared_ptr<AdSpot>())
    {
        HandleScope scope;
        wrap(This, as);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new AdSpotJS(args.This(), ML::make_std_sp(new AdSpot()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        registerRWProperty(&AdSpot::id, "id",
                           v8::ReadOnly | v8::DontDelete);
        registerRWProperty(&AdSpot::reservePrice, "reservePrice",
                           v8::ReadOnly | v8::DontDelete);

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("width"), widthsGetter,
                          0, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(ReadOnly | DontDelete));
        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("height"), heightsGetter,
                          0, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(ReadOnly | DontDelete));
    }

    static v8::Handle<v8::Value>
    widthsGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            const auto & f = getShared(info.This())->formats;
            vector<int> v2;
            for (unsigned i = 0;  i < f.size();  ++i)
                v2.push_back(f[i].width);
            return JS::toJS(v2);
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    heightsGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            const auto & f = getShared(info.This())->formats;
            vector<int> v2;
            for (unsigned i = 0;  i < f.size();  ++i)
                v2.push_back(f[i].height);
            return JS::toJS(v2);
        } HANDLE_JS_EXCEPTIONS;
    }

};

void to_js(JS::JSValue & value, const std::shared_ptr<AdSpot> & as)
{
    value = AdSpotJS::toJS(as);
}

void to_js(JS::JSValue & value, const AdSpot & as)
{
    to_js(value, ML::make_std_sp(new AdSpot(as)));
}

std::shared_ptr<AdSpot>
from_js(const JSValue & value, std::shared_ptr<AdSpot> *)
{
    return AdSpotJS::fromJS(value);
}

AdSpot
from_js(const JSValue & value, AdSpot *)
{
    return *AdSpotJS::fromJS(value).get();
}


/*****************************************************************************/
/* BID REQUEST JS                                                            */
/*****************************************************************************/

const char * BidRequestName = "BidRequest";

struct BidRequestJS
    : public JSWrapped2<BidRequest, BidRequestJS, BidRequestName,
                        bidRequestModule> {

    BidRequestJS(v8::Handle<v8::Object> This,
              const std::shared_ptr<BidRequest> & bid
                  = std::shared_ptr<BidRequest>())
    {
        HandleScope scope;
        wrap(This, bid);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            if (args.Length() > 0) {
                if (BidRequestJS::tmpl->HasInstance(args[0])) {
                    // Copy an existing bid request
                    std::shared_ptr<BidRequest> oldBr
                        = JS::fromJS(args[0]);
                    auto br = std::make_shared<BidRequest>();
                    *br = *oldBr;
                    new BidRequestJS(args.This(), br);
                    
                }
                else if (args[0]->IsString()) {
                    // Parse from a string
                    Utf8String request = getArg<Utf8String>(args, 0, "request");
                    string source = getArg<string>(args, 1, "datacratic", "source");
                    new BidRequestJS(args.This(),
                                     ML::make_std_sp(BidRequest::parse(source, request)));
                }
                else if (args[0]->IsObject()) {
                    // Parse from an object by going through JSON
                    Json::Value json = JS::fromJS(args[0]);
                    auto br = std::make_shared<BidRequest>();
                    *br = BidRequest::createFromJson(json);
                    new BidRequestJS(args.This(), br);
                }
                else throw ML::Exception("Cannot convert " + cstr(args[0])
                                         + " to BidRequest");
            } else {
                new BidRequestJS(args.This(), ML::make_std_sp(new BidRequest()));
            }
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        registerRWProperty(&BidRequest::auctionId, "id", v8::DontDelete);
        registerRWProperty(&BidRequest::timestamp, "timestamp", v8::DontDelete);
        registerRWProperty(&BidRequest::isTest, "isTest", v8::DontDelete);
        registerRWProperty(&BidRequest::url, "url", v8::DontDelete);
        registerRWProperty(&BidRequest::meta, "meta", v8::DontDelete);
        registerRWProperty(&BidRequest::ipAddress, "ipAddress", v8::DontDelete);
        registerRWProperty(&BidRequest::userAgent, "userAgent", v8::DontDelete);
        registerRWProperty(&BidRequest::language, "language", v8::DontDelete);
        registerRWProperty(&BidRequest::protocolVersion,
                           "protocolVersion", v8::DontDelete);
        registerRWProperty(&BidRequest::exchange, "exchange", v8::DontDelete);
        registerRWProperty(&BidRequest::provider, "provider", v8::DontDelete);
        registerRWProperty(&BidRequest::spots, "spots", v8::DontDelete);
        
        //registerRWProperty(&BidRequest::winSurcharge, "winSurchage",
        //                   v8::DontDelete);
        
        // TODO: these should go...
        registerRWProperty(&BidRequest::creative, "creative", v8::DontDelete);
        registerMemberFn(&BidRequest::toJson, "toJSON");

        NODE_SET_PROTOTYPE_METHOD(t, "getSegmentsFromSource",
                                  getSegmentsFromSource);

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("segments"), segmentsGetter,
                          segmentsSetter, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(DontDelete));

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("restrictions"), restrictionsGetter,
                          restrictionsSetter, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(DontDelete));

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("userIds"), userIdsGetter,
                          userIdsSetter, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(DontDelete));

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("location"), locationGetter,
                          locationSetter, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(DontDelete));
    }

    static Handle<v8::Value>
    getSegmentsFromSource(const Arguments & args)
    {
        try {
            string segmentProvider = getArg(args, 0, "segmentProvider");
            auto sh = getShared(args.This());
            auto it = sh->segments.find(segmentProvider);
            if (it == sh->segments.end())
                return v8::Null();
            return JS::toJS(it->second);
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    segmentsGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            v8::Handle<v8::Value> segs
                = SegmentsBySourceJS::toJS
                (ML::make_unowned_std_sp(getShared(info.This())->segments));
            SegmentsBySourceJS * wrapper
                = SegmentsBySourceJS::getWrapper(segs);
            wrapper->owner_ = getSharedPtr(info.This());
            return segs;
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    segmentsSetter(v8::Local<v8::String> property,
                  v8::Local<v8::Value> value,
                  const v8::AccessorInfo & info)
    {
        try {
            if (SegmentsBySourceJS::tmpl->HasInstance(value)) {
                getShared(info.This())->segments
                    = *SegmentsBySourceJS::getShared(value);
                return;
            }
            if (value->IsObject()) {
                map<std::string, std::shared_ptr<SegmentList> > segs;
                segs = from_js(JSValue(value), &segs);
                getShared(info.This())->segments = SegmentsBySource(segs);
                return;
            }
            throw ML::Exception("can't convert " + cstr(value)
                                + " into segment list");
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }

    static v8::Handle<v8::Value>
    restrictionsGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            v8::Handle<v8::Value> segs
                = SegmentsBySourceJS::toJS
                (ML::make_unowned_std_sp(getShared(info.This())->restrictions));
            SegmentsBySourceJS * wrapper
                = SegmentsBySourceJS::getWrapper(segs);
            wrapper->owner_ = getSharedPtr(info.This());
            return segs;
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    restrictionsSetter(v8::Local<v8::String> property,
                       v8::Local<v8::Value> value,
                       const v8::AccessorInfo & info)
    {
        try {
            if (SegmentsBySourceJS::tmpl->HasInstance(value)) {
                getShared(info.This())->restrictions
                    = *SegmentsBySourceJS::getShared(value);
                return;
            }
            if (value->IsObject()) {
                map<std::string, std::shared_ptr<SegmentList> > segs;
                segs = from_js(JSValue(value), &segs);
                getShared(info.This())->restrictions = SegmentsBySource(segs);
                return;
            }
            throw ML::Exception("can't convert " + cstr(value)
                                + " into segment list");
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }

    static v8::Handle<v8::Value>
    userIdsGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            auto owner = getSharedPtr(info.This());
            return UserIdsJS::toJS(owner->userIds, owner);
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    userIdsSetter(v8::Local<v8::String> property,
                       v8::Local<v8::Value> value,
                       const v8::AccessorInfo & info)
    {
        try {
            if (UserIdsJS::tmpl->HasInstance(value)) {
                getShared(info.This())->userIds
                    = *UserIdsJS::getShared(value);
                return;
            }
            if (value->IsObject()) {
                map<std::string, Id> ids;
                ids = from_js(JSValue(value), &ids);
                auto & uids = getShared(info.This())->userIds;
                for (auto it = ids.begin(), end = ids.end();
                     it != end;  ++it)
                    uids.add(it->second, it->first);
                return;
            }
            throw ML::Exception("can't convert " + cstr(value)
                                + " into segment list");
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }

    static v8::Handle<v8::Value>
    locationGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            auto owner = getSharedPtr(info.This());
            return LocationJS::toJS(owner->location, owner);
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    locationSetter(v8::Local<v8::String> property,
                       v8::Local<v8::Value> value,
                       const v8::AccessorInfo & info)
    {
        try {
            if (LocationJS::tmpl->HasInstance(value)) {
                getShared(info.This())->location
                    = *LocationJS::getShared(value);
                return;
            }
            if (value->IsObject()) {
                Json::Value json = JS::fromJS(value);
                getShared(info.This())->location
                    = Location::createFromJson(json);
                return;
            }
            throw ML::Exception("can't convert " + cstr(value)
                                + " into location info");
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }

};

std::shared_ptr<BidRequest>
from_js(const JSValue & value, std::shared_ptr<BidRequest> *)
{
    return BidRequestJS::fromJS(value);
}

BidRequest *
from_js(const JSValue & value, BidRequest **)
{
    return BidRequestJS::fromJS(value).get();
}

std::shared_ptr<BidRequest>
from_js_ref(const JSValue & value, std::shared_ptr<BidRequest> *)
{
    return BidRequestJS::fromJS(value);
}

void to_js(JS::JSValue & value, const std::shared_ptr<BidRequest> & br)
{
    value = BidRequestJS::toJS(br);
}


std::shared_ptr<BidRequest>
getBidRequestSharedPointer(const JS::JSValue & value)
{
    if(BidRequestJS::tmpl->HasInstance(value))
    {
        std::shared_ptr<BidRequest> br = BidRequestJS::getSharedPtr(value);
        return br;
    }
    std::shared_ptr<BidRequest> br;
    return br;
}


extern "C" void
init(Handle<v8::Object> target)
{
    Datacratic::JS::registry.init(target, bidRequestModule);
}

} // namespace JS
} // namespace Datacratic
