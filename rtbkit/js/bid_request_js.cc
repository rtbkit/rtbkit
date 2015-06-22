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
#include "currency_js.h"
#include <boost/make_shared.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "rtbkit/openrtb/openrtb_parsing.h"


using namespace std;
using namespace v8;
using namespace node;

namespace Datacratic {

struct JSConverters {
    std::function<void (void *, const JS::JSValue &)> fromJs;
    std::function<v8::Handle<v8::Value> (const void *, std::shared_ptr<void>)> toJs;
};

namespace JS {


const char * const bidRequestModule = "bid_request";
//so we can do require("standalone_demo")

void to_js(JS::JSValue & value, const Format & f)
{
    value = JS::toJS(f.print());
}


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
            ->SetIndexedPropertyHandler(getIndexed, setIndexed, queryIndexed,
                                        deleteIndexed, listIndexed);
        t->InstanceTemplate()
            ->SetNamedPropertyHandler(getNamed, setNamed, queryNamed,
                                      deleteNamed, listNamed);
    }

    static v8::Handle<v8::Value>
    getIndexed(uint32_t index, const v8::AccessorInfo & info)
    {
        try {
            SegmentsBySource * segs = getShared(info.This());

            string strIdx = to_string(index);
            return (segs->count(strIdx) > 0
                    ? JS::toJS(segs->at(strIdx))
                    : NULL_HANDLE);
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
        SegmentsBySource * segs = getShared(info.This());

        string strIdx = to_string(index);
        return (segs->count(strIdx) > 0
                ? v8::Integer::New(ReadOnly | DontDelete)
                : NULL_HANDLE);
    }

    static v8::Handle<v8::Array>
    listIndexed(const v8::AccessorInfo & info)
    {
        v8::HandleScope scope;
        SegmentsBySource * segs = getShared(info.This());

        int sz = segs->size();
        v8::Handle<v8::Array> result(v8::Array::New(sz));

        return scope.Close(result);
    }

    static v8::Handle<v8::Boolean>
    deleteIndexed(uint32_t index,
                  const v8::AccessorInfo & info)
    {
        return NULL_HANDLE;
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

// To/from JS goes via JSON for the moment...
template<typename Obj, typename Base>
struct PropertyAccessViaJson {
    static v8::Handle<v8::Value>
    getter(v8::Local<v8::String> property,
           const v8::AccessorInfo & info)
    {
        try {
            const ValueDescription * vd
                = reinterpret_cast<const ValueDescription *>
                (v8::External::Unwrap(info.Data()));
            Obj * o = Base::getShared(info.This());
            const StructureDescriptionBase::FieldDescription & fd
                = vd->getField(cstr(property));
            StructuredJsonPrintingContext context;
            fd.description->printJson(addOffset(o, fd.offset), context);
            return JS::toJS(context.output);
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    setter(v8::Local<v8::String> property,
           v8::Local<v8::Value> value,
           const v8::AccessorInfo & info)
    {
        try {
            const ValueDescription * vd
                = reinterpret_cast<const ValueDescription *>
                (v8::External::Unwrap(info.Data()));
            Obj * o = Base::getShared(info.This());
            const StructureDescriptionBase::FieldDescription & fd
                = vd->getField(cstr(property));
            Json::Value val = JS::fromJS(JSValue(value));
            StructuredJsonParsingContext context(val);
            fd.description->parseJson(addOffset(o, fd.offset), context);
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }
};

#if 0
struct WrappedStructureJS: public JS::JSWrapped {
    const ValueDescription * desc;
};
#endif

void
setFromJs(void * field,
          const JSValue & value,
          const ValueDescription & desc);

v8::Handle<v8::Value>
getFromJs(const void * field,
          const ValueDescription & desc,
          const std::shared_ptr<void> & owner);

struct WrappedArrayJS: public JSWrappedBase {
    const ValueDescription * desc;
    void * value;  // value being read

    static v8::Persistent<v8::FunctionTemplate> tmpl;
    
    WrappedArrayJS(v8::Handle<v8::Object> This,
                   void * value = 0,
                   std::shared_ptr<void> owner = nullptr)
    {
        wrap(This, 64 /* bytes */, typeid(*this));
        this->value = value;
        this->owner_ = owner;
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new WrappedArrayJS(args.This(), nullptr);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        tmpl = RegisterBase("WrappedArrayJS", "bonus", New);
        
        tmpl->InstanceTemplate()
            ->SetIndexedPropertyHandler(getIndexed, setIndexed, queryIndexed,
                                        deleteIndexed, listIndexed);
        
        tmpl->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("length"), lengthGetter,
                          0, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(ReadOnly | DontEnum | DontDelete));
    }

    static WrappedArrayJS * getWrapper(const v8::Handle<v8::Object> & object)
    {
        return unwrap<WrappedArrayJS>(object);
    }

    static v8::Handle<v8::Value>
    lengthGetter(v8::Local<v8::String> property,
                 const AccessorInfo & info)
    {
        try {
            WrappedArrayJS * wrapper = getWrapper(info.This());
            size_t size = wrapper->desc->getArrayLength(wrapper->value);
            return JS::toJS(size);
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    getIndexed(uint32_t index, const v8::AccessorInfo & info)
    {
        v8::HandleScope scope;
        try {
            WrappedArrayJS * wrapper = getWrapper(info.This());
            size_t size = wrapper->desc->getArrayLength(wrapper->value);
            
            if (index >= size)
                return v8::Undefined();

            void * element = wrapper->desc->getArrayElement(wrapper->value, index);
            
            return scope.Close(getFromJs(element, wrapper->desc->contained(),
                                         wrapper->owner_));
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    setIndexed(uint32_t index,
               v8::Local<v8::Value> value,
               const v8::AccessorInfo & info)
    {
        try {
            throw ML::Exception("setIndexed not done yet");
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Integer>
    queryIndexed(uint32_t index,
                 const v8::AccessorInfo & info)
    {
        WrappedArrayJS * wrapper = getWrapper(info.This());
        size_t size = wrapper->desc->getArrayLength(wrapper->value);

        if (index < size)
            return v8::Integer::New(v8::ReadOnly | v8::DontDelete);
        
        return NULL_HANDLE;
    }
    
    static v8::Handle<v8::Boolean>
    deleteIndexed(uint32_t index,
                  const v8::AccessorInfo & info)
    {
        return NULL_HANDLE;
    }

    static v8::Handle<v8::Array>
    listIndexed(const v8::AccessorInfo & info)
    {
        v8::HandleScope scope;

        WrappedArrayJS * wrapper = getWrapper(info.This());
        size_t sz = wrapper->desc->getArrayLength(wrapper->value);

        v8::Handle<v8::Array> result(v8::Array::New(sz));

        for (unsigned i = 0;  i < sz;  ++i) {
            result->Set(v8::Uint32::New(i),
                        v8::Uint32::New(i));
        }
        
        return scope.Close(result);
    }
};

v8::Persistent<v8::FunctionTemplate>
WrappedArrayJS::
tmpl;

// Wrap an array of values where elements are got or set in their entirity
struct WrappedValueArrayJS: public JSWrappedBase {
    const ValueDescription * desc;
};

// Wrap an array of fundamental types that can be directly mapped by the
// Javascript runtime
struct WrappedFundamentalValueArrayJS: public JSWrappedBase {
    const ValueDescription * desc;
};

struct WrappedStructureJS: public JSWrappedBase {
    const ValueDescription * desc;
    void * value;  // value being read

    static v8::Persistent<v8::FunctionTemplate> tmpl;
    
    WrappedStructureJS(v8::Handle<v8::Object> This,
                   void * value = 0,
                   std::shared_ptr<void> owner = nullptr)
    {
        wrap(This, 64 /* bytes */, typeid(*this));
        this->value = value;
        this->owner_ = owner;
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new WrappedStructureJS(args.This(), nullptr);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        tmpl = RegisterBase("WrappedStructureJS", "bonus", New);
        
        tmpl->InstanceTemplate()
            ->SetNamedPropertyHandler(getNamed, setNamed, queryNamed,
                                      deleteNamed, listNamed);
    }

    static WrappedStructureJS * getWrapper(const v8::Handle<v8::Object> & object)
    {
        return unwrap<WrappedStructureJS>(object);
    }

    static v8::Handle<v8::Value>
    getNamed(v8::Local<v8::String> property,
             const v8::AccessorInfo & info)
    {
        try {
            string name = cstr(property);

            WrappedStructureJS * wrapper = getWrapper(info.This());
            const ValueDescription::FieldDescription * fd
                = wrapper->desc->hasField(wrapper->value, name);
            
            if (!fd)
                return NULL_HANDLE;

            return getFromJs(addOffset(wrapper->value, fd->offset),
                             *fd->description,
                             wrapper->owner_);

        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    setNamed(v8::Local<v8::String> property,
             v8::Local<v8::Value> value,
             const v8::AccessorInfo & info)
    {
        try {
            string name = cstr(property);

            WrappedStructureJS * wrapper = getWrapper(info.This());
            const ValueDescription::FieldDescription & fd
                = wrapper->desc->getField(name);
            
            setFromJs(addOffset(wrapper->value, fd.offset), value,
                      *fd.description);

            return v8::Undefined();
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Integer>
    queryNamed(v8::Local<v8::String> property,
               const v8::AccessorInfo & info)
    {
        string name = cstr(property);
        WrappedStructureJS * wrapper = getWrapper(info.This());
        if (!wrapper->desc->hasField(wrapper->value, name))
            return NULL_HANDLE;

        return v8::Integer::New(DontDelete);
    }

    static v8::Handle<v8::Boolean>
    deleteNamed(v8::Local<v8::String> property,
                const v8::AccessorInfo & info)

    {
        return v8::False();
    }

    static v8::Handle<v8::Array>
    listNamed(const v8::AccessorInfo & info)
    {
        try {
            HandleScope scope;
            WrappedStructureJS * wrapper = getWrapper(info.This());
            int numFields = wrapper->desc->getFieldCount(wrapper->value);
            
            int i = 0;
            v8::Handle<v8::Array> result = v8::Array::New(numFields);
            
            auto onField = [&] (const ValueDescription::FieldDescription & fd)
                {
                    result->Set(v8::Uint32::New(i++), JS::toJS(fd.fieldName));
                };
            
            wrapper->desc->forEachField(wrapper->value, onField);

            return scope.Close(result);
        } catch (const std::exception & exc) {
            cerr << "got exception in listNamed" << endl;
            cerr << exc.what() << endl;
            cerr << cstr(info.This()) << endl;
            //backtrace();
            return NULL_HANDLE;
        }
    }

};

v8::Persistent<v8::FunctionTemplate>
WrappedStructureJS::
tmpl;

namespace {

struct Init {
    Init()
    {
        registry.introduce("WrappedArrayJS", "bonus", WrappedArrayJS::Initialize);
        registry.introduce("WrappedStructureJS", "bonus", WrappedStructureJS::Initialize);
    }
} init;

} // file scope

void
initJsConverters(const ValueDescription & desc)
{
    if (desc.jsConverters || desc.jsConvertersInitialized)
        return;
    
    std::unique_ptr<JSConverters> converters
        (new JSConverters);

    // Take a pointer so we bind over the pointer and copy it, not the
    // description
    auto descPtr = &desc;

    //cerr << "***** desc.kind = " << desc.kind << " for "
    //     << desc.typeName << endl;

    // Is it a structure?
    if (desc.kind == ValueKind::STRUCTURE) {
        //cerr << "got structure " << desc.typeName << endl;

        // Return structure-based converters
        converters->fromJs = [=] (void * field, const JSValue & val)
            {
                // Is it an object of the correct type?

                if (!val->IsObject()) {
                    throw ML::Exception("attempt to create structure from non-object "
                                        + cstr(val));
                }
#if 0
                if (WrappedStructureJS::tmpl->HasInstance(val)) {
                    // Copy element by element
                }
#endif

                // Firstly, clear all of the fields to their default value
                descPtr->setDefault(field);

                v8::HandleScope scope;
                auto objPtr = v8::Object::Cast(*val);
                
                v8::Local<v8::Array> properties = objPtr->GetOwnPropertyNames();

                for (int i = 0; i < properties->Length(); ++i) {
                    auto keyVal = properties->Get(i);
                    string fieldName = cstr(keyVal);
                    v8::Local<v8::Value> fieldVal = objPtr->Get(keyVal);

                    auto * fd = descPtr->hasField(field, fieldName);
                    if (!fd) {
                        Json::Value val = JS::fromJS(fieldVal);
                        // TODO: some kind of on-unknown-field function

                        cerr << "got unknown JS field " << fieldName << " = "
                             << val.toString() << endl;
                        
                        continue;
                    }

                    setFromJs(addOffset(field, fd->offset), fieldVal,
                              *fd->description);
                }
            };

        converters->toJs = [=] (const void * field, std::shared_ptr<void> owner)
            {
                //cerr << "to JS structure for " << descPtr->typeName << endl;

                // Wrap it in a wrapped array object
                v8::Local<v8::Object> result
                    = WrappedStructureJS::tmpl->GetFunction()->NewInstance();
                auto wrapper = WrappedStructureJS::getWrapper(result);
                wrapper->value = (void *)field;
                wrapper->desc = descPtr;
                wrapper->owner_ = owner;

                return result;
            };
    }

    // Is it an array?
    if (desc.kind == ValueKind::ARRAY) {
        //cerr << "got array " << desc.typeName << endl;

        const ValueDescription * innerDesc = &descPtr->contained();
        
        // Convert each element of the array
        converters->fromJs = [=] (void * field, const JSValue & val)
            {
                //cerr << "from JS array" << endl;

                // Convert the entire lot in the JS into our type
                if (val->IsNull()) {
                    descPtr->setArrayLength(field, 0);
                    return;
                }

                // Is it an object of the correct type?
                if (val->IsObject()) {
                    if (WrappedArrayJS::tmpl->HasInstance(val)) {
                        //cerr << "Is a wrapped array" << endl;
                        auto wrapper = WrappedArrayJS::getWrapper(val);

                        // Same type?; do a direct copy
                        if (wrapper->desc->type == descPtr->type) {
                            descPtr->copyValue(wrapper->value, field);
                            return;
                        }

                        // Otherwise, copy element by element
                        size_t len = wrapper->desc->getArrayLength(wrapper->value);
                        
                        descPtr->setArrayLength(field, len);

                        auto & valContained = wrapper->desc->contained();

                        // Same inner type?
                        if (innerDesc->type == valContained.type) {
                            for (unsigned i = 0;  i < len;  ++i) {
                                innerDesc->copyValue(wrapper->desc->getArrayElement(wrapper->value, i),
                                                     descPtr->getArrayElement(field, i));
                            }
                            return;
                        }

                        for (unsigned i = 0;  i < len;  ++i) {
                            for (unsigned i = 0;  i < len;  ++i) {
                                innerDesc->convertAndCopy
                                (wrapper->desc->getArrayElement(wrapper->value, i),
                                 *wrapper->desc,
                                 descPtr->getArrayElement(field, i));
                            }
                        }

                        return;

                        cerr << "descPtr = " << descPtr
                             << "wrapper->desc = " << wrapper->desc << endl;
                        cerr << "descPtr = " << descPtr->typeName
                             << "wrapper->desc = " << wrapper->desc->typeName << endl;
                        cerr << "field = " << field << " wrapper->value = "
                             << wrapper->value << endl;

                        // Copy element by element
                    }
                }

                if(!val->IsArray()) {
                    throw ML::Exception("invalid JSValue for array extraction");
                }

                auto arrPtr = v8::Array::Cast(*val);

                descPtr->setArrayLength(field, arrPtr->Length());

                for(int i=0; i<arrPtr->Length(); ++i) {
                    auto val = arrPtr->Get(i);
                    setFromJs(descPtr->getArrayElement(field, i), val, *innerDesc);
                }
            };

        converters->toJs = [=] (const void * field, const std::shared_ptr<void> & owner)
            {
                //cerr << "to JS array for " << descPtr->typeName << endl;

                // Wrap it in a wrapped array object
                v8::Local<v8::Object> result
                    = WrappedArrayJS::tmpl->GetFunction()->NewInstance();
                auto wrapper = WrappedArrayJS::getWrapper(result);
                wrapper->value = (void *)field;
                wrapper->desc = descPtr;
                wrapper->owner_ = owner;

                return result;
            };
    }

    // Is it optional?
    if (desc.kind == ValueKind::OPTIONAL) {
        //cerr << "got optional " << desc.typeName << endl;

        const ValueDescription * innerDesc = &descPtr->contained();

        //cerr << "innerDesc = " << innerDesc << endl;

        // Return optional converters
        converters->fromJs = [=] (void * field, const JSValue & value)
            {
                //cerr << "optional value = " << cstr(value) << endl;
                // If the value is null, then we remove the optional value
                if (value->IsNull() || value->IsUndefined()) {
                    descPtr->setDefault(field);
                    return;
                }

                //cerr << "*** setting inner value" << endl;

                // Otherwise we get the inner value and set it
                setFromJs(descPtr->optionalMakeValue(field),
                          value,
                          *innerDesc);
            };

        converters->toJs = [=] (const void * field, std::shared_ptr<void> owner)
            -> v8::Handle<v8::Value>
            {
                // If the value is missing, we return null
                if (descPtr->isDefault(field))
                    return v8::Null();

                // Otherwise we return the inner value
                return getFromJs(descPtr->optionalGetValue(field),
                                 *innerDesc,
                                 owner);
            };
    }

    //   Does it have any parent classes?

    // Is it an arithmetic type?

    // Default goes through JSON
    if (!converters->fromJs) {
        converters->fromJs = [=] (void * field, const JSValue & value)
            {
                Json::Value val = JS::fromJS(value);
                StructuredJsonParsingContext context(val);
                descPtr->parseJson(field, context);
            };
    }

    if (!converters->toJs) {
        converters->toJs = [=] (const void * field, std::shared_ptr<void>)
            {
                StructuredJsonPrintingContext context;
                descPtr->printJson(field, context);
                return JS::toJS(context.output);
            };
    }

    desc.jsConverters = converters.release();
    desc.jsConvertersInitialized = true;
}

void
setFromJs(void * field,
          const JSValue & value,
          const ValueDescription & desc)
{
    if (!desc.jsConvertersInitialized) {
        initJsConverters(desc);
    }

    desc.jsConverters->fromJs(field, value);
}

v8::Handle<v8::Value>
getFromJs(const void * field,
          const ValueDescription & desc,
          const std::shared_ptr<void> & owner)
{
    if (!desc.jsConvertersInitialized) {
        initJsConverters(desc);
    }

    return desc.jsConverters->toJs(field, owner);
}


template<typename Obj, typename Base>
struct PropertyAccessViaDescription {
    static v8::Handle<v8::Value>
    getter(v8::Local<v8::String> property,
           const v8::AccessorInfo & info)
    {
        try {
            const ValueDescription * vd
                = reinterpret_cast<const ValueDescription *>
                (v8::External::Unwrap(info.Data()));
            auto p = Base::getSharedPtr(info.This());
            Obj * o = p.get();
            const StructureDescriptionBase::FieldDescription & fd
                = vd->getField(cstr(property));
            return getFromJs(addOffset(o, fd.offset), *fd.description, p);
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    setter(v8::Local<v8::String> property,
           v8::Local<v8::Value> value,
           const v8::AccessorInfo & info)
    {
        try {
            const ValueDescription * vd
                = reinterpret_cast<const ValueDescription *>
                (v8::External::Unwrap(info.Data()));
            Obj * o = Base::getShared(info.This());
            const StructureDescriptionBase::FieldDescription & fd
                = vd->getField(cstr(property));
            setFromJs(addOffset(o, fd.offset), value, *fd.description);
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }
};

template<typename Wrapper, typename T>
void registerFieldFromDescription(const StructureDescription<T> & desc,
                                  const std::string & fieldName)
{
    Wrapper::tmpl->InstanceTemplate()
        ->SetAccessor(v8::String::NewSymbol(fieldName.c_str()),
                      PropertyAccessViaDescription<T, Wrapper>::getter,
                      PropertyAccessViaDescription<T, Wrapper>::setter,
                      v8::External::Wrap((void *)&desc),
                      v8::DEFAULT,
                      v8::PropertyAttribute(v8::DontDelete));
}

/*****************************************************************************/
/* AD SPOT JS                                                                */
/*****************************************************************************/

extern const char * AdSpotName;
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

        static DefaultDescription<AdSpot> desc;

        for (auto & f: desc.fields) {
            const char * name = f.first;
            registerFieldFromDescription<JSWrapped2>(desc, name);
        }

        registerRWProperty(&AdSpot::id, "id",
                           v8::DontDelete);
        registerRWProperty(&AdSpot::reservePrice, "reservePrice",
                           v8::DontDelete);
        registerROProperty(&AdSpot::formats, "formats",
                           v8::ReadOnly | v8::DontDelete);

#if 0
        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("width"), widthsGetter,
                          0, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(ReadOnly | DontDelete));
        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("height"), heightsGetter,
                          0, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(ReadOnly | DontDelete));
#endif
    }

#if 0
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
#endif

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
                    //cerr << "copy existing" << endl;
                    // Copy an existing bid request
                    std::shared_ptr<BidRequest> oldBr
                        = JS::fromJS(args[0]);
                    auto br = std::make_shared<BidRequest>();
                    *br = *oldBr;
                    new BidRequestJS(args.This(), br);
                    
                }
                else if (args[0]->IsString()) {
                    // Parse from a string
                    //cerr << "parse string" << endl;
                    Datacratic::UnicodeString request = 
                         getArg<Datacratic::UnicodeString>(args, 0, "request");
                    string source = getArg<string>(args, 1, "datacratic", "source");
                    new BidRequestJS(args.This(),
                                     ML::make_std_sp(BidRequest::parse(source, request)));
                }
                else if (args[0]->IsObject()) {
                    //cerr << "parse object" << endl;
                    // Parse from an object by going through JSON
                    Json::Value json = JS::fromJS(args[0]);
                    //cerr << "JSON = " << json << endl;
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
        registerRWProperty(&BidRequest::timeAvailableMs, "timeAvailableMs", v8::DontDelete);
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
        registerRWProperty(&BidRequest::ext, "ext", v8::DontDelete);

        static DefaultDescription<BidRequest> desc;

        /// \deprecated spots member is there for backward compatibility
        registerFieldFromDescription<BidRequestJS>(desc, "spots");
        registerFieldFromDescription<BidRequestJS>(desc, "imp");
        registerFieldFromDescription<BidRequestJS>(desc, "app");
        registerFieldFromDescription<BidRequestJS>(desc, "device");
        registerFieldFromDescription<BidRequestJS>(desc, "user");
        registerFieldFromDescription<BidRequestJS>(desc, "imp");


        //registerRWProperty(&BidRequest::imp, "imp", v8::DontDelete);
        
        //registerRWProperty(&BidRequest::winSurcharge, "winSurchage",
        //                   v8::DontDelete);
        
        // TODO: these should go...
        //registerRWProperty(&BidRequest::creative, "creative", v8::DontDelete);
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
