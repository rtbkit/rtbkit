#include "v8.h"
#include "soa/service/s3.h"
#include "soa/js/js_wrapped.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_call.h"
#include "soa/js/js_registry.h"
#include "soa/js/js_value.h"
#include "soa/js/js_wrapped.h"

using namespace v8;
using namespace std;
using namespace node;

namespace Datacratic{
namespace JS{


extern const char * const S3ApiModule;
const char * const S3ApiModule = "s3";
const char * S3ApiName = "S3Api";

struct S3ApiJS : public JSWrapped2<S3Api, S3ApiJS, S3ApiName, S3ApiModule>{
    S3ApiJS(v8::Handle<v8::Object> This,
        const std::shared_ptr<S3Api>& store = std::shared_ptr<S3Api>())
    {
        HandleScope scope;
        wrap(This, store);
    }

    static Handle<v8::Value> New(const Arguments & args){
        try{
            S3Api* storePtr;
            if(args.Length() == 0){
                storePtr = new S3Api();
            }else if(args.Length() == 2){
                storePtr = new S3Api(getArg<string>(args, 0, ""), getArg<string>(args, 1, ""));
            }else{
                cerr << "Undefined S3Api constructor" << endl;
                throw ML::Exception("undefined S3Api constructor");
            }
            new S3ApiJS(args.This(), ML::make_std_sp(storePtr));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize(){
        Persistent<FunctionTemplate> t = Register(New);
        registerMemberFn(&S3Api::downloadToFile, "downloadToFile");
        registerMemberFn(&S3Api::setDefaultBandwidthToServiceMbps, "setDefaultBandwidthToServiceMbps");
    }
};


Handle<v8::Value>
registerS3BucketJS(const Arguments & args)
{
    try {
        std::string bucketName = getArg(args, 0, "bucketName");
        std::string accessKeyId = getArg(args, 1, "accessKeyId");
        std::string accessKey = getArg(args, 2, "accessKey");

        registerS3Bucket(bucketName, accessKeyId, accessKey);
        return NULL_HANDLE;
    } HANDLE_JS_EXCEPTIONS;
}

extern "C" void
init(Handle<v8::Object> target){
    Datacratic::JS::registry.init(target, S3ApiModule);

    target->Set(String::NewSymbol("registerS3Bucket"),
        v8::Persistent<FunctionTemplate>::New
        (v8::FunctionTemplate::New(registerS3BucketJS))->GetFunction());
}




}//namespace JS
}//namespace Datacratic
