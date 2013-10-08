/* -*- C++ -*-
 * nprobe.h
 *
 *  Created on: Sep 10, 2013
 *      Author: jan
 */

#ifndef NPROBE_H_
#define NPROBE_H_

#include "jml/arch/thread_specific.h"

#include <unordered_map>
#include <chrono>
#include <memory>
#include <functional>
#include <string>
#include <stack>
#include <vector>
#include <tuple>
#include <city.h>
#include <iostream>


namespace Datacratic
{

#define GCC_VERSION (__GNUC__ * 10000       \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)

#if GCC_VERSION >= 40700
    typedef std::chrono::steady_clock clock_type;
#else
    typedef std::chrono::monotonic_clock clock_type;
#endif

typedef std::tuple<const char*,std::string,uint32_t> ProbeCtx;


/******************************************************************************/
/* SPAN                                                                       */
/******************************************************************************/

struct Span
{
    std::string              tag_ ;
    uint32_t                 id_ ;
    int32_t                  pid_;
    clock_type::time_point start_, end_;
    Span(uint32_t id = 0, uint32_t pid = 0) : id_ (id), pid_(pid) {}
};


/******************************************************************************/
/* SINK                                                                       */
/******************************************************************************/

typedef std::tuple<const char*,std::string,uint32_t> ProbeCtx;

// default sink (see nprobe.cc)
extern void syslog_probe_sink(const ProbeCtx& ctx, const std::vector<Span>& vs);

typedef std::function<void(const ProbeCtx&, const std::vector<Span>&)> SinkCb;



/******************************************************************************/
/* TRACE                                                                      */
/******************************************************************************/

/**
  * Main Tracing class
  * 
  * Create an instance of this class to start tracing :
  *
  *    Trace<BidRequest> trace(br, "function");
  *
  * Traces can be nested within a given scope :
  *
  *    Trace<BidRequest> trace1(br, "function");
  *    
  *    // additional code
  *
  *    Trace<BidRequest> trace2(br, "function");
  *
  * make_trace<T> constructs an object of type Trace<T> and relies
  * on template argument deduction :
  *
  *     auto trace = make_trace(br, "function")
  *
  * The TRACE() macro auto-magically generates an unique name for
  * the trace object:
  *
  *     TRACE(br, "function")
  *
  * 
*/    

template <typename T>
class Trace
{
public:
    Trace (const std::shared_ptr<T>& object, const std::string& tag)
    {
        init(*object.get(), tag);
    }

    Trace (const T& object, const std::string& tag)
    {
        init(object, tag);
    }

    void init(const T &object, const std::string &tag)
    {
        pctx_ = do_probe(object);
        auto & uid = std::get<1>(pctx_);
        auto sampling_freq = std::get<2>(pctx_);
        key_ = CityHash64(uid.c_str(), uid.size());
        probed_ = false;
        spans_ = nullptr;

        if (key_ % sampling_freq) return ;

        probed_ = true;
        if (!PSTACKS.get()) PSTACKS.create();
        spans_ = &(*PSTACKS.get())[key_];
        Span sp;
        sp.tag_   = tag;

        auto &stack = std::get<1>(*spans_);
        if (!stack.empty())
            sp.pid_ = stack.top().id_;
        else
            sp.pid_ = -1;
        sp.id_ = ++std::get<0>(*spans_);
        sp.start_ = clock_type::now () ;
        stack.emplace (sp);
    }

    ~Trace ()
    {
        if (!probed_) return;
        auto &stack = std::get<1>(*spans_);
        auto &vector = std::get<2>(*spans_);
        stack.top().end_ = clock_type::now () ;
        vector.emplace_back (std::move(stack.top()));
        stack.pop() ;
        if (stack.empty ())
        {
            if (S_sink_)
                S_sink_(pctx_, vector);
            (*PSTACKS.get()).erase(key_);
        }
    }

    static SinkCb                                  S_sink_;
    static void set_sinkCb (SinkCb sink_cb)        {
        S_sink_ = sink_cb;
    }

    typedef std::tuple<int,std::stack<Span>, std::vector<Span>> pstack_t;

    std::tuple<const char*,std::string,uint32_t>   pctx_ ; // probe ctx
    size_t                                         key_ ;
    bool                                           probed_ ;
    pstack_t*                                      spans_;  // our entry in PSTACKS

    // this is our structure.
    typedef std::unordered_map<size_t,pstack_t> ProbeStacks;
    static ML::Thread_Specific<ProbeStacks> PSTACKS;
};

template<typename T>
Trace<T> make_trace(const T &object, const std::string &tag) {
    return Trace<T>(object, tag);
}

template<typename T>
Trace<T> make_trace(const std::shared_ptr<T> &object, const std::string &tag) {
    return Trace<T>(object, tag);
}

#define PREFIX_ trace__
#define CAT(a, b) a##b
#define LABEL_(a) CAT(PREFIX_, a)
#define UNIQUE_LABEL LABEL_(__LINE__)

#define TRACE(object, tag) \
    auto UNIQUE_LABEL = make_trace(object, tag); \
    (void) 0


#define TRACE_FN(object, tag, fn) \
    auto UNIQUE_LABEL = make_trace(object, tag); \
    UNIQUE_LABEL.set_sinkCb(fn); \
    (void) 0
    
template<typename T>
ML::Thread_Specific<typename Trace<T>::ProbeStacks>
Trace<T>::PSTACKS { };

template <typename T>
SinkCb
Trace<T>::S_sink_ = syslog_probe_sink ;

} // Datacratic


#endif /* NPROBE_H_ */
