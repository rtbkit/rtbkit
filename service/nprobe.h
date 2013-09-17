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



namespace RTBKIT
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
    uint32_t                 id_, pid_;
    clock_type::time_point start_, end_;
    Span(uint32_t id = 0, uint32_t pid = 0) : id_ (id), pid_(pid) {}
};


/******************************************************************************/
/* SINK                                                                       */
/******************************************************************************/

typedef std::tuple<const char*,std::string,uint32_t> ProbeCtx;

// default sink (see nprobe.cc)
extern void syslog_probe_sink(const RTBKIT::ProbeCtx& ctx, const std::vector<RTBKIT::Span>& vs);

typedef std::function<void(const ProbeCtx&, const std::vector<Span>&)> SinkCb;



/******************************************************************************/
/* TRACE                                                                      */
/******************************************************************************/

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
        key_ = CityHash64(std::get<1>(pctx_).c_str(),std::get<1>(pctx_).size());
        probed_ = false;
        spans_ = nullptr;

        if (key_ % std::get<2>(pctx_)) return ;
        probed_ = true;
        if (!PSTACKS.get()) PSTACKS.create();
        spans_ = &(*PSTACKS.get())[key_];
        Span sp;
        sp.tag_   = tag;
        if (!std::get<1>(*spans_).empty())
            sp.pid_ = std::get<1>(*spans_).top().id_;
        else
            sp.pid_ = 0;
        sp.id_ = ++std::get<0>(*spans_);
        sp.start_ = clock_type::now () ;
        std::get<1>(*spans_).emplace (sp);
    }

    ~Trace ()
    {
        if (!probed_) return;
        std::get<1>(*spans_).top().end_ = clock_type::now () ;
        std::get<2>(*spans_).emplace_back (std::move(std::get<1>(*spans_).top()));
        std::get<1>(*spans_).pop() ;
        if (std::get<1>(*spans_).empty ())
        {
            if (S_sink_)
                S_sink_(pctx_, std::get<2>(*spans_));
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


} // RTBKIT;


#endif /* NPROBE_H_ */
