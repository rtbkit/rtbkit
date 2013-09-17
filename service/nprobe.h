/*
 * nprobe.h
 *
 *  Created on: Sep 10, 2013
 *      Author: jan
 */

#ifndef NPROBE_H_
#define NPROBE_H_

#include <unordered_map>
#include <chrono>
#include <memory>
#include <functional>
#include <string>
#include <stack>
#include <vector>
#include <tuple>
#include <city.h>


#if 1 //  __GNUC_MINOR__ <= 7
#define USE_BOOST_TSS
#endif

#ifdef USE_BOOST_TSS
#include <boost/thread/tss.hpp>
#endif


namespace RTBKIT
{

typedef std::tuple<const char*,std::string,uint32_t> ProbeCtx;

//// base template
//template <typename X>
//ProbeCtx
//do_probe(X const&);

struct Span
{
    std::string              tag_ ;
    uint32_t                 id_, pid_;
    std::chrono::monotonic_clock::time_point start_, end_;
    Span(uint32_t id = 0, uint32_t pid = 0) : id_ (id), pid_(pid) {}
    Span& operator=(const Span&) =delete;
};
typedef std::function<void(const ProbeCtx&, const std::vector<Span>&)> SinkCb;


namespace detail
{
typedef std::tuple<int,std::stack<Span>, std::vector<Span>> pstack_t;
// this is our structure.
typedef std::unordered_map<size_t,pstack_t> ProbeStacks;
#ifndef USE_BOOST_TSS
static thread_local ProbeStacks PSTACKS;
#else
static boost::thread_specific_ptr<ProbeStacks> PSTACKS;
#endif

// default sink (see nprobe.cc)
extern void default_probe_sink(const RTBKIT::ProbeCtx& ctx, const std::vector<RTBKIT::Span>& vs);
}

template <typename T>
class Trace
{
public:
    Trace (const std::shared_ptr<T>& t, const std::string& tag)
        : Trace (*t.get(), tag)
    {

    }
    Trace (const T& t, const std::string& tag)
        : pctx_    (do_probe(t))
        , key_     (CityHash64(std::get<1>(pctx_).c_str(),std::get<1>(pctx_).size()))
        , probed_  (false)
        , spans_   (0)
    {
        if (key_ % std::get<2>(pctx_)) return ;
        probed_ = true;
        using detail::PSTACKS;
#ifdef USE_BOOST_TSS
        if (!PSTACKS.get())
            PSTACKS.reset (new detail::ProbeStacks());
        spans_ = &(*PSTACKS.get())[key_];
#else
        spans_ = &detail::PSTACKS[key_];
#endif
        Span sp;
        sp.tag_   = tag;
        if (!std::get<1>(*spans_).empty())
            sp.pid_ = std::get<1>(*spans_).top().id_;
        else
            sp.pid_ = 0;
        sp.id_ = ++std::get<0>(*spans_);
        sp.start_ = std::chrono::monotonic_clock::now () ;
        std::get<1>(*spans_).emplace (sp);
    }

    ~Trace ()
    {
        if (!probed_) return;
        std::get<1>(*spans_).top().end_ = std::chrono::monotonic_clock::now () ;
        std::get<2>(*spans_).emplace_back (std::get<1>(*spans_).top());
        std::get<1>(*spans_).pop() ;
        if (std::get<1>(*spans_).empty ())
        {
            using detail::PSTACKS;
            if (S_sink_)
                S_sink_(pctx_, std::get<2>(*spans_));
#ifdef BOOST_USE_TSS
            (*PSTACKS.get()).erase(key_));
#else
            PSTACKS->erase(key_);
#endif
        }
    }

    static SinkCb                                  S_sink_;
    static void set_sinkCb (SinkCb sink_cb)        {
        S_sink_ = sink_cb;
    }
    std::tuple<const char*,std::string,uint32_t>   pctx_ ; // probe ctx
    size_t                                         key_ ;
    bool                                           probed_ ;
    detail::pstack_t*                              spans_;  // our entry in PSTACKS
};

} // RTBKIT;


#endif /* NPROBE_H_ */
