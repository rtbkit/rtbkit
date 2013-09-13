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
#include <tuple>
#include <city.h>
#include <boost/asio.hpp>
#include <boost/scoped_ptr.hpp>

#if 1 //  __GNUC_MINOR__ <= 7
#define USE_BOOST_TSS
#endif

#ifdef USE_BOOST_TSS
#include <boost/thread/tss.hpp>
#endif


namespace RTBKIT {

typedef std::tuple<const char*,std::string,uint32_t> ProbeCtx;

struct Span
{
    std::string              tag_ ;
    uint32_t                 id_, pid_;
    std::chrono::steady_clock::time_point start_, end_;
    Span(uint32_t id, uint32_t pid) : id_ (id), pid_(pid) {}
    Span() : Span(0,0) {}
    Span& operator=(const Span&) =delete;
};
typedef std::function<void(const ProbeCtx&, const std::vector<Span>&)> SinkCb;


namespace detail
{
// this is our structure.
typedef std::unordered_map<
size_t,
std::pair<std::stack<Span>,std::vector<Span>>
> ProbeStacks;
#ifndef USE_BOOST_TSS
static thread_local ProbeStacks PSTACKS;
#else
static boost::thread_specific_ptr<ProbeStacks> PSTACKS;
#endif

class MulticastSender
{
public:
    MulticastSender(const boost::asio::ip::address& multicast_addr,
                    const unsigned short multicast_port)
        : ep_(multicast_addr, multicast_port)
    {
        socket_.reset(new boost::asio::ip::udp::socket(svc_, ep_.protocol()));
    }

    ~MulticastSender()
    {
        socket_.reset(NULL);
    }

public:
    void send_data(const std::string& msg)
    {
        socket_->send_to( boost::asio::buffer(msg), ep_);
    }

private:
    boost::asio::ip::udp::endpoint                  ep_;
    boost::scoped_ptr<boost::asio::ip::udp::socket> socket_;
    boost::asio::io_service                         svc_;
};
}

static
detail::MulticastSender
MC_(boost::asio::ip::address::from_string("234.2.3.4"), 30001);


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
        if (!spans_->first.empty())
            sp.pid_ = spans_->first.top().id_;
        sp.id_ = sp.pid_ + 1;
        sp.start_ = std::chrono::steady_clock::now () ;
        spans_->first.emplace (sp);
    }

    ~Trace ()
    {
        if (!probed_) return;
        spans_->first.top().end_ = std::chrono::steady_clock::now () ;
        spans_->second.emplace_back (spans_->first.top());
        spans_->first.pop() ;
        if (spans_->first.empty ())
        {
            using detail::PSTACKS;
            if (S_sink_)
                S_sink_(pctx_, spans_->second);
#ifdef BOOST_USE_TSS
            (*PSTACKS.get()).erase(key_));
#else
            PSTACKS->erase(key_);
#endif
        }
    }

    static SinkCb                                  S_sink_;
    std::tuple<const char*,std::string,uint32_t>   pctx_ ; // probe ctx
    size_t                                         key_ ;
    bool                                           probed_ ;
    std::pair<std::stack<Span>,std::vector<Span>>* spans_;  // our entry in PSTACKS
};

// template <typename T>
// SinkCb
// Trace<T>::S_sink_ = nullptr;

} // RTBKIT;


#endif /* NPROBE_H_ */
