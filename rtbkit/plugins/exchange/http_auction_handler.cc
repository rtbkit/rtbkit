/* http_auction_handler.cc
   Jeremy Barnes, 14 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Handler for auctions.
*/

#include "http_auction_handler.h"
#include "http_exchange_connector.h"

#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/guard.h"
#include "jml/utils/set_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/timers.h"
#include <set>

#include <boost/foreach.hpp>

using namespace std;
using namespace ML;

namespace RTBKIT {

/*****************************************************************************/
/* HTTP AUCTION LOGGER                                                       */
/*****************************************************************************/

HttpAuctionLogger::
HttpAuctionLogger(std::string const & filename, int count) : requestFilename(filename), requestLimit(count) {
    requestCount = requestFile = 0;
}

void
HttpAuctionLogger::
recordRequest(HttpHeader const & headers, std::string const & body) {
    std::lock_guard<std::mutex> guard(lock);
    if(!stream) {
        std::string filename = ML::format("%d-%s", requestFile, requestFilename);
        stream.open(filename);
    }

    stream << headers << body << std::endl;
    ++requestCount;
    if(requestCount == requestLimit) {
        stream.close();
        ++requestFile;
        requestCount = 0;
    }
}

void
HttpAuctionLogger::
close() {
    if(stream) {
        stream.close();
    }
}

unsigned
HttpAuctionLogger::
parse(const std::string & filename,
      const std::function<void(const std::string &)> & callback)
{
    cerr << "reading packets from " << filename << endl;

    filter_istream stream(filename);
    ML::Parse_Context context(filename, stream);

    int count = 0;

    while(context) {
        try {
            Parse_Context::Hold_Token hold(context);

            while (context) {
                Parse_Context::Revert_Token token(context);
                if (context.match_literal("POST")) break;
                token.ignore();
                context.expect_line();
            }

            std::string request = context.expect_text("\r") + "\r\n";
            context.expect_literal("\r\n");

            int contentLength = -1;

            while (!context.eof()) {
                string line;
                if (context.match_literal("Content-Length:")) {
                    context.skip_whitespace();
                    contentLength = context.expect_int();
                    line = ML::format("Content-Length: %d", contentLength);
                }
                else {
                    line = context.expect_text("\r\n");
                }

                request += line + "\r\n";
                context.expect_literal("\r\n");

                if (line.empty()) break;
            }

            if (contentLength == -1)
                throw ML::Exception("no content-length");

            for (unsigned i = 0;  i < contentLength;  ++i) {
                context.match_literal('\n');
                request += *context++;
            }

            context.match_eol();

            callback(request);
            ++count;
        }
        catch (const std::exception & exc) {
            std::cerr << context.where() << ": got exception: "
                      << exc.what() << std::endl;
            break;
        }
    }

    return count;
}

/*****************************************************************************/
/* HTTP AUCTION HANDLER                                                      */
/*****************************************************************************/

long HttpAuctionHandler::created = 0;
long HttpAuctionHandler::destroyed = 0;

HttpAuctionHandler::
HttpAuctionHandler()
    : hasTimer(false), disconnected(false), servingRequest(false)
{
    atomic_add(created, 1);
}

HttpAuctionHandler::
~HttpAuctionHandler()
{
    if (servingRequest)
        ML::atomic_add(endpoint->numServingRequest, -1);
    //cerr << "deleting HttpAuctionHandler at " << this << endl;
    //backtrace();
    atomic_add(destroyed, 1);
    if (hasTimer)
        throw Exception("deleted auction handler with timer");
}

void
HttpAuctionHandler::
onGotTransport()
{
    transport().activities.limit(5);
    //transport().activities.clear();

    addActivityS("gotTransport");

    //cerr << "bid handler got transport" << endl;

    this->endpoint = dynamic_cast<HttpExchangeConnector *>(get_endpoint());
    if (!this->endpoint)
        throw Exception("HttpAuctionHandler needs to be owned by an "
                        "HttpExchangeConnector");

    HttpConnectionHandler::onGotTransport();

    startReading();
}

void
HttpAuctionHandler::
handleDisconnect()
{
    doEvent("auctionDisconnection");

    disconnected = true;

    closeWhenHandlerFinished();
}

void
HttpAuctionHandler::
handleTimeout(Date date, size_t cookie)
{
    checkMagic();

    std::shared_ptr<Auction> auction_ = auction;
    if (auction->tooLate()) return;  // was sent before timeout

#if 0
    cerr << "auction " << auction_->id << " timed out with tooLate "
         << auction->tooLate() << " and response "
         << auction->getResponseJson().toString() << endl;
#endif

    if (!auction_) {
        cerr << "doAuctionTimeout with no auction" << endl;
        abort();
    }

    if (auction_->finish()) {
        doEvent("auctionTimeout");
        if (this->endpoint->onTimeout)
            this->endpoint->onTimeout(auction, date);
    }
}

void
HttpAuctionHandler::
cancelTimer()
{
    if (!hasTimer) return;

    //cerr << "cancelling timer" << endl;

    addActivityS("cancelTimer");
    ConnectionHandler::cancelTimer();
    hasTimer = false;
}

void
HttpAuctionHandler::
onDisassociate()
{
    //cerr << "onDisassociate" << endl;

    cancelTimer();

    /* We need to make sure that the handler doesn't try to send us
       anything. */
    if (auction && !auction->tooLate()) {
        auction->isZombie = true;
        doEvent("disconnectWithActiveAuction");
        //transport().activities.dump();
        //cerr << "disassociation of HttpAuctionHandler " << this
        //     << " when auction not finished"
        //     << endl;
        //backtrace();
        //throw Exception("attempt to disassociate HttpAuctionHandler when not "
        //                "too late: error %s", error.c_str());
    }

    endpoint->finishedWithHandler(shared_from_this());
}

void
HttpAuctionHandler::
onCleanup()
{
    onDisassociate();
}

void
HttpAuctionHandler::
doEvent(const char * eventName,
        StatEventType type,
        float value,
        const char * units,
        std::initializer_list<int> extra)
{
    //cerr << eventName << " " << value << endl;
    endpoint->recordEvent(eventName, type, value, extra);
}

void
HttpAuctionHandler::
incNumServingRequest()
{
    ML::atomic_add(endpoint->numServingRequest, 1);
}

void
HttpAuctionHandler::
handleHttpPayload(const HttpHeader & header,
                  const std::string & payload)
{
    // Unknown resource?  Handle it...
    //cerr << header.verb << " " << header.resource << endl;
    //cerr << header << endl;
    //cerr << payload << endl;
    //cerr << endpoint->auctionVerb << " " << endpoint->auctionResource << endl;

    if (header.resource != endpoint->auctionResource
        || header.verb != endpoint->auctionVerb) {
        endpoint->handleUnknownRequest(*this, header, payload);
        return;
    }

    if(logger) {
        logger->recordRequest(header, payload);
    }

    ML::atomic_add(endpoint->numRequests, 1);

    doEvent("auctionReceived");
    doEvent("auctionBodyLength", ET_OUTCOME, payload.size(), "bytes");

    incNumServingRequest();
    servingRequest = true;

    addActivityS("handleHttpPayload");

    stopReading();

    // First check if we are authorized to bid.  If not we drop the auction
    // with prejudice.
    Date now = Date::now();

    if (!endpoint->isEnabled(now)) {
        doEvent("auctionEarlyDrop.notEnabled");
        dropAuction("endpoint not enabled");
        return;
    }
    
    double acceptProbability = endpoint->acceptAuctionProbability;

    if (acceptProbability < 1.0
        && random() % 1000000 > 1000000 * acceptProbability) {
        // early drop...
        doEvent("auctionEarlyDrop.randomEarlyDrop");
        dropAuction("random early drop");
        return;
    }

    double timeAvailableMs = getTimeAvailableMs(header, payload);
    double networkTimeMs = getRoundTripTimeMs(header);

    doEvent("auctionStartLatencyMs",
            ET_OUTCOME,
            now.secondsSince(endpoint->getStartTime()) * 1000.0, "ms");

    doEvent("auctionTimeAvailableMs",
            ET_OUTCOME,
            timeAvailableMs, "ms");


    if (timeAvailableMs - networkTimeMs < 5.0) {
        // Do an early drop of the bid request without even creating an
        // auction

        doEvent("auctionEarlyDrop.timeLeftMs",
                ET_OUTCOME,
                timeAvailableMs, "ms");

        doEvent(ML::format("auctionEarlyDrop.peer.%s",
                               transport().getPeerName().c_str()).c_str());

        dropAuction(ML::format("timeleft of %f is too low",
                               timeAvailableMs));

        return;
    }

    /* This is the callback that the auction will call once finish()
       has been called on it.  It will be called exactly once.
    */
    auto handleAuction = [=] (std::shared_ptr<Auction> auction)
        {
            //cerr << "HANDLE AUCTION CALLED AFTER "
            //<< Date::now().secondsSince(auction->start) * 1000
            //<< "ms" << endl;

            if (!auction || auction->isZombie)
                return;  // Was already externally terminated; this is invalid

            if (this->transport().lockedByThisThread())
                this->sendResponse();
            else {
                this->transport()
                    .doAsync(boost::bind(&HttpAuctionHandler::sendResponse,
                                         this),
                             "AsyncSendResponse");
            }
        };
    
    // We always give ourselves 5ms to bid in, no matter what (let upstream
    // deal with it if it's really that much too slow).
    Date expiry = firstData.plusSeconds
        (max(5.0, (timeAvailableMs - networkTimeMs)) / 1000.0);

    try {
        auto bidRequest = parseBidRequest(header, payload);

        if (!bidRequest) {
            endpoint->recordHit("error.noBidRequest");
            //cerr << "got no bid request" << endl;
            // The request was handled; nothing to do
            return;
        }

        auction.reset(new Auction(endpoint,
                                  handleAuction, bidRequest,
                                  bidRequest->toJsonStr(),
                                  "datacratic",
                                  firstData, expiry));

        endpoint->adjustAuction(auction);


#if 0
        static std::mutex lock;
        std::unique_lock<std::mutex> guard(lock);
        cerr << "bytes before = " << payload.size() << " after "
             << auction->requestStr.size() << " ratio "
             << 100.0 * auction->requestStr.size() / payload.size()
             << "%" << endl;
        string s = bidRequest->serializeToString();
        cerr << "serialized bytes before = " << payload.size() << " after "
             << s.size() << " ratio "
             << 100.0 * s.size() / payload.size() << "%" << endl;
#endif

    } catch (const std::exception & exc) {
        // Inject the error message in
        std::stringstream details;
        details << "Error parsing bid request : " <<  exc.what() << std::endl;
        sendErrorResponse("INVALID_BID_REQUEST" , details.str() );
        return;
    }

    doEvent("auctionNetworkLatencyMs",
            ET_OUTCOME,
            (firstData.secondsSince(auction->request->timestamp)) * 1000.0,
            "ms");

    doEvent("auctionTotalStartLatencyMs",
            ET_OUTCOME,
            (now.secondsSince(auction->request->timestamp)) * 1000.0,
            "ms");

    doEvent("auctionStart");

    addActivity("gotAuction %s", auction->id.toString().c_str());
    
    if (now > expiry) {
        doEvent("auctionAlreadyExpired");

        string msg = format("auction started after time already elapsed: "
                            "%s vs %s, available time = %.1fms, "
                            "firstData = %s",
                            now.print(4).c_str(),
                            expiry.print(4).c_str(),
                            timeAvailableMs,
                            firstData.print(4).c_str());
        cerr << msg << endl;

        dropAuction("auction already expired");

        return;
    }
    
    addActivity("timeAvailable: %.1fms", timeAvailableMs);
    
    scheduleTimerAbsolute(expiry, 1);
    hasTimer = true;
    
    addActivity("gotTimer for %s",
                expiry.print(4).c_str());

    auction->doneParsing = Date::now();

    ML::atomic_add(endpoint->numAuctions, 1);
    endpoint->onNewAuction(auction);
}

void
HttpAuctionHandler::
sendResponse()
{
    checkMagic();

    if (!transport().lockedByThisThread())
        throw Exception("sendResponse must be in handler context");

    addActivityS("sendResponse");

    //cerr << "locked by " << bid->lock.get_thread_id() << endl;
    //cerr << "my thread " << ACE_OS::thr_self() << endl;

    Date before = Date::now();

    /* Make sure the transport isn't dead. */
    transport().checkMagic();

    if (!transport().lockedByThisThread())
        throw Exception("transport not locked by this thread");

    if (!auction) {
        throw Exception("sending response for cleared auction");
    }

    if (!auction->tooLate())
        throw Exception("auction is not finished");

    addActivity("sendResponse (lock took %.2fms)",
                Date::now().secondsSince(before) * 1000);
    
    cancelTimer();

    endpoint->onAuctionDone(auction);

    //cerr << "sendResponse " << this << ": disconnected "
    //     << disconnected << endl;

    if (disconnected) {
        closeWhenHandlerFinished();
        return;
    }
    
    HttpResponse response = getResponse();
    
    Date startTime = auction->start;
    Date beforeSend = Date::now();

    auto onSendFinished = [=] ()
        {
            //static int n = 0;
            //ML::atomic_add(n, 1);
            //cerr << "sendFinished canBlock = " << canBlock << " "
            //<< n << endl;
            this->addActivityS("sendFinished");
            double sendTime = Date::now().secondsSince(beforeSend);
            if (sendTime > 0.01)
                cerr << "sendTime = " << sendTime << " for "
                     << (auction ? auction->id.toString() : "NO AUCTION")
                     << endl;

            this->doEvent("auctionResponseSent");
            this->doEvent("auctionTotalTimeMs",
                          ET_OUTCOME,
                          Date::now().secondsSince(this->firstData) * 1000.0,
                          "ms",
                          { 90, 95, 98, 99 });

            if (random() % 1000 == 0) {
                this->transport().closeWhenHandlerFinished();
            }
            else {
                this->transport().associateWhenHandlerFinished
                (this->makeNewHandlerShared(), "sendFinished");
            }
        };

    addActivityS("beforeSend");
    
    double timeTaken = beforeSend.secondsSince(startTime) * 1000;

    response.extraHeaders
        .push_back({"X-Processing-Time-Ms", to_string(timeTaken)});

    putResponseOnWire(response, onSendFinished);
}

void
HttpAuctionHandler::
dropAuction(const std::string & reason)
{
    auto onSendFinished = [=] ()
        {
            if (random() % 1000 == 0) {
                this->transport().closeWhenHandlerFinished();
            }
            else {
                this->transport().associateWhenHandlerFinished
                (this->makeNewHandlerShared(), "sendFinished");
            }
        };

    putResponseOnWire(endpoint->getDroppedAuctionResponse(*this, reason),
                      onSendFinished);
}

void
HttpAuctionHandler::
sendErrorResponse(const std::string & error,
                  const std::string & details)
{
    putResponseOnWire(endpoint->getErrorResponse(*this,  error + ": " + details));
    endpoint->onAuctionError("EXCHANGE_ERROR", auction, error + ": " + details);
}

std::string
HttpAuctionHandler::
status() const
{
    if (!hasTransport())
        return "newly minted HttpAuctionHandler";

    string result = format("GenericHttpAuctionHandler: %p readState %d hasTimer %d",
                           this, readState, hasTimer);
    result += "auction: ";
    if (auction) result += getResponse().body;
    else result += "NULL";
    return result;
}

HttpResponse
HttpAuctionHandler::
getResponse() const
{
    return endpoint->getResponse(*this, this->header, *auction);
}

std::shared_ptr<BidRequest>
HttpAuctionHandler::
parseBidRequest(const HttpHeader & header,
                const std::string & payload)
{
    return endpoint->parseBidRequest(*this, header, payload);
}

double
HttpAuctionHandler::
getTimeAvailableMs(const HttpHeader & header,
                   const std::string & payload)
{
    return endpoint->getTimeAvailableMs(*this, header, payload);
}

double
HttpAuctionHandler::
getRoundTripTimeMs(const HttpHeader & header)
{
    return endpoint->getRoundTripTimeMs(*this, header);
}

} // namespace RTBKIT
