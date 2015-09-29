/* openrtb_bid_request_parser.h                                -*- C++ -*-
   Jean-Michel Bouchard, 19 August 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

   Bid request parser interface
*/

#pragma once

#include <string>
#include "rtbkit/common/bid_request.h"
#include "jml/utils/parse_context.h"
#include "rtbkit/plugins/bid_request/bid_request_parser.h"
#include "soa/service/logs.h"

namespace RTBKIT {

struct OpenRTBBidRequestLogs {
    static Logging::Category trace; 
    static Logging::Category error;
    static Logging::Category trace22;
    static Logging::Category error22;
};

/*****************************************************************************/
/* OPEN RTB BID REQUEST PARSER                                               */
/*****************************************************************************/

//struct OpenRTBBidRequestParser : IBidRequestParser<OpenRTB::BidRequest, const std::string>,
//                                 IBidRequestParser<OpenRTB::BidRequest, ML::Parse_Context>
struct OpenRTBBidRequestParser
{

    OpenRTB::BidRequest parseBidRequest(const std::string & jsonValue);
    OpenRTB::BidRequest parseBidRequest(ML::Parse_Context & context);

    OpenRTB::BidRequest toBidRequest(const RTBKIT::BidRequest & br);

    RTBKIT::BidRequest* parseBidRequest(ML::Parse_Context & context,
                                        const std::string & provider,
                                        const std::string & exchange);

    RTBKIT::BidRequest* parseBidRequest(const std::string & json,
                                        const std::string & provider,
                                        const std::string & exchange);

    static std::unique_ptr<OpenRTBBidRequestParser>
        openRTBBidRequestParserFactory(const std::string & version);

    struct OpenRTBParsingContext {
        std::unique_ptr<RTBKIT::BidRequest> br;
        AdSpot spot;
    } ctx;
    
    virtual ~OpenRTBBidRequestParser(){};

    protected :
        // All those methods are defined based on OpenRTB 2.1
        // further versions just have to redefine those methods
        virtual void onBidRequest(OpenRTB::BidRequest & br);
        virtual void onImpression(OpenRTB::Impression & imp);
        virtual void onBanner(OpenRTB::Banner & banner);
        virtual void onVideo(OpenRTB::Video & video);
        virtual void onSite(OpenRTB::Site & site);
        virtual void onApp(OpenRTB::App & app);
        virtual void onContext(OpenRTB::Context & context);
        virtual void onContent(OpenRTB::Content & content);
        virtual void onProducer(OpenRTB::Producer & producer);
        virtual void onPublisher(OpenRTB::Publisher & publisher);
        virtual void onDevice(OpenRTB::Device & device);
        virtual void onGeo(OpenRTB::Geo & geo);
        virtual void onUser(OpenRTB::User & user);
        virtual void onData(OpenRTB::Data & data);
        virtual void onSegment(OpenRTB::Segment & segment);

        std::unordered_map<int, std::string> apiFrameworks;

    private:
        RTBKIT::BidRequest * createBidRequestHelper(OpenRTB::BidRequest & br,
                                    const std::string & provider,
                                    const std::string & exchange);
};

struct OpenRTBBidRequestParser2point1 : OpenRTBBidRequestParser {

    OpenRTBBidRequestParser2point1() {
        apiFrameworks = { {1, "VPAID 1.0"},
                          {2, "VPAID 2.0"},
                          {3, "MRAID"},
                          {4, "ORMMA"}
        };
    };

    private:
        virtual void onDevice(OpenRTB::Device& device);
};

struct OpenRTBBidRequestParser2point2 : OpenRTBBidRequestParser {

    OpenRTBBidRequestParser2point2() {
        apiFrameworks = { {1, "VPAID 1.0"},
                          {2, "VPAID 2.0"},
                          {3, "MRAID-1"},
                          {4, "ORMMA"},
                          {5, "MRAID-2"}
        };
    };

    private :
        virtual void onBidRequest(OpenRTB::BidRequest & br);
        virtual void onImpression(OpenRTB::Impression & imp);
        virtual void onBanner(OpenRTB::Banner & banner);
        virtual void onVideo(OpenRTB::Video & video);
        virtual void onDevice(OpenRTB::Device & device);
        virtual void onRegulations(OpenRTB::Regulations & regs);
        virtual void onPMP(OpenRTB::PMP & pmp);
        virtual void onDeal(OpenRTB::Deal & deal);
};

} // namespace RTBKIT
