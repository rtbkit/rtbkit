/* rubicon_exchange_connector_test.cc
   Jeremy Barnes, 12 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Exchange connector for Rubicon.
*/

#include "rtbkit/plugins/exchange/rubicon_exchange_connector.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include <boost/lexical_cast.hpp>

#include "rtbkit/core/router/router.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/banker/null_banker.h"
#include "rtbkit/testing/test_agent.h"

// for generation of dynamic creative
#include "cairomm/surface.h"
#include "cairomm/context.h"

#include <type_traits>


using namespace std;
using namespace RTBKIT;

// http://web.archiveorange.com/archive/v/QVfaOb8fEnu9f52fo4y2
// This is needed to allow a std::function to bind into the sigc library
namespace sigc
{
    template <typename Functor>
    struct functor_trait<Functor, false>
    {
        typedef decltype (::sigc::mem_fun (std::declval<Functor&> (),
                                           &Functor::operator())) _intermediate;

        typedef typename _intermediate::result_type result_type;
        typedef Functor functor_type;
    };
}


/*****************************************************************************/
/* TEST RUBICON EXCHANGE CONNECTOR                                           */
/*****************************************************************************/

/** This class extends the standard Rubicon exchange connector with
    functionality needed for Rubicon's integration test page.

    It includes the following extra functionality:
    * 
*/

struct TestRubiconExchangeConnector: public RubiconExchangeConnector {

    TestRubiconExchangeConnector(const std::string & name,
                             std::shared_ptr<ServiceProxies> proxies)
        : RubiconExchangeConnector(name, proxies)
    {
    }
    
    /** Simple function to return a dynamic creative with the win price
        rendered in it to demonstrate that the win price was correctly
        decoded.
    */
    HttpResponse getCreative(int width, int height, float winPrice,
                             string format) const
    {
        string imageData;
        string contentType;

        {
            auto writeData = [&] (const unsigned char * data,
                                  unsigned length) -> Cairo::ErrorStatus
                {
                    //cerr << "wrote " << length << " bytes" << endl;
                    imageData.append((const char *)data, length);
                    return CAIRO_STATUS_SUCCESS;
                };

            static const Cairo::RefPtr<Cairo::Surface> logo
                = Cairo::ImageSurface::create_from_png("static/rtbkit-logo-256x50.png");

            Cairo::RefPtr<Cairo::Surface> surface;
            if (format == "svg") {
                surface = Cairo::SvgSurface::create_for_stream(Cairo::Surface::SlotWriteFunc(writeData), width, height);
                contentType = "image/svg+xml";
            }
            else if (format == "png") {
                surface = Cairo::ImageSurface::create(Cairo::FORMAT_ARGB32, width, height);
                contentType = "image/png";
            }
            else throw ML::Exception("unknown creative format");

            Cairo::RefPtr<Cairo::Context> cr = Cairo::Context::create(surface);

            cr->save(); // save the state of the context
            cr->set_source_rgb(0.86, 0.85, 0.47);
            cr->paint(); // fill image with the color
            cr->restore(); // color is back to black now

            cr->save(); // save the state of the context
            cr->set_source(logo, 0.10, 0.10);
            cr->paint();
            cr->restore();

            cr->save();
            // draw a border around the image
            cr->set_line_width(5.0); // make the line wider
            cr->rectangle(0.0, 0.0, cairo_image_surface_get_width(surface->cobj()), height);
            cr->stroke();

            cr->set_source_rgba(0.0, 0.0, 0.0, 0.7);
            // draw a circle in the center of the image
            cr->arc(width / 2.0, height / 2.0,
                    height / 4.0, 0.0, 2.0 * M_PI);
            cr->stroke();

            // draw a diagonal line
            cr->move_to(width / 4.0, height / 4.0);
            cr->line_to(width * 3.0 / 4.0, height * 3.0 / 4.0);
            cr->stroke();
            cr->restore();
            
            Cairo::RefPtr<Cairo::ToyFontFace> font =
                Cairo::ToyFontFace::create("Bitstream Charter",
                                           Cairo::FONT_SLANT_ITALIC,
                                           Cairo::FONT_WEIGHT_BOLD);
            cr->set_font_face(font);
            cr->set_font_size(48.0);
            cr->move_to(width / 4.0, height / 2.0);
            cr->show_text("Win Price: " + to_string(winPrice));

            cr->show_page();

            if (format == "png")
                surface->write_to_png_stream(writeData);
        }
        
        return HttpResponse(200, contentType, imageData);
    }

    virtual void
    handleUnknownRequest(HttpAuctionHandler & connection,
                         const HttpHeader & header,
                         const std::string & payload) const
    {
        // Redirect to the actual ad, but lets us get the price
        if (header.resource == "/creative.png"
            || header.resource == "/creative.svg") {
            int width = boost::lexical_cast<int>(header.queryParams.getValue("width"));
            int height = boost::lexical_cast<int>(header.queryParams.getValue("height"));
            string encodedPrice = header.queryParams.getValue("price");

            //cerr << "encodedPrice = " << encodedPrice << endl;
            
            float decodedPrice = decodeWinPrice("hheehhee", encodedPrice);
            auto response = getCreative(width, height, decodedPrice,
                                        header.resource == "/creative.png" ? "png" : "svg");
            connection.putResponseOnWire(response);
            return;
        }
        else if (header.resource == "/redirect.js") {
            //cerr << "redirect to " << header << endl;

            RestParams params;
            string price;
            string redir;

            for (auto & p: header.queryParams) {
                if (p.first == "price")
                    price = p.second;
                if (p.first == "redir") {
                    redir = p.second;
                    continue;
                }
                params.push_back(p);
            }

            string redirLocation = redir + params.uriEscaped();
            
            HttpResponse response(302, "none", "", { {"Location", redirLocation} });
            connection.putResponseOnWire(response);
            return;
        }
        connection.sendErrorResponse("unknown resource " + header.resource);
    }
};

int main(int argc, char ** argv)
{
    RubiconExchangeConnector::decodeWinPrice("hheehhee", "386C13726472656E");

    std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());

    // The agent config service lets the router know how our agent is configured
    AgentConfigurationService agentConfig(proxies, "config");
    agentConfig.init();
    agentConfig.bindTcp();
    agentConfig.start();

    // We need a router for our exchange connector to work
    Router router(proxies, "router");
    router.unsafeDisableMonitor();  // Don't require a monitor service
    router.init();

    // Set a null banker that blindly approves all bids so that we can
    // bid.
    router.setBanker(std::make_shared<NullBanker>(true));

    // Start the router up
    router.bindTcp();
    router.start();

    // Create our exchange connector and configure it to listen on port
    // 10002.  Note that we need to ensure that port 10002 is open on
    // our firewall.
    std::unique_ptr<TestRubiconExchangeConnector> connector
        (new TestRubiconExchangeConnector("connector", proxies));

    connector->configureHttp(1, 10002, "0.0.0.0");
    connector->start();
    connector->enableUntil(Date::positiveInfinity());

    // Tell the router about the new exchange connector
    router.addExchange(connector.release());

    // This is our bidding agent, that actually calculates the bid price
    TestAgent agent(proxies, "agent");

    string hostName = "dev2.recoset.com:10002";

    agent.config.providerConfig["rubicon"]["seat"] = "123";

    // Configure the agent for bidding
    for (auto & c: agent.config.creatives) {
        c.providerConfig["rubicon"]["adomain"][0] = "rtbkit.org";
        c.providerConfig["rubicon"]["adm"]
            = "<img src=\"http://"
            + hostName
            + "/creative.png?width="
            + to_string(c.format.width)
            + "&height="
            + to_string(c.format.height)
            + "&price=${AUCTION_PRICE:BF}\"/>";
        c.providerConfig["rubicon"]["crid"] = c.name;
        c.providerConfig["rubicon"]["attr"][0] = 0;
    }

    agent.onBidRequest = [&] (
            double timestamp,
            const Id & id,
            std::shared_ptr<BidRequest> br,
            Bids bids,
            double timeLeftMs,
            const Json::Value & augmentations)
        {
            Bid& bid = bids[0];

            bid.bid(bid.availableCreatives[0], USD_CPM(1.234));

            agent.doBid(id, bids, Json::Value());
            ML::atomic_inc(agent.numBidRequests);
        };

    agent.init();
    agent.start();
    agent.configure();

    cerr << agent.config.toJson() << endl;

    for (;;) {
        // Make the agent send its configuration to the router
        ML::sleep(10.0);
        //proxies->events->dump(cerr);
    }
}
