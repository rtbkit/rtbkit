/* bid_aggregator_test.cc
   Jeremy Barnes, 31 January 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Class to aggregate up bidding data and allow basic querying and feature
   vector generation.

   Time:
   - Filter by time period
   - Generate time series over the time period up to the minute granularity

   - Per-minute rollups / time resolution

   - Data segmented on
     - campaign
     - strategy
     - exchange
     - hour (implicitly)

   - Primary attributes (indexed):
     - exchange
     - format
     - creative
     - domain
     - hour
     - minute
     - weekday
     - url
     - country
     - state
     - city

   - Secondary attributes (non-indexed):
     - data profile
     - browser
     - OS
     - Anything else in the bid request, bid, click or impression...   
     - timezone

   - Primary metrics
     - bid price (mean, median, min, max, std)
     - win price (mean, median, min, max, std)
     - surplus (mean, median, min, max, std)
     - wins
     - losses
     - impressions (ie, 1 for each row)
     - clicks
     - eventually, anything that can be expressed as a behaviour

   - Aggregators
     - sum
     - count
     - average
     - weighted average (x, y)
     - ratio (x / y)

   Operations:
     - PivotTable
       - Create a pivot table over the given attributes with the given filter
     - TimeSeries
       - Create a time series over the given attributes (date + metrics)
     - Range
     - Dictionary
       - Return the possible values and ranges of the given attributes and their overall count (used for ui generation)
     - ReturnData
       - Return the full bid requests for the given data

   Data sizing
     - 120,000 MATCHEDWINs are 160MB uncompressed and 28MB gzip compressed
     - Assume with attribute memoization and cleanup it's 60GB
     - That is $100 per day, ie a small campaign
     - Assume 50% wins and 50% losses, so 320MB overall or 60MB compressed
     - 30 days = 1.8GB compressed to scan or 10GB uncompressed
     - At $1 CPM, 1GB for each $1000 spent

   Index format
     - sorted heap data structure
     - segment on campaign, strategy, hour
     - sort by primary attributes with low entropy ones first
     - categorical attributes
         - first store index of key -> offset
         - then store the data itself: either next level or leaf data

     - Leaf nodes associated with index: metrics + bid request offset


     zcat MATCHEDALL-2012-02-17.log.gz | awk '{ counts[$1] += 1; } END { for (i in counts) print i, counts[i]; }'
     MATCHEDWIN 646806
     MATCHEDIMPRESSION 597248
     MATCHEDCLICK 8
     MATCHEDLOSS 2124058
     jeremy@ag1:~/projects/platform$ ls -l MATCHEDALL-2012-02-17.log.gz 
     -rw-r--r-- 1 jeremy jeremy 983665573 2012-02-18 14:04 MATCHEDALL-2012-02-17.log.gz
*/



#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/logger/logger.h"
#include "soa/logger/log_message_splitter.h"
#include "rtbkit/common/bid_request.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/string_functions.h"
#include "jml/utils/lightweight_hash.h"
#include "jml/utils/json_parsing.h"
#include <unordered_map>
#include "jml/arch/bitops.h"
#include "jml/arch/bit_range_ops.h"
#include "soa/types/id.h"
#include <boost/tuple/tuple.hpp>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/make_shared.hpp>


using namespace std;
using namespace Datacratic;
using namespace ML;


struct StringMap : public ValueMap {
    std::unordered_map<string, int> toInt;
    std::vector<std::string> toString;

    int operator [] (const std::string & str)
    {
        auto res = toInt.insert(make_pair(str, toInt.size()));
        if (res.second)
            toString.push_back(str);
        return res.first->second;
    }

    const std::string & operator [] (int i) const
    {
        if (i >= 0 && i < toString.size())
            return toString[i];
        throw ML::Exception("no stringmap");
    }

    /** Convert the given string value into an integral representation.
        Guaranteed not to fail; will add the value if it doesn't exist.
    */
    virtual int parse(const std::string & str)
    {
        return operator [] (str);
    }

    /** Return the string corresponding to the given integral value.  Throws
        an exception if the integer doesn't correspond to a valid value.
    */
    virtual std::string print(int val) const
    {
        return operator [] (val);
    }

    /** Returns the number of entries in the value table. */
    virtual size_t size() const
    {
        return toInt.size();
    }

    /** Returns whether or not the given key is in the value map. */
    virtual bool has(const std::string & str) const
    {
        return toInt.count(str);
    }

    /** Returns some kind of type specific path identifier that can be
        used to identify the string map in serialization/reconstitution.
    */
    virtual std::string identifier() const
    {
        return id;
    }

    std::string id;
};

struct FileKey {
    string exchange;
    string strategy;
    string campaign;

    bool operator == (const FileKey & other) const
    {
        return exchange == other.exchange
            && strategy == other.strategy
            && campaign == other.campaign;
    }

    bool operator != (const FileKey & other) const
    {
        return ! operator == (other);
    }

    bool operator < (const FileKey & other) const
    {
        return ML::less_all(exchange, other.exchange,
                            strategy, other.strategy,
                            campaign, other.campaign);
    }

    std::string print() const
    {
        return campaign + "|" + strategy + "|" + exchange;
    }
};

inline std::ostream & operator << (std::ostream & stream, const FileKey & key)
{
    return stream << key.print();
}

using namespace boost::iostreams;

struct SchemaInferrer {

    std::shared_ptr<const TypeHandler> currentHandler;
    std::vector<std::shared_ptr<const TypeHandler> > oldHandlers;
    std::vector<std::string> taxonomy;

    SchemaInferrer()
        : entries(0), bytesIn(0), bytesOut(0)
    {
        buffer.reset(new CompressedBuffer());
    }

    ~SchemaInferrer()
    {
    }

    std::string stats()
    {
        return ML::format("%8zd vals, %2zd br, %8zd in, %8zd out, %8zd cmp, %5.2f%% r1, %5.2f%% r2",
                          entries, taxonomy.size(),
                          bytesIn, bytesOut, bytesCompressed(),
                          100.0 * bytesOut / bytesIn,
                          100.0 * bytesCompressed() / bytesIn);
    }

    struct CompressedBuffer {
        CompressedBuffer()
        {
            filter.push(gzip_compressor(6));
            filter.push(sink);
        }

        ~CompressedBuffer()
        {
            filter.reset();
        }

        void close()
        {
            if (filter)
                boost::iostreams::close(filter);
        }

        void flush()
        {
            boost::iostreams::flush(filter);
        }
        
        size_t bytesOut() const
        {
            return sink.str().size();
        }

        ostringstream sink;
        filtering_ostream filter;
    };

    std::shared_ptr<CompressedBuffer> buffer;

    size_t entries;
    size_t bytesIn;
    size_t bytesOut;
    size_t bytesCompressed() const { return buffer->bytesOut(); }

    void close()
    {
        buffer->close();
    }

    void flush()
    {
        buffer->flush();
    }

    Value accept(const std::string & field, ValueManager * owner,
                 HandlerContext & hContext)
    {
        ML::Parse_Context context(field, field.data(), field.size());
        
        try {
            std::shared_ptr<const TypeHandler> newHandler;
            Value value;
            boost::tie(value, newHandler)
                = parseJston(context, currentHandler, owner, hContext);

            if (newHandler != currentHandler) {
                taxonomy.push_back(field);
                oldHandlers.push_back(currentHandler);
                //cerr << "field " << fields[10] << " induced new handler "
                //     << handler->typeToJson().toString() << endl;
            }
            
            currentHandler = newHandler;

            string s = value.toString();
            
            boost::iostreams::write(buffer->filter, s.c_str(), s.size());

            ++entries;
            bytesIn += field.length();
            bytesOut += s.size();

            return value;
        } catch (...) {
            cerr << "parsing field " << field << endl;
            throw;
        }
    }
};

struct BidRequestEncoder {
};

#if 0
struct ValueConstructor : public ValueManager {
    std::shared_ptr<TypeHandler> handler;
    std::vector<Value> values;
    
    struct Entry {
        ValueConstructor * value;
        int index;

        template<typename T> void operator = (const T & t)
        {
            ViewHandler<T> vh(handler->getElementType(index));
            vh.set(Value(0, 0, 0, ValueVersion(),
                         0, index),
                   t, value);
        }
    };

    Entry operator [] (int index)
    {
        return Entry(this, index);
    }

    Value toValue(ValueManager * owner) const
    {
        return handler->constructValue(values, owner);
    }

    virtual Value replaceValue(const Value & element,
                               Value && replace_with,
                               const TypeHandler * type)
    {
    }

    virtual Value replaceValue(const Value & element,
                               const ConstValueBlock & replace_with,
                               const TypeHandler * type)
    {
        int index = element.fixed_width();
        values[index] = replace
    }
};
#endif

struct WinLossOutput : public LogOutput {

    std::shared_ptr<CompoundHandler> keyHandler;

#if 0
     - exchange
     - format
     - creative
     - domain
     - hour
     - minute
     - weekday
     - url
     - country
     - state
     - city
#endif

   std::shared_ptr<StringMap> formatMap;
   std::shared_ptr<StringMap> domainMap;
   std::shared_ptr<StringMap> urlMap;
   std::shared_ptr<StringMap> countryMap;
   std::shared_ptr<StringMap> stateMap;
   std::shared_ptr<StringMap> cityMap;

    WinLossOutput()
        : numMsgs(0), start(Date::now())
    {
        using std::make_shared;
        keyHandler = make_shared<CompoundHandler>();
        keyHandler->add_field("campaign", make_shared<StringHandler>());
        keyHandler->add_field("strategy", make_shared<StringHandler>());
        keyHandler->add_field("exchange", make_shared<StringHandler>());
        keyHandler->add_field("dateToHour", make_shared<DateHandler>());
        keyHandler->add_field("hourOfDay", make_shared<IntHandler>());
        keyHandler->add_field("weekday", make_shared<IntHandler>());
        keyHandler->add_field("format", make_shared<StringMapHandler>(formatMap));
        keyHandler->add_field("creative", make_shared<IntHandler>());
        keyHandler->add_field("domain", make_shared<StringMapHandler>(domainMap));
        keyHandler->add_field("url", make_shared<StringMapHandler>(urlMap));
        keyHandler->add_field("minute", make_shared<IntHandler>());
        keyHandler->add_field("country", make_shared<StringMapHandler>(countryMap));
        keyHandler->add_field("state", make_shared<StringMapHandler>(stateMap));
        keyHandler->add_field("city", make_shared<StringMapHandler>(cityMap));
    }


    size_t numMsgs;
    Date start;

    StringMap exchanges;
    StringMap strategies;
    StringMap campaigns;

    struct Data {
        SchemaInferrer br_schema;
        SchemaInferrer md_schema;
    };

    std::map<FileKey, Data> data;

    HandlerContext hContext;
    ValueManager owner;

    virtual void logMessage(const std::string & channel,
                            const std::string & message)
    {
        if (channel.empty() || message.empty()) return;

        if (++numMsgs % 10000 == 0) {
            Date now = Date::now();
            cerr << "message " << numMsgs << " in "
                 << now.secondsSince(start) << "s ("
                 << 1.0 * numMsgs / now.secondsSince(start) << "/s)"
                 << endl;
        }

        //if (numMsgs == 1)
        //    cerr << "got message " << message << endl;

        // Channels:
        // MATCHEDLOSS:  a bid that we submitted but did not win
        // MATCHEDWIN:   a bid that we won
        
        // Eventually:
        // WINORPHAN:    a bid that we won but without an impression
        // MATCHEDIMPRESSION: a bid, impression pair with no click
        // IMPRESSIONORPHAN: an impression that we won but without a bid req
        // MATCHEDCLICK: a bid, impression, click pair
        // CLICKORPHAN:  a click without a bid request and impression
        
        // Channel: MATCHEDWIN, MATCHEDLOSS, MATCHEDIMPRESSION, MATCHED
        // Message: tab separated timestamp, url, providerid

        // MATCHEDWIN message
        //  0      date
        //  1      bid request ID
        //  2      spot id
        //  3      client
        //  4      strategy
        //  5      win price micros
        //  6      bid price micros
        //  7      surplus
        //  8      bid request JSON
        //  9      bid JSON
        // 10      metadata JSON
        // 11      creative id   --> 30213
        // 12      campaign_slug --> netProphets / real time bidding / ron
        // 13      strategy_slug --> airtransat_tour_opt

        LogMessageSplitter<32> fields(message);

        Date timestamp;
        {
            ML::Parse_Context context(fields[0], fields[0].start, fields[0].end);
            timestamp = Date::expect_date_time(context, "%y-%M-%d", "%H:%M:%S");
            if (!context.eof())
                context.exception("expected date");
        }

        Id auctionId(fields[1]);
        Id spotId(fields[2]);

        string client = fields[3];
        string strategy = fields[4];
        
        int winPrice JML_UNUSED = boost::lexical_cast<int>(fields[5]);
        int bidPrice JML_UNUSED = boost::lexical_cast<int>(fields[6]);
        double surplus JML_UNUSED = boost::lexical_cast<double>(fields[7]);

        string br = fields[8];
        std::shared_ptr<BidRequest> req
            (BidRequest::parse("datacratic", br));

        // Find which part of a bid request needs to be removed to make it valid
        auto fixBidRequest = [] (const std::string & br) -> std::pair<int, int>
            {
                pair<int, int> r(0, 0);
                int & start = r.first;
                int & end = r.second;
                Parse_Context context(br, br.c_str(), br.size());
                if (!context.match_literal('{')) return r;
                start = context.get_offset();
                if (!context.match_literal("\"version\":")) return r;
                string ver;
                if (!ML::matchJsonString(context, ver)) return r;
                if (!context.match_literal(',')) return r;
                end = context.get_offset();
                return r;
            };

        int start, end;
        boost::tie(start, end) = fixBidRequest(br);
        //cerr << "br = " << br << endl;
        //cerr << "start = " << start << " end = " << end << endl;
        if (start < end)
            br.erase(start, end - start);
        //cerr << "br fixed = " << br << endl;

#if 0
        int spotNum = req->findSpotIndex(spotId);
        
        std::string format = req->imp[spotNum].format();
#endif

        int creativeId JML_UNUSED = boost::lexical_cast<int>(fields[11]);

        std::string country JML_UNUSED = req->location.countryCode;
        std::string region JML_UNUSED = req->location.regionCode;
        UnicodeString city JML_UNUSED = req->location.cityName;
        
#if 0
     - domain
     - hour
     - minute
     - weekday
     - url
     - country
     - state
     - city
#endif

        FileKey key;
        key.exchange = req->exchange;
        key.campaign = fields[12];
        key.strategy = fields[13];

        Data & fileData = data[key];

        fileData.br_schema.accept(br, &owner, hContext);
        fileData.md_schema.accept(fields[10], &owner, hContext);
    }

    void finish()
    {
        size_t totalBytesIn = 0, totalBytesOut = 0, totalOutCompressed = 0;
        cerr << "got " << data.size() << " files to save" << endl;
        for (auto it = data.begin(), end = data.end(); it != end;  ++it) {
            cerr << it->first << endl;
            it->second.md_schema.flush();
            it->second.md_schema.close();
            cerr << "  md " << it->second.md_schema.stats() << endl;
            totalBytesIn += it->second.md_schema.bytesIn;
            totalBytesOut += it->second.md_schema.bytesOut;
            totalOutCompressed += it->second.md_schema.bytesCompressed();

            it->second.br_schema.flush();
            it->second.br_schema.close();
            cerr << "  br " << it->second.br_schema.stats() << endl;
            totalBytesIn += it->second.br_schema.bytesIn;
            totalBytesOut += it->second.br_schema.bytesOut;
            totalOutCompressed += it->second.br_schema.bytesCompressed();
        }

        cerr << "total of " << totalBytesIn << " in, " << totalBytesOut
             << " out, ratio = " << 100.0 * totalBytesOut / totalBytesIn
             << "%" << endl;
        cerr << "compressed total of " << totalBytesOut << " in, " << totalOutCompressed
             << " out, ratio = " << 100.0 * totalOutCompressed / totalBytesOut
             << "%" << endl;
        cerr << "overall compression ratio = " << 100.0 * totalOutCompressed / totalBytesIn
             << "%" << endl;
    }

    virtual void close()
    {
    }

};

BOOST_AUTO_TEST_CASE( test_bid_aggregator )
{
    // Create a logger to get the behaviour events from

    std::shared_ptr<WinLossOutput> output
        (new WinLossOutput());

    Logger logger;
    logger.addOutput(output);

    logger.start();

    //logger.subscribe("/home/jeremy/platform/rtb_router_logger.ipc",
    //                 vector<string>({ "BEHAVIOUR" }));

    uint64_t maxEntries = -1;
    //maxEntries = 10000;
    logger.replay("MATCHEDWIN-2012-01-31.log.gz", maxEntries);

    cerr << "waiting until finished" << endl;

    logger.waitUntilFinished();

    cerr << "finished replay" << endl;

    output->finish();

    logger.shutdown();
}
