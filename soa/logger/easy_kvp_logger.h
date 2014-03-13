#pragma once

#include "soa/logger/kvp_logger_interface.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/options_description.hpp>
#include "soa/types/date.h"
#include <iostream>

namespace Datacratic{

/**
 * EasyKvpLogger is a wrapper that easily allows to log a runId, key and value.
 * It's initialized via a boost ptree and default keys. (See the .cc file.)
 */
class EasyKvpLogger{
    typedef std::map<std::string, std::string> strmap;
    typedef std::map<std::string, std::string>::const_iterator strmap_citerator;

    public:
        EasyKvpLogger(const boost::property_tree::ptree& pt, 
            const std::string& coll,
            const std::string& envVar,
            const bool& logStarEnd = false,
            const strmap& defaults = strmap());
        EasyKvpLogger(const boost::property_tree::ptree& pt, 
            const std::string& coll, const std::string& envVar, 
            const std::string& runId,
            const bool& logStartEnd = false,
            const strmap& defaults = strmap());

        EasyKvpLogger(const boost::program_options::variables_map& vm,
            const std::string& coll,
            const std::string& envVar,
            const bool& logStarEnd = false,
            const strmap& defaults = strmap());
                    
        ~EasyKvpLogger();

        void log(strmap& kvpMap);
        
        /**
         * Same as log but we don't need to declare the strmap before passing,
         * it, we can do it inline
         */
        void clog(const strmap& kvpMap);
        void log(const std::string& key, const std::string& value);
        std::string getRunId();

        /**
         * Clears the defaults values and uses the new ones.
         */
        void setDefaults(const strmap& defaults);

        static boost::program_options::options_description get_options();

    private:
        const std::string coll;
        std::string runId;
        std::shared_ptr<IKvpLogger> logger;
        strmap defaults;
        const bool logStartEnd;
        Date start;
        void defineLogger(const boost::property_tree::ptree& pt,
            const strmap& defaults, const bool& logStartEnd);
        void defineLogger(const boost::program_options::variables_map& vm,
            const strmap& defaults, const bool& logStartEnd);
        void logStart();
        void setEnvVar(const std::string& envVar, const std::string& runId);
        void setDateAsRunIdIfNotInEnvVar(const std::string & envVar);


        /**
         * Adds defaults values + runId, prior to logging.
         */
        void addDefaultsToMap(strmap& kvpMap);
};


}//namespace Datacratic
