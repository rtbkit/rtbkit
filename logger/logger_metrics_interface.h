/* logger_metrics_interface.h                                      -*- C++ -*-
   François-Michel L'Heureux, 21 May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#pragma once

#include <iostream>
#include <ostream>
#include <cstring>
#include <mutex>
#include <functional>
#include "boost/variant.hpp"
#include "jml/arch/exception.h"
#include "soa/jsoncpp/json.h"
#include "soa/types/date.h"


namespace Datacratic {

/****************************************************************************/
/* LOGGER METRICS                                                           */
/****************************************************************************/

/**
 * KvpLogger are key-value-pair loggers
 * Mix of patterns:
 * - Provides a factory to instanciate a logger
 * - Provides an interface for metrics loggers
 * - Provides adaptor functions to avoid defining redundant functions in
 *   implementations
 */
struct ILoggerMetrics {
    // ORDER OF VARIANT IMPORTANT!
    typedef boost::variant<int, float, double, size_t, uint32_t> Numeric;
    typedef boost::variant<int, float, double, size_t, uint32_t, std::string> NumOrStr;

    ILoggerMetrics() = delete;

    static std::shared_ptr<ILoggerMetrics> setup(const std::string & configKey,
                                                 const std::string & coll,
                                                 const std::string & appName);
    static std::shared_ptr<ILoggerMetrics>
        setupFromJson(const Json::Value & config,
                      const std::string & coll, const std::string & appName);

    /**
     * Factory like getter for kvp
     */
    static std::shared_ptr<ILoggerMetrics> getSingleton();

    void logMetrics(const Json::Value &);
    template<typename jsonifiable>
    void logMetrics(const jsonifiable & j)
    {
        auto fct = [&] () {
            Json::Value root = j.toJson();
            logMetrics(root);
        };
        failSafeHelper(fct);
    };
    void logMetrics(const std::vector<std::string> & path, const Numeric & val)
    {
        auto fct = [&] () {
            logInCategory(METRICS, path, val);
        };
        failSafeHelper(fct);
    }

    void logProcess(const Json::Value & j)
    {
        auto fct = [&]() {
            logInCategory(PROCESS, j);
        };
        failSafeHelper(fct);
    }
    template<typename jsonifiable>
    void logProcess(const jsonifiable & j)
    {
        auto fct = [&] () {
            Json::Value root = j.toJson();
            logProcess(root);
        };
        failSafeHelper(fct);
    };
    void logProcess(const std::vector<std::string> & path, const NumOrStr & val)
    {
        auto fct = [&] () {
            logInCategory(PROCESS, path, val);
        };
        failSafeHelper(fct);
    }

    void logMeta(const Json::Value & j)
    {
        auto fct = [&] () {
            logInCategory(META, j);
        };
        failSafeHelper(fct);
    }
    template<typename jsonifiable>
    void logMeta(const jsonifiable & j)
    {
        auto fct = [&] () {
            Json::Value root = j.toJson();
            logMeta(root);
        };
        failSafeHelper(fct);
    };
    void logMeta(const std::vector<std::string> & path, const NumOrStr & val)
    {
        auto fct = [&] () {
            logInCategory(META, path, val);
        };
        failSafeHelper(fct);
    }

    void close();
    virtual ~ILoggerMetrics(){};

protected:
    const static std::string METRICS;
    const static std::string PROCESS;
    const static std::string META;

    ILoggerMetrics(const std::string & coll)
        : coll(coll), startDate(Date::now())
    {
    }
    virtual void logInCategory(const std::string & category,
                               const std::vector<std::string> & path,
                               const NumOrStr & val) = 0;
    virtual void logInCategory(const std::string & category,
                               const Json::Value & j) = 0;
    virtual std::string getProcessId() const = 0;

    void failSafeHelper(const std::function<void()> & fct);

    std::string coll;

private:
    static void setupLogger(const Json::Value & config,
                            const std::string & coll,
                            const std::string & appName);
    static bool failSafe;
    static std::string parentObjectId;

    const Date startDate;
};

} // namespace Datacratic
