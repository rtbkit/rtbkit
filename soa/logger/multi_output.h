/* multi_output.h                                                  -*- C++ -*-
   Jeremy Barnes, 21 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Multiple output sources based upon the event.
*/

#ifndef __logger__multi_output_h__
#define __logger__multi_output_h__

#include "logger.h"
#include "rotating_output.h"
#include <unordered_map>
#include <mutex>
#include <thread>

namespace Datacratic {


/*****************************************************************************/
/* MULTI OUTPUT                                                              */
/*****************************************************************************/

/** Class that writes its output to multiple places. */

struct MultiOutput : public LogOutput {
    /** Function that creates a logger from a given timestamp. */
    typedef std::function<std::shared_ptr<LogOutput> (std::string)> CreateLogger;

    MultiOutput();

    virtual ~MultiOutput();

    virtual void logMessage(const std::string & channel,
                            const std::string & message);

    virtual void close();

    /** Set up the output class to log messages of the given type to files
        with the given pattern.

        channel - the channel that will be matched by this rule.  If the
                  field is the empty string, then the rule will match
                  any non-matching channels.
        pattern - The pattern to decide which file it should be logged
                  to.  It understands %(0), %(1), ... %(nnnnn) to mean
                  field nnnnn from the message (where %(0) is the channel).
        createLogger - The function to be called to create a subordinate
                  logger for the given filename.
    */
    void logTo(const std::string & channel,
               const std::string & pattern,
               const CreateLogger & createLogger);

    virtual Json::Value stats() const;

    virtual void clearStats();


private:

    struct ChannelEntry {
        
        std::string tmplate;

        enum TokenType {
            TOK_LITERAL,
            TOK_CHANNEL,
            TOK_FIELD
        };

        struct Token {
            Token()
                : type(TOK_LITERAL), field(-1)
            {
            }

            TokenType type;
            std::string literal;
            int field;
        };

        std::vector<Token> tokens;

        CreateLogger createLogger;

        void parse(const std::string & tmplate);

        std::string apply(const std::string & channel,
                          const std::string & message) const;
    };

    mutable std::mutex lock;
    
    /** A key for each channel */
    std::unordered_map<std::string, ChannelEntry> channels;

    /** Outputs, one per output key. */
    std::unordered_map<std::string, std::shared_ptr<LogOutput> > outputs;
};


} // namespace Datacratic


#endif /* __logger__multi_output_h__ */
