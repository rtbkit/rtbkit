/* multi_output.cc                                                 -*- C++ -*-
   Jeremy Barnes, 21 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Output into multiple files based upon the fields of the message.
*/

#include "multi_output.h"
#include "log_message_splitter.h"
#include "file_output.h"

using namespace std;


namespace Datacratic {


/*****************************************************************************/
/* MULTI OUTPUT                                                              */
/*****************************************************************************/

MultiOutput::
MultiOutput()
{
}

MultiOutput::
~MultiOutput()
{
    for (auto &output : outputs) {
        output.second->close();
    }
}

void
MultiOutput::ChannelEntry::
parse(const std::string & tmplate)
{
    tokens.clear();
    
    const char * p = tmplate.c_str();
    const char * e = p + tmplate.size();

    std::string currentLiteral;

    auto pushLiteral = [&] ()
        {
            if (currentLiteral.empty()) return;
            Token lit;
            lit.type = TOK_LITERAL;
            lit.literal = currentLiteral;
            tokens.push_back(lit);
            currentLiteral = "";
        };

    auto pushToken = [&] (Token token)
        {
            pushLiteral();
            tokens.push_back(token);
        };

    for (; p < e;  ++p) {
        char c = *p;
        switch (c) {
        case '$': {
            if (p == e - 1)
                throw ML::Exception("channel key ended");
            c = *++p;
            switch (c) {
            case '$':
                currentLiteral += '$';
                break;
            case '(': {
                string num;
                for (++p; p != e && *p != ')';  ++p)
                    num += *p;
                if (p == e)
                    throw ML::Exception("didn't close field number");
                int segNum = boost::lexical_cast<int>(num);
                Token token;
                if (segNum == 0)
                    token.type = TOK_CHANNEL;
                else {
                    token.type = TOK_FIELD;
                    token.field = segNum - 1;
                }
                pushToken(token);
                //++p;
                break;
            }
            default:
                throw ML::Exception("unacceptable channel key after '$'");
            }
            break;
        }
        default:
            currentLiteral += c;
        }
    }

    pushLiteral();

    //cerr << "parsing " << tmplate << " we got " << tokens.size() << " tokens"
    //     << endl;
    for (auto token: tokens) {
        cerr << "  " << token.type << " " << token.literal << " "
             << token.field << endl;
    }
}

std::string
MultiOutput::ChannelEntry::
apply(const std::string & channel,
      const std::string & message) const
{
    string result;

    LogMessageSplitter<128> split(message);

    for (auto token: tokens) {
        switch (token.type) {

        case TOK_LITERAL:
            result += token.literal;
            break;

        case TOK_CHANNEL:
            result += channel;
            break;

        case TOK_FIELD:
            result += split[token.field];
            break;

        default:
            throw ML::Exception("unknown token");
        }
    }

    return result;
}

void
MultiOutput::
logMessage(const std::string & channel,
           const std::string & message)
{
    std::unique_lock<std::mutex> guard(lock);
    auto it = channels.find(channel);
    if (it == channels.end())
        it = channels.find("");
    if (it == channels.end())
        return;


    //cerr << "got entry " << it->first << endl;

    auto & channelEntry = it->second;

    // 1.  Find which key the message should be logged to
    std::string messageKey = channelEntry.apply(channel, message);
    
    //cerr << "logging " << channel << " to " << it->first
    //     << " with messageKey " << messageKey << endl;
    //cerr << "messageKey = " << messageKey << endl;

    // 2.  Get the logger under the key; create if necessary
    std::shared_ptr<LogOutput> output;
    auto jt = outputs.find(messageKey);
    if (jt == outputs.end()) {
        cerr << "creating " << messageKey << endl;
        jt = outputs.insert(make_pair(messageKey,
                                      channelEntry.createLogger(messageKey))).first;
    }
    output = jt->second;
    guard.unlock();

    output->logMessage(channel, message);
}

void
MultiOutput::
logTo(const std::string & channel,
      const std::string & pattern,
      const CreateLogger & createLogger)
{
    ChannelEntry entry;
    entry.createLogger = createLogger;
    entry.parse(pattern);
    
    std::unique_lock<std::mutex> guard(lock);
    channels[channel] = entry;
}

void
MultiOutput::
close()
{
    throw ML::Exception("MultiOutput::close() needs implementation");
}

Json::Value
MultiOutput::
stats() const
{
    return Json::Value();
}

void
MultiOutput::
clearStats()
{
}

} // namespace Datacratic
