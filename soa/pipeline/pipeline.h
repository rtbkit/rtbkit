/* pipeline.h
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

namespace Datacratic
{
    struct Environment {
        std::string expandVariables(std::string const & text) const;
        std::string getVariable(std::string const & text) const;

        void set(std::string key, std::string value);

    private:
        std::map<std::string, std::string> keys;
    };

    CREATE_STRUCTURE_DESCRIPTION(Environment)

    struct Pipeline :
        public Blocks
    {
        Pipeline();

        virtual Connector * createConnector(IncomingPin * incoming, OutgoingPin * outgoing) = 0;

        ReadingPin<Environment> environment;
    };
}

