/* default_pipeline.h
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

namespace Datacratic
{
    struct DefaultPipeline :
        public Pipeline
    {
        DefaultPipeline();

        void run();

        Connector * createConnector(IncomingPin * incoming, OutgoingPin * outgoing);

    private:
        struct State {
            DefaultPipeline * pipeline;
            int count;
            Block * block;
        };

        struct DefaultConnector :
            public Connector
        {
            DefaultConnector(IncomingPin * incoming, OutgoingPin * outgoing);

            void push();

            State * state; 
        };

        std::set<std::shared_ptr<DefaultConnector>> connectors;
        std::vector<Block *> ready;
        std::map<Block *, State> states;

        friend struct DefaultConnector;
    };
}

