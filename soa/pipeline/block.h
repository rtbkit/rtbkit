/* block.h
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

namespace Datacratic
{
    struct Block {
        Block();
        Block(Pipeline * pipeline);

        virtual ~Block() {
        }

        std::string const & getName() const;
        std::string const & getPath() const;
        Pipeline * getPipeline() const;

        Connector * createConnector(IncomingPin * incoming, OutgoingPin * outgoing);
        void add(Pin * item);

        std::vector<IncomingPin *> const & getIncomingPins() const;
        std::vector<OutgoingPin *> const & getOutgoingPins() const;

        virtual void run();

        static Logging::Category print;
        static Logging::Category error;
        static Logging::Category trace;
        static Logging::Category debug;

    private:
        Pipeline * pipeline;
        Block * parent;
        std::string name;
        std::string path;
        std::vector<IncomingPin *> incomings;
        std::vector<OutgoingPin *> outgoings;

        friend struct Blocks;
        friend struct Pipeline;
    };

    struct Blocks :
        public Block
    {
        Blocks();
        Blocks(Pipeline * pipeline);

        std::vector<std::shared_ptr<Block>> const & getBlocks() const;

        template<typename T, class = typename std::enable_if<std::is_base_of<Block, T>::value>::type>
        T * create(std::string name) {
            auto handle = std::make_shared<T>();
            auto result = handle.get();
            add(std::static_pointer_cast<Block>(handle), std::move(name));
            return result;
        }

        void add(std::shared_ptr<Block> item, std::string name);

    private:
        std::vector<std::shared_ptr<Block>> blocks;
    };
}

