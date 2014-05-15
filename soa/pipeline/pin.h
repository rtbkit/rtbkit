/* pin.h
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

namespace Datacratic
{
    // base abstraction for pins of any types
    struct Pin
    {
        Pin(Block * block, std::string name);

        virtual ~Pin() {
        }

        std::string getName() const;
        std::string getPath() const;

        bool isIncomingPin() const;
        bool isOutgoingPin() const;

        Block * getBlock() const;

        Connector * connectWith(Pin * pin);
        Connector * connectWith(Pin & pin);
        void disconnect();
        void disconnectFrom(Connector * connector);

        virtual std::shared_ptr<const ValueDescription> getValueDescription() const;

    private:
        Connector * createConnector(Pin * pin);

        virtual IncomingPin * getAsIncomingPin();
        virtual OutgoingPin * getAsOutgoingPin();
        virtual void onCreateConnector(Connector * handle);
        virtual void onDeleteConnector(Connector * handle);

        bool isCompatibleWith(Pin * pin) const;

    private:
        std::string name;
        Block * block;
        bool incoming;
        bool outgoing;
        bool bus;

        friend class IncomingPin;
        friend class IncomingBus;
        friend class OutgoingPin;
    };

    // base abstraction for handling connection between an incoming and an outgoing pin
    struct Connector {
        Connector(IncomingPin * incoming, OutgoingPin * outgoing);

        virtual ~Connector() {
        }

        virtual void push();
        virtual void pull();

        IncomingPin * getIncomingPin() const;
        OutgoingPin * getOutgoingPin() const;

    private:
        virtual void onDisconnect();

        IncomingPin * incoming;
        OutgoingPin * outgoing;

        friend class IncomingPin;
        friend class OutgoingPin;
    };

    // base abstraction for pin that consumes data
    struct IncomingPin
        : public Pin
    {
        IncomingPin(Block * block, std::string name);

        bool isConnected() const;
        Connector * getConnector() const;

        virtual void readFrom(OutgoingPin * pin) = 0;

    private:
        IncomingPin * getAsIncomingPin();
        void onCreateConnector(Connector * handle);
        void onDeleteConnector(Connector * handle);

        Connector * connector;
    };

    // base abstraction for bus that consumes data
    struct IncomingBus
        : public Pin
    {
        IncomingBus(Block * block, std::string name);

        std::vector<std::shared_ptr<IncomingPin>> const & getIncomingPins() const;

    private:
        virtual std::shared_ptr<IncomingPin> createPin(Block * block, std::string name) = 0;
        IncomingPin * getAsIncomingPin();

        std::vector<std::shared_ptr<IncomingPin>> pins;
    };

    // base abstraction for pin that produces data
    struct OutgoingPin :
        public Pin
    {
        OutgoingPin(Block * block, std::string name);

        bool isConnected() const;
        std::vector<Connector *> const & getConnectors() const;

    private:
        OutgoingPin * getAsOutgoingPin();
        void onCreateConnector(Connector * handle);
        void onDeleteConnector(Connector * handle);

        std::vector<Connector *> connectors;
    };

    // pin for writing outgoing data
    template<typename T>
    struct WritingPin :
        public OutgoingPin
    {
        WritingPin(Block * block, std::string name) :
            OutgoingPin(block, std::move(name)) {
        }

        void set(std::shared_ptr<T> value) {
            data = std::move(value);
        }

        template<typename U = T, typename... Ts>
        U * set(Ts... args) {
            auto item = std::make_shared<U>(args...);
            data = std::static_pointer_cast<T>(item);
            return item.get();
        }

        void push() {
            auto & items = getConnectors();
            for(auto item : items) {
                item->push();
            }
        }

        void push(std::shared_ptr<T> value) {
            data = std::move(value);
            push();
        }

        template<typename U = T, typename... Ts>
        void push(Ts... args) {
            set(std::forward<Ts...>(args...));
            push();
        }

        std::shared_ptr<T> const & get() const {
            return data;
        }

        T & operator*() const {
            return *data;
        }

        T * operator->() const {
            return data.get();
        }

        std::shared_ptr<const ValueDescription> getValueDescription() const {
            return getDefaultDescriptionShared<T>();
        }

    private:
        std::shared_ptr<T> data;
    };

    // pin for reading incoming data
    template<typename T>
    struct ReadingPin :
        public IncomingPin
    {
        ReadingPin(Block * block, std::string name) :
            IncomingPin(block, std::move(name)) {
        }

        std::shared_ptr<const T> const & get() const {
            return data;
        }

        T const & operator*() const {
            return *data;
        }

        T const * operator->() const {
            return data.get();
        }

        void set(std::shared_ptr<T> value) {
            data = std::move(value);
        }

        template<typename U = T, typename... Ts>
        U * set(Ts... args) {
            auto item = std::make_shared<U>(std::forward<Ts...>(args...));
            data = std::static_pointer_cast<T>(item);
            return item.get();
        }

        std::shared_ptr<const ValueDescription> getValueDescription() const {
            return getDefaultDescriptionShared<T>();
        }

        void readFrom(OutgoingPin * pin) {
            auto item = static_cast<WritingPin<T> *>(pin);
            data = item->get();
        }

    private:
        std::shared_ptr<const T> data;
    };

    // bus for reading from multiple incoming sources
    template<typename T>
    struct ReadingBus :
        public IncomingBus
    {
        ReadingBus(Block * block, std::string name) :
            IncomingBus(block, std::move(name)) {
        }

        std::shared_ptr<const ValueDescription> getValueDescription() const {
            return getDefaultDescriptionShared<T>();
        }

    private:
        std::shared_ptr<IncomingPin> createPin(Block * block, std::string name) {
            return std::make_shared<ReadingPin<T>>(block, std::move(name));
        }
    };

    // streaming handler types
    template<typename T>
    struct Stream {
        std::function<void(T const &)> pushHandler;
        std::function<void()> doneHandler;
    };

    template<typename T>
    struct DefaultDescription<Stream<T>> :
        public ValueDescriptionI<Stream<T>, ValueKind::ANY>
    {
        DefaultDescription(ValueDescriptionT<T> * inner) :
            inner(inner) {
        }

        DefaultDescription(std::shared_ptr<const ValueDescriptionT<T>> inner
                           = getDefaultDescriptionShared((T *)0)) :
            inner(inner) {
        }

        std::shared_ptr<const ValueDescriptionT<T>> inner;
    };

    // pin for producing streaming data
    template<typename T>
    struct PushingPin :
        public ReadingBus<Stream<T>>
    {
        PushingPin(Block * block, std::string name) :
            ReadingBus<Stream<T>>(block, std::move(name)) {
        }

        void push(T const & value) {
            for(auto & item : this->getIncomingPins()) {
                auto & pin = *std::static_pointer_cast<ReadingPin<Stream<T>>>(item);
                pin->pushHandler(value);
            }
        }

        void done() {
            for(auto & item : this->getIncomingPins()) {
                auto & pin = *std::static_pointer_cast<ReadingPin<Stream<T>>>(item);
                pin->doneHandler();
            }
        }
    };

    // pin for consuming streaming data
    template<typename T>
    struct PullingPin :
        public WritingPin<Stream<T>>
    {
        PullingPin(Block * block, std::string name) :
            WritingPin<Stream<T>>(block, std::move(name)) {
            auto stream = std::make_shared<Stream<T>>();
            this->set(stream);
        }
    };
}

