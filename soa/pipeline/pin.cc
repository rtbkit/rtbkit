/* pin.cc
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

Pin::Pin(Block * block, std::string name) :
    name(std::move(name)),
    block(block),
    incoming(false),
    outgoing(false),
    bus(false) {
}

std::string Pin::getName() const {
    return name;
}

std::string Pin::getPath() const {
    return block->getPath() + "/" + name;
}

bool Pin::isIncomingPin() const {
    return incoming;
}

bool Pin::isOutgoingPin() const {
    return outgoing;
}

Block * Pin::getBlock() const {
    return block;
}

Connector * Pin::connectWith(Pin * pin) {
    if(!pin) {
        disconnect();
        return nullptr;
    }

    if(outgoing) {
        if(pin->outgoing) {
            THROW(block->error) << "cannot connect 2 outgoing pins" << std::endl;
        }

        return pin->connectWith(this);
    }

    if(pin->incoming && !pin->bus) {
        auto item = static_cast<IncomingPin *>(pin)->getConnector();
        if(item) {
            return connectWith(item->getOutgoingPin());
        }
    }
    else {
        if(isCompatibleWith(pin)) {
            return createConnector(pin);
        }
        else {
            auto a = getValueDescription();
            auto b = pin->getValueDescription();
            THROW(block->error) << "pins types aren't compatibles incoming='"
                                     << (a ? a->typeName : "?")
                                     << "' outgoing='"
                                     << (b ? b->typeName : "?")
                                     << "'"
                                     << std::endl;
        }
    }

    return nullptr;
}

Connector * Pin::connectWith(Pin & pin) {
    return connectWith(&pin);
}

void Pin::disconnect() {
    onDeleteConnector(nullptr);
}

void Pin::disconnectFrom(Connector * connector) {
    onDeleteConnector(connector);
}

std::shared_ptr<const ValueDescription> Pin::getValueDescription() const {
    return nullptr;
}

Connector * Pin::createConnector(Pin * pin) {
    auto incoming = getAsIncomingPin();
    auto outgoing = pin->getAsOutgoingPin();

    LOG(block->debug) << "connecting '"
                      << incoming->getPath()
                      << "' and '"
                      << outgoing->getPath()
                      << "'"
                      << std::endl;

    auto item = block->createConnector(incoming, outgoing);
    static_cast<Pin *>(incoming)->onCreateConnector(item);
    static_cast<Pin *>(outgoing)->onCreateConnector(item);
    return item;
}

IncomingPin * Pin::getAsIncomingPin() {
    return nullptr;
}

OutgoingPin * Pin::getAsOutgoingPin() {
    return nullptr;
}

void Pin::onCreateConnector(Connector * handle) {
}

void Pin::onDeleteConnector(Connector * handle) {
}

bool Pin::isCompatibleWith(Pin * pin) const {
    auto a = getValueDescription();
    auto b = pin->getValueDescription();
    return a && b && a->typeName == b->typeName;
}

Connector::Connector(IncomingPin * incoming, OutgoingPin * outgoing) :
    incoming(incoming),
    outgoing(outgoing) {
}

void Connector::push() {
}

void Connector::pull() {
}

IncomingPin * Connector::getIncomingPin() const {
    return incoming;
}

OutgoingPin * Connector::getOutgoingPin() const {
    return outgoing;
}

void Connector::onDisconnect() {
}

IncomingPin::IncomingPin(Block * block, std::string name) :
    Pin(block, std::move(name)),
    connector(nullptr) {
    incoming = true;
    block->add(this);
}

bool IncomingPin::isConnected() const {
    return !!connector;
}

Connector * IncomingPin::getConnector() const {
    return connector;
}

IncomingPin * IncomingPin::getAsIncomingPin() {
    return this;
}

void IncomingPin::onCreateConnector(Connector * handle) {
    if(connector) {
        disconnect();
    }

    connector = handle;
}

void IncomingPin::onDeleteConnector(Connector * handle) {
    if(handle && connector != handle) {
        THROW(block->error) << "unknown connection" << std::endl;
    }

    connector = nullptr;
}

IncomingBus::IncomingBus(Block * block, std::string name) :
    Pin(block, std::move(name)) {
    incoming = true;
    bus = true;
}

std::vector<std::shared_ptr<IncomingPin>> const & IncomingBus::getIncomingPins() const {
    return pins;
}

IncomingPin * IncomingBus::getAsIncomingPin() {
    auto pin = createPin(block, name + "-" + std::to_string(pins.size()));
    pins.push_back(pin);
    return pin.get();
}

OutgoingPin::OutgoingPin(Block * block, std::string name) :
    Pin(block, std::move(name)) {
    outgoing = true;
    block->add(this);
}

bool OutgoingPin::isConnected() const {
    return !connectors.empty();
}

std::vector<Connector *> const & OutgoingPin::getConnectors() const {
    return connectors;
}

OutgoingPin * OutgoingPin::getAsOutgoingPin() {
    return this;
}

void OutgoingPin::onCreateConnector(Connector * handle) {
    auto i = std::find(connectors.begin(), connectors.end(), handle);
    if(connectors.end() != i) {
        THROW(block->error) << "duplicate connection" << std::endl;
    }

    connectors.push_back(handle);
}

void OutgoingPin::onDeleteConnector(Connector * handle) {
    if(handle) {
        auto i = std::find(connectors.begin(), connectors.end(), handle);
        if(i == connectors.end()) {
            THROW(block->error) << "unknown connection" << std::endl;
        }

        handle->onDisconnect();

        *i = std::move(connectors.back());
        connectors.pop_back();
    }
    else {
        for(auto item : connectors) {
            item->onDisconnect();
        }

        connectors.clear();
    }
}

