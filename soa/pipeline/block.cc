/* block.cc
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

Block::Block() :
    pipeline(nullptr),
    parent(nullptr) {
    //debug.deactivate();
}

Block::Block(Pipeline * pipeline) :
    pipeline(pipeline),
    parent(nullptr) {
    //debug.deactivate();
}

Pipeline * Block::getPipeline() const {
    return pipeline;
}

std::string const & Block::getName() const {
    return name;
}

std::string const & Block::getPath() const {
    return path;
}

Connector * Block::createConnector(IncomingPin * incoming, OutgoingPin * outgoing) {
    return pipeline->createConnector(incoming, outgoing);
}

void Block::add(Pin * item) {
    if(item->isIncomingPin()) {
        incomings.push_back(static_cast<IncomingPin *>(item));
    }

    if(item->isOutgoingPin()) {
        outgoings.push_back(static_cast<OutgoingPin *>(item));
    }
}

std::vector<IncomingPin *> const & Block::getIncomingPins() const {
    return incomings;
}

std::vector<OutgoingPin *> const & Block::getOutgoingPins() const {
    return outgoings;
}

void Block::run() {
}

Blocks::Blocks() {
}

Blocks::Blocks(Pipeline * pipeline) :
    Block(pipeline) {
}

std::vector<std::shared_ptr<Block>> const & Blocks::getBlocks() const {
    return blocks;
}

void Blocks::add(std::shared_ptr<Block> item, std::string name) {
    item->pipeline = pipeline;
    item->parent = this;
    item->path = path + "/" + name;
    item->name = std::move(name);
    LOG(debug) << "created block name='" << item->name << "' path='" << item->path << "'" << std::endl;
    blocks.push_back(std::move(item));
}

Logging::Category Block::print("print");
Logging::Category Block::error("error", print);
Logging::Category Block::trace("trace", print);
Logging::Category Block::debug("debug", trace);
