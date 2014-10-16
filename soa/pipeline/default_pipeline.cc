/* default_pipeline.cc
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

DefaultPipeline::DefaultPipeline() {
}

void DefaultPipeline::run() {
    states.clear();
    ready.clear();

    for(auto item : getBlocks()) {
        State state;
        state.pipeline = this;
        state.block = item.get();
        state.count = 0;
        for(auto pin : item->getIncomingPins()) {
            if(pin->isConnected()) {
                state.count++;
            }
        }

        if(state.count == 0) {
            LOG(debug) << "block ready to run name='" << state.block->getPath() << "'" << std::endl;
            ready.push_back(state.block);
        }
        else {
            states[state.block] = state;
        }
    }

    for(auto & item : connectors) {
        auto block = item->getIncomingPin()->getBlock();
        auto state = &states[block];
        item->state = state;
    }

    while(!ready.empty()) {
        auto item = ready.back();
        ready.pop_back();
        LOG(debug) << "running block='" << item->getPath() << "'" << std::endl;
        item->run();
    }
}

Connector * DefaultPipeline::createConnector(IncomingPin * incoming, OutgoingPin * outgoing) {
    auto item = std::make_shared<DefaultConnector>(incoming, outgoing);
    connectors.insert(item);
    return item.get();
}

DefaultPipeline::
DefaultConnector::DefaultConnector(IncomingPin * incoming, OutgoingPin * outgoing) :
    Connector(incoming, outgoing),
    state(nullptr) {
}

void DefaultPipeline::DefaultConnector::push() {
    auto incoming = getIncomingPin();
    auto outgoing = getOutgoingPin();
    LOG(state->pipeline->debug) << "push from '" << outgoing->getPath() << "'" << std::endl;
    incoming->readFrom(outgoing);
    --state->count;
    if(state->count == 0) {
        LOG(state->pipeline->debug) << "block ready to run name='" << state->block->getPath() << "'" << std::endl;
        state->pipeline->ready.push_back(state->block);
    }
}

