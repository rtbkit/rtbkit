/**
 * mongo_init.h
 * Mich, 2015-09-03
 * Copyright (c) 2015 Datacratic. All rights reserved.
 **/
#pragma once

#include "mongo/bson/bson.h"
#include "mongo/util/net/hostandport.h"


using namespace mongo;

namespace Datacratic {
    bool _mongoInitialized;

struct MongoAtInit {

    MongoAtInit() {
        if (!_mongoInitialized) {
            _mongoInitialized = true;
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
            using mongo::client::initialize;
            using mongo::client::Options;
            auto status = initialize();
            if (!status.isOK()) {
                throw ML::Exception("Mongo initialize failed");
            }
        }
    }
} atInit;

}
