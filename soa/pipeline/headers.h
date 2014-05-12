/* headers.h
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include <string>
#include <vector>
#include <set>

namespace Datacratic
{
    struct Block;
    struct Pipeline;
    struct IncomingPin;
    struct OutgoingPin;
    struct Connector;
}

#include "soa/types/basic_value_descriptions.h"
#include "soa/service/logs.h"
#include "soa/pipeline/pin.h"
#include "soa/pipeline/block.h"
#include "soa/pipeline/pipeline.h"
#include "soa/pipeline/default_pipeline.h"
#include "soa/pipeline/file_reader_block.h"
#include "soa/pipeline/file_writer_block.h"
#include "soa/pipeline/importer_block.h"

