/* win_cost_model.h                                                    -*- C++ -*-
   Eric Robert, 13 May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Win cost model provides a function describing the costs associated with
   winning at a given price.

*/

#pragma once

#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/bids.h"
#include "rtbkit/core/agent_configuration/agent_config.h"

namespace RTBKIT {

class WinCostModelDescription;

using namespace Datacratic;

/*****************************************************************************/
/* WIN COST MODEL                                                            */
/*****************************************************************************/

struct WinCostModel {
    WinCostModel();
    WinCostModel(std::string name, Json::Value data);

    /// Get the win cost from the model
    Amount evaluate(Bid const & bid, Amount const & price) const;

    Json::Value toJson() const;
    static WinCostModel fromJson(Json::Value const & json);

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    /// Model signature
    typedef std::function<Amount (WinCostModel const & model,
                                  Bid const & bid,
                                  Amount const & price)> Model;

    /// Register a given type of model.
    /// Should be done in a static initilalizer on shared library load.
    static void registerModel(const std::string & name,
                              Model model);

    static void createDescription(WinCostModelDescription&);

public:
    std::string name;
    Json::Value data;
};

IMPL_SERIALIZE_RECONSTITUTE(WinCostModel);

CREATE_CLASS_DESCRIPTION(WinCostModel)

} // namespace RTBKIT

