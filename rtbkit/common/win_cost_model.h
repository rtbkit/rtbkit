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
#include "rtbkit/common/plugin_interface.h"

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

    // FIXME: this is being kept just for compatibility reasons.
    // we don't want to break compatibility now, although this interface does not make
    // sense any longer  
    // so any use of it should be considered deprecated
    static void registerModel(const std::string & name,
                              Model model)
    {
        PluginInterface<WinCostModel>::registerPlugin(name, model);
    }
  
    // --- plugin interface init
    // plugin interface expects this type to be called Factory
    typedef Model Factory;
    // plugin interface needs to be able to request the root name of the plugin library
    static const std::string libNameSufix() {return "win_cost_model";};


    static void createDescription(WinCostModelDescription&);

public:
    std::string name;
    Json::Value data;
};

IMPL_SERIALIZE_RECONSTITUTE(WinCostModel);

CREATE_CLASS_DESCRIPTION(WinCostModel)

} // namespace RTBKIT

