/*
 * relative_win_cost_model.cc
 * added by Nikolay 26 May 2015
   Copyright (c) 2015.  All rights reserved.
 */
#include "rtbkit/common/win_cost_model.h"

namespace RTBKIT {

namespace {

struct RelativeWinCostModel {

    static Amount evalute(WinCostModel const & model,  Bid const & bid, Amount const & price)
    {
        Amount priceReturn(price);
        if (model.data.isMember("win")){
            priceReturn.value = (bid.price.value * priceReturn.value) / 1000; // rtbkit converts win price to CPM (e.g. $0.5 = 500CPM), so we need put it back to relative value
        }
      return priceReturn;
    }
};

struct AtInit {
    AtInit()
    {
      PluginInterface<WinCostModel>::registerPlugin("relative", RelativeWinCostModel::evalute);
    }
} atInit;
} // file scope

} // namespace RTBKIT
