/* null_banker.cc
   Jeremy Barnes, 11 October 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Null banker implementation.
*/

#include "null_banker.h"

using namespace std;
using namespace ML;

namespace RTBKIT {


/*****************************************************************************/
/* NULL BANKER                                                               */
/*****************************************************************************/

void
NullBudgetController::
addAccountSync(const AccountKey & account)
{
    cerr << "added account " << account << endl;
}

void
NullBudgetController::
topupTransferSync(const AccountKey & account,
                  CurrencyPool amount)
{
    cerr << "transferring " << amount << " to " << account << endl;
}

void
NullBudgetController::
setBudgetSync(const std::string & topLevelCampaign,
              CurrencyPool amount)
{
    cerr << "setting budget for " << topLevelCampaign
         << " to " << amount << endl;
}

void
NullBudgetController::
addBudgetSync(const std::string & topLevelCampaign,
              CurrencyPool amount)
{
    cerr << "adding budget for " << topLevelCampaign
         << " to " << amount << endl;
}


/*****************************************************************************/
/* NULL BANKER                                                               */
/*****************************************************************************/

NullBanker::NullBanker(bool authorize, const std::string & serviceName)
    : authorize_(authorize), serviceName_(serviceName)
{
}
//----------------------------------------------------------------------
bool
NullBanker::
authorizeBid(const AccountKey & account,
             const std::string & item,
             Amount amountToAuthorize)
{
    return authorize_;
}
//----------------------------------------------------------------------
void
NullBanker::
commitBid(const AccountKey & account,
          const std::string & item,
          Amount amountPaid,
          const LineItems & lineItems)
{
}

//----------------------------------------------------------------------
void
NullBanker::
forceWinBid(const AccountKey & account,
            Amount amountPaid,
            const LineItems & lineItems)
{
}

MonitorIndicator
NullBanker::
getProviderIndicators() const
{
    MonitorIndicator ind;
    ind.serviceName = serviceName_;
    ind.status = true;
    ind.message = "NullBanker: OK";
    return ind;
}

#if 0
//----------------------------------------------------------------------
std::string NullBanker::getCampaignStatusStr(const std::string & campaign,
                                             const std::string &strategy) const
{
    return "{}";
}
//----------------------------------------------------------------------
Json::Value NullBanker::getCampaignStatusJson(const std::string & campaign,
        const std::string &strategy) const
{
    return Json::Value();
}
//----------------------------------------------------------------------
Json::Value NullBanker::dumpAllCampaignsJson() const
{
    return Json::Value();
}
#endif
#if 0
//----------------------------------------------------------------------
Banker::OutstandingStats
NullBanker::
totalOutstandingStats(const std::string & campaign,
                      const std::string & strategy) const
{
    return OutstandingStats();
}
#endif

} // namespace RTBKIT
