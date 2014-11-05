/* banker_account_test.cc
   Jeremy Barnes, 16 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Test for Banker accounts.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "jml/arch/format.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/guard.h"
#include "rtbkit/common/account_key.h"
#include "rtbkit/core/banker/account.h"
#include "jml/utils/environment.h"
#include <boost/thread/thread.hpp>
#include "jml/arch/atomic_ops.h"
#include "jml/arch/timers.h"
#include "jml/utils/ring_buffer.h"


using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_account_set_budget )
{
    Account account;

    /* set initial budget */
    account.setBudget(USD(8));
    BOOST_CHECK_EQUAL(account.balance, USD(8));
    BOOST_CHECK_EQUAL(account.budgetIncreases, USD(8));
    BOOST_CHECK_EQUAL(account.budgetDecreases, USD(0));

    /* adjust budget down:
       1 usd added to adjustmentsOut (deduced from budget) */
    account.setBudget(USD(7));
    BOOST_CHECK_EQUAL(account.balance, USD(7));
    BOOST_CHECK_EQUAL(account.budgetIncreases, USD(8));
    BOOST_CHECK_EQUAL(account.budgetDecreases, USD(1));

    /* adjust budget up:
       1 usd added to adjustmentsIn to balance with adj.Out */
    account.setBudget(USD(8));
    BOOST_CHECK_EQUAL(account.balance, USD(8));
    BOOST_CHECK_EQUAL(account.budgetIncreases, USD(9));
    BOOST_CHECK_EQUAL(account.budgetDecreases, USD(1));

    /* adjust budget up:
       3 usd put in budget */
    account.setBudget(USD(13));
    BOOST_CHECK_EQUAL(account.balance, USD(13));
    BOOST_CHECK_EQUAL(account.budgetIncreases, USD(14));
    BOOST_CHECK_EQUAL(account.budgetDecreases, USD(1));

    /* negative adjustments must be limited by "available":
       of the previous 13 usd budget, 10 already have been spent, which means
       we cannot go below 10 USD, even though 3 USD are still available
     */
    account.allocatedOut = USD(10);
    account.balance = USD(3);
    account.checkInvariants();
    account.setBudget(USD(9));
    BOOST_CHECK_EQUAL(account.balance, USD(0));
    BOOST_CHECK_EQUAL(account.budgetIncreases, USD(14));
    BOOST_CHECK_EQUAL(account.budgetDecreases, USD(4));

    /* we adjust the budget down the the least possible value and ensure that
       "available" is adjusted by taking the "allocatedOut" into account */
    account.setBudget(USD(10));
    BOOST_CHECK_EQUAL(account.balance, USD(0));
}

BOOST_AUTO_TEST_CASE( test_account_tojson )
{
    Account account;

    Json::Value testState = Json::parse(
        "{ 'md':  { 'objectType': 'Account',"
        "           'version': 1 },"
        "  'type': 'none',"
        "  'budgetIncreases': {},"
        "  'budgetDecreases': {},"
        "  'spent': {},"
        "  'status': 'active',"
        "  'recycledIn': {},"
        "  'recycledOut': {},"
        "  'allocatedIn': {},"
        "  'allocatedOut': {},"
        "  'commitmentsMade': {},"
        "  'commitmentsRetired': {},"
        "  'adjustmentsIn': {},"
        "  'adjustmentsOut': {},"
        "  'lineItems': {},"
        "  'adjustmentLineItems': {}}");

    /* fresh and clean account */
    BOOST_CHECK_EQUAL(account.toJson(), testState);

    /* account with a 10 USD budget */
    account.setBudget(USD(10));
    testState["budgetIncreases"]["USD/1M"] = 10000000;
    BOOST_CHECK_EQUAL(account.toJson(), testState);
}

BOOST_AUTO_TEST_CASE( test_account_hierarchy )
{
    Account budgetAccount;
    budgetAccount.setBudget(USD(10));

    Account commitmentAccount, spendAccount;

    ShadowAccount shadowCommitmentAccount;
    ShadowAccount shadowSpendAccount;

    commitmentAccount.setBalance(budgetAccount, USD(2));

    BOOST_CHECK_EQUAL(budgetAccount.balance, USD(8));
    BOOST_CHECK_EQUAL(commitmentAccount.balance, USD(2));

    shadowCommitmentAccount.syncFromMaster(commitmentAccount);
    shadowSpendAccount.syncFromMaster(spendAccount);

    BOOST_CHECK_EQUAL(shadowCommitmentAccount.balance, USD(2));
    BOOST_CHECK_EQUAL(shadowSpendAccount.balance, USD(0));


    auto doBidding = [&] ()
        {
            bool auth1 = shadowCommitmentAccount.authorizeBid("ad1", USD(1));
            bool auth2 = shadowCommitmentAccount.authorizeBid("ad2", USD(1));
            bool auth3 = shadowCommitmentAccount.authorizeBid("ad3", USD(1));

            BOOST_CHECK_EQUAL(auth1, true);
            BOOST_CHECK_EQUAL(auth2, true);
            BOOST_CHECK_EQUAL(auth3, false);
    
            Amount detached = shadowCommitmentAccount.detachBid("ad1");
            BOOST_CHECK_EQUAL(detached, USD(1));

            shadowCommitmentAccount.cancelBid("ad2");

            shadowSpendAccount.commitDetachedBid(detached, USD(0.50), LineItems());

            shadowCommitmentAccount.syncToMaster(commitmentAccount);
            shadowSpendAccount.syncToMaster(spendAccount);
        };

    // Do the same kind of bid 5 times
    for (unsigned i = 0;  i < 5;  ++i) {

        doBidding();

        cerr << "budget" << budgetAccount << endl;
        cerr << "spend " << spendAccount << endl;
        cerr << "commitment " << commitmentAccount << endl;
        cerr << "shadow spend" << shadowSpendAccount << endl;
        cerr << "shadow commitment" << shadowCommitmentAccount << endl;

        spendAccount.recuperateTo(budgetAccount);

        cerr << "after recuperation" << endl;
        cerr << "budget" << budgetAccount << endl;
        cerr << "spend " << spendAccount << endl;
   
        commitmentAccount.setBalance(budgetAccount, USD(2));
        
        cerr << "after setBalance" << endl;
        cerr << "budget" << budgetAccount << endl;
        cerr << "spend " << spendAccount << endl;
        cerr << "commitment " << commitmentAccount << endl;

        shadowCommitmentAccount.syncFromMaster(commitmentAccount);
        shadowSpendAccount.syncFromMaster(spendAccount);

        cerr << "after sync" << endl;
        cerr << "shadow spend" << shadowSpendAccount << endl;
        cerr << "shadow commitment" << shadowCommitmentAccount << endl;

        BOOST_CHECK_EQUAL(commitmentAccount.balance, USD(2));
        BOOST_CHECK_EQUAL(shadowCommitmentAccount.balance, USD(2));
        BOOST_CHECK_EQUAL(spendAccount.balance, USD(0));
        BOOST_CHECK_EQUAL(shadowSpendAccount.balance, USD(0));
    }
}

BOOST_AUTO_TEST_CASE( test_account_recycling )
{
    Accounts accounts;

    AccountKey campaign("campaign");
    AccountKey strategy("campaign:strategy");
    AccountKey strategy2("campaign:strategy2");
    AccountKey spend("campaign:strategy:spend");
    AccountKey spend2("campaign:strategy2:spend");

    accounts.createBudgetAccount(campaign);
    accounts.createBudgetAccount(strategy);
    accounts.createBudgetAccount(strategy2);
    accounts.createSpendAccount(spend);
    accounts.createSpendAccount(spend2);

    // Top level budget of $10
    accounts.setBudget(campaign, USD(10));

    // Make $2 available in the strategy account
    accounts.setBalance(strategy, USD(2), AT_NONE);
    accounts.setBalance(strategy2, USD(2), AT_NONE);
    
    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(6));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(2));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy2), USD(2));

    accounts.setBalance(spend, USD(1), AT_NONE);
    //accounts.setBalance(spend2, USD(1), AT_NONE);

    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(6));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy2), USD(2));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend2), USD(0));

    accounts.setBalance(spend, USD(1), AT_NONE);
    //accounts.setBalance(spend2, USD(1), AT_NONE);

    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(6));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy2), USD(2));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend2), USD(0));

    accounts.setBalance(strategy, USD(2), AT_NONE);
    //accounts.setBalance(strategy2, USD(2), AT_NONE);

    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(5));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(2));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy2), USD(2));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend2), USD(0));
}

BOOST_AUTO_TEST_CASE( test_account_close )
{
    Accounts accounts;

    AccountKey campaign("campaign");
    AccountKey strategy("campaign:strategy");
    AccountKey strategy2("campaign:strategy2");
    AccountKey spend("campaign:strategy:spend");
    AccountKey spend2("campaign:strategy:spend2");
    AccountKey spend3("campaign:strategy2:spend");

    accounts.createBudgetAccount(campaign);
    accounts.createBudgetAccount(strategy);
    accounts.createBudgetAccount(strategy2);
    accounts.createSpendAccount(spend);
    accounts.createSpendAccount(spend2);
    accounts.createSpendAccount(spend3);

    // Top level budget of $10
    accounts.setBudget(campaign, USD(10));

    // Make $2 available in the strategy account
    accounts.setBalance(strategy, USD(3), AT_NONE);
    accounts.setBalance(strategy2, USD(2), AT_NONE);
    
    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(5));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(3));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy2), USD(2));

    accounts.setBalance(spend, USD(1), AT_NONE);
    accounts.setBalance(spend2, USD(1), AT_NONE);
    accounts.setBalance(spend3, USD(1), AT_NONE);

    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(5));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy2), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend2), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend3), USD(1));
    
    accounts.closeAccount(strategy);

    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(8));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(0));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy2), USD(1));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend), USD(0));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend2), USD(0));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend3), USD(1));

    accounts.closeAccount(campaign);

    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(10));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(0));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy2), USD(0));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend), USD(0));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend2), USD(0));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend3), USD(0));


    // check if accounts are closed.
    BOOST_CHECK_EQUAL(accounts.getAccount(campaign).status, Account::CLOSED);
    BOOST_CHECK_EQUAL(accounts.getAccount(strategy).status, Account::CLOSED);
   
    accounts.setBalance(strategy, USD(5), AT_NONE); 
    accounts.reactivateAccount(strategy);

    BOOST_CHECK_EQUAL(accounts.getAccount(campaign).status, Account::ACTIVE);
    BOOST_CHECK_EQUAL(accounts.getAccount(strategy).status, Account::ACTIVE);

}


BOOST_AUTO_TEST_CASE( test_account_close_with_spend )
{
    Accounts accounts;

    AccountKey campaign("campaign");
    AccountKey strategy("campaign:strategy");
    AccountKey spend("campaign:strategy:spend");

    accounts.createBudgetAccount(campaign);
    accounts.createBudgetAccount(strategy);
    accounts.createSpendAccount(spend);

    ShadowAccounts shadow; 

    // Top level budget of $10
    accounts.setBudget(campaign, USD(10));

    // Make $2 available in the strategy account
    accounts.setBalance(strategy, USD(4), AT_NONE);
    
    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(6));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(4));

    accounts.setBalance(spend, USD(2), AT_NONE);

    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(6));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(2));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend), USD(2));

    // Bid on an ad
    shadow.activateAccount(spend);
    shadow.syncFrom(accounts);
 
    bool auth = shadow.authorizeBid(spend, "ad1", USD(1));
    BOOST_CHECK_EQUAL(auth, true);

    shadow.commitBid(spend, "ad1", USD(1), LineItems());

    shadow.syncTo(accounts);
    
    cerr << "before close account" << endl;
    cerr << accounts.getAccountSummary(campaign) << endl;
    cerr << accounts.getAccount(campaign) << endl;
    cerr << accounts.getAccount(strategy) << endl;
    cerr << accounts.getAccount(spend) << endl;

    accounts.closeAccount(campaign);
    
    cerr << "after close account" << endl;
    cerr << accounts.getAccountSummary(campaign) << endl;
    cerr << accounts.getAccount(campaign) << endl;
    cerr << accounts.getAccount(strategy) << endl;
    cerr << accounts.getAccount(spend) << endl;

    BOOST_CHECK_EQUAL(accounts.getBalance(campaign), USD(9));
    BOOST_CHECK_EQUAL(accounts.getBalance(strategy), USD(0));
    BOOST_CHECK_EQUAL(accounts.getBalance(spend), USD(0));

    // check if accounts are closed.
    BOOST_CHECK_EQUAL(accounts.getAccount(campaign).status, Account::CLOSED);
    BOOST_CHECK_EQUAL(accounts.getAccount(strategy).status, Account::CLOSED);

}

BOOST_AUTO_TEST_CASE( test_accounts )
{
    Accounts accounts;

    AccountKey budget("budget");
    AccountKey commitment("budget:commitment");
    AccountKey spend("budget:spend");

    ShadowAccounts shadow;

    accounts.createBudgetAccount(budget);
    accounts.createSpendAccount(commitment);
    accounts.createSpendAccount(spend);

    // Top level budget of $10
    accounts.setBudget(budget, USD(10));

    // Make $2 available in the commitment account
    accounts.setBalance(commitment, USD(2), AT_SPEND);
    
    BOOST_CHECK_EQUAL(accounts.getBalance(budget), USD(8));
    BOOST_CHECK_EQUAL(accounts.getBalance(commitment), USD(2));

    shadow.activateAccount(commitment);
    shadow.activateAccount(spend);

    auto doBidding = [&] ()
        {
            shadow.syncFrom(accounts);

            bool auth1 = shadow.authorizeBid(commitment, "ad1", USD(1));
            bool auth2 = shadow.authorizeBid(commitment, "ad2", USD(1));
            bool auth3 = shadow.authorizeBid(commitment, "ad3", USD(1));

            BOOST_CHECK_EQUAL(auth1, true);
            BOOST_CHECK_EQUAL(auth2, true);
            BOOST_CHECK_EQUAL(auth3, false);

            shadow.checkInvariants();

            Amount detached = shadow.detachBid(commitment, "ad1");
            BOOST_CHECK_EQUAL(detached, USD(1));

            shadow.checkInvariants();

            shadow.cancelBid(commitment, "ad2");

            shadow.checkInvariants();

            shadow.commitDetachedBid(spend, detached, USD(0.50), LineItems());

            shadow.syncTo(accounts);

            accounts.checkInvariants();

            cerr << accounts.getAccountSummary(budget) << endl;
    
        };

    // Do the same kind of bid 5 times
    for (unsigned i = 0;  i < 5;  ++i) {

        cerr << accounts.getAccountSummary(budget) << endl;
        cerr << accounts.getAccount(budget) << endl;
        cerr << accounts.getAccount(commitment) << endl;
        cerr << accounts.getAccount(spend) << endl;

        doBidding();

        //cerr << "budget" << budgetAccount << endl;
        //cerr << "spend " << spendAccount << endl;
        //cerr << "commitment " << commitmentAccount << endl;
    
        accounts.recuperate(spend);

        accounts.checkInvariants();

        //cerr << "after recuperation" << endl;
        //cerr << "budget" << budgetAccount << endl;
        //cerr << "spend " << spendAccount << endl;
   
        accounts.setBalance(commitment, USD(2), AT_SPEND);
        
        accounts.checkInvariants();

        //cerr << "after setBalance" << endl;
        //cerr << "budget" << budgetAccount << endl;
        //cerr << "spend " << spendAccount << endl;
        //cerr << "commitment " << commitmentAccount << endl;
    }

    cerr << accounts.getAccountSummary(budget) << endl;
}

BOOST_AUTO_TEST_CASE( test_multiple_bidder_threads )
{
    Accounts master;

    AccountKey campaign("campaign");
    AccountKey strategy("campaign:strategy");

    // Create a budget for the campaign
    master.createBudgetAccount(strategy);
    master.setBudget(campaign, USD(10));

    // Do 1,000 topup transfers of one micro

    int nTopupThreads = 2;
    int nAddBudgetThreads = 2;
    int nBidThreads = 2; 
    //int nSpendThreads = 2;
    int numTransfersPerThread = 10000;
    int numAddBudgetsPerThread = 10;

    volatile bool finished = false;

    auto runTopupThread = [&] ()
        {
            while (!finished) {
                master.setBalance(strategy, USD(0.10), AT_BUDGET);
            }
        };

    auto runAddBudgetThread = [&] ()
        {
            for (unsigned i = 0;  i < numAddBudgetsPerThread;  ++i) {
                
                AccountSummary summary = master.getAccountSummary(campaign);
                cerr << summary << endl;
                master.setBudget(campaign, summary.budget + USD(1));

                ML::sleep(1.0);
            }
        };

    uint64_t numBidsCommitted = 0;

    ML::RingBufferSRMW<Amount> toCommitThread(1000000);
    

    auto runBidThread = [&] (int threadNum)
        {
            ShadowAccounts shadow;
            AccountKey account = strategy;
            account.push_back("bid" + to_string(threadNum));

            master.createSpendAccount(account);
            shadow.activateAccount(account);
            shadow.syncFrom(master);

            int done = 0;
            for (;  !finished;  ++done) {
                string item = "item";

                // Every little bit, do a sync and a re-up
                if (done && done % 1000 == 0) {
                    shadow.syncTo(master);
                    master.setBalance(account, USD(0.10), AT_NONE);
                    shadow.syncFrom(master);
                    //cerr << "done " << done << " bids" << endl;
                }
                
                // Authorize 10
                if (!shadow.authorizeBid(account, item, MicroUSD(1))) {
                    continue;
                }

                // In half of the cases, we cancel.  In the other half, we
                // transfer it off to the commit thread

                if (done % 2 == 0) {
                    // Commit 1
                    shadow.commitBid(account, item, MicroUSD(1), LineItems());
                    ML::atomic_inc(numBidsCommitted);
                }
                else {
                    Amount amount = shadow.detachBid(account, item);
                    toCommitThread.push(amount);
                }
            }

            shadow.sync(master);

            cerr << "finished shadow account with "
                 << done << " bids" << endl;
            cerr << master.getAccount(account) << endl;

        };

    auto runCommitThread = [&] (int threadNum)
        {
            ShadowAccounts shadow;
            AccountKey account = strategy;
            account.push_back("commit" + to_string(threadNum));

            master.createSpendAccount(account);
            shadow.activateAccount(account);
            shadow.syncFrom(master);

            while (!finished || toCommitThread.couldPop()) {
                Amount amount;
                if (toCommitThread.tryPop(amount, 0.1)) {
                    shadow.commitDetachedBid(account, amount, MicroUSD(1), LineItems());
                    ML::atomic_inc(numBidsCommitted);
                }
                shadow.syncTo(master);
            }

            shadow.syncTo(master);
            cerr << "done commit thread" << endl;
        };

    boost::thread_group budgetThreads;

    for (unsigned i = 0;  i < nAddBudgetThreads;  ++i)
        budgetThreads.create_thread(runAddBudgetThread);

    boost::thread_group bidThreads;
    for (unsigned i = 0;  i < nBidThreads;  ++i)
        bidThreads.create_thread(std::bind(runBidThread, i));

    for (unsigned i = 0;  i < nTopupThreads;  ++i)
        bidThreads.create_thread(runTopupThread);

    bidThreads.create_thread(std::bind(runCommitThread, 0));
    

    budgetThreads.join_all();

    finished = true;

    bidThreads.join_all();

    uint32_t amountAdded       = nAddBudgetThreads * numAddBudgetsPerThread;
    uint32_t amountTransferred = nTopupThreads * numTransfersPerThread;

    cerr << "numBidsCommitted = "  << numBidsCommitted << endl;
    cerr << "amountTransferred = " << amountTransferred << endl;
    cerr << "amountAdded =       " << amountAdded << endl;

    cerr << "campaign" << endl;
    cerr << master.getAccountSummary(campaign) << endl;
    cerr << master.getAccount(campaign) << endl; 

    cerr << "strategy" << endl;
    cerr << master.getAccountSummary(strategy) << endl;
    cerr << master.getAccount(strategy) << endl; 


#if 0    
    RedisBanker banker("bankerTest", "b", s, redis);
    banker.sync();
    Json::Value status = banker.getCampaignStatusJson("testCampaign", "");

    cerr << status << endl;




    BOOST_CHECK_EQUAL(status["available"]["micro-USD"].asInt(), 1000000 - amountTransferred + amountAdded);
    BOOST_CHECK_EQUAL(status["strategies"][0]["available"]["micro-USD"].asInt(),
                      amountTransferred - numBidsCommitted);
    BOOST_CHECK_EQUAL(status["strategies"][0]["transferred"]["micro-USD"].asInt(),
                      amountTransferred);
    BOOST_CHECK_EQUAL(status["strategies"][0]["spent"]["micro-USD"].asInt(),
                      numBidsCommitted);
    BOOST_CHECK_EQUAL(status["spent"]["micro-USD"].asInt(), numBidsCommitted);

    //BOOST_CHECK_EQUAL(status["available"].
#endif
}

BOOST_AUTO_TEST_CASE( test_recycling )
{
    Accounts accounts;
    Account t, s, sp; /* "top", "sub" and "spend" */

    /* setup */
    accounts.createAccount({"t"}, AT_BUDGET);
    accounts.setBudget({"t"}, USD(666));

    accounts.setBalance({"t", "s"}, USD(10), AT_BUDGET);

    // s.setBalance(10)
    // -> as.budgetIncrease increased by 10
    s = accounts.getAccount({"t", "s"});
    BOOST_CHECK_EQUAL(s.budgetIncreases, USD(10));

    // s.setBalance(7)
    // -> t.recycledIn increased by 3 (total: 3)
    // and s.recycledOut increased by 3 (total: 3)
    accounts.setBalance({"t", "s"}, USD(7), AT_NONE);
    t = accounts.getAccount({"t"});
    BOOST_CHECK_EQUAL(t.recycledIn, USD(3));
    s = accounts.getAccount({"t", "s"});
    BOOST_CHECK_EQUAL(s.recycledOut, USD(3));

    // s.setBalance(8)
    // -> t.recycledOut increased by 1 (total: 1)
    // and s.recycledIn increased by 1 (total: 1)
    accounts.setBalance({"t", "s"}, USD(8), AT_NONE);
    t = accounts.getAccount({"t"});
    BOOST_CHECK_EQUAL(t.recycledOut, USD(1));
    s = accounts.getAccount({"t", "s"});
    BOOST_CHECK_EQUAL(s.recycledIn, USD(1));

    // sp.setBalance(5)
    // -> sp.budgetIncrease increased by 5 (total: 5),
    //    s.allocatedOut increased by 5 (total: 5)
    accounts.setBalance({"t", "s", "sp"}, USD(5), AT_SPEND);
    sp = accounts.getAccount({"t", "s", "sp"});
    BOOST_CHECK_EQUAL(sp.budgetIncreases, USD(5));
    s = accounts.getAccount({"t", "s"});
    BOOST_CHECK_EQUAL(s.allocatedOut, USD(5));

    /* mixup */
    // sp.setBalance(4)
    // -> sp.recycledOut by 1 (total: 1),
    // s.recycledIn increased by 1 (total: 2)
    accounts.setBalance({"t", "s", "sp"}, USD(4), AT_NONE);
    sp = accounts.getAccount({"t", "s", "sp"});
    BOOST_CHECK_EQUAL(sp.recycledOut, USD(1));
    s = accounts.getAccount({"t", "s"});
    BOOST_CHECK_EQUAL(s.recycledIn, USD(2));
}

BOOST_AUTO_TEST_CASE( test_getRecycledUp )
{
    Accounts accounts;
    CurrencyPool recycledIn, recycledOut;

    /* setup */
    accounts.createAccount({"t"}, AT_BUDGET);
    accounts.setBudget({"t"}, USD(666));

    // s.setBalance(10)
    accounts.setBalance({"t", "s"}, USD(10), AT_BUDGET);
    accounts.getRecycledUp({"t", "s"}, recycledIn, recycledOut);
    BOOST_CHECK_EQUAL(recycledIn, USD(0));
    BOOST_CHECK_EQUAL(recycledOut, USD(0));
    
    // s.setBalance(7)
    // t.recycledIn == 3 but t.recycledIn(up) == 0
    // s.recycledOut(up) == 3
    accounts.setBalance({"t", "s"}, USD(7), AT_NONE);
    accounts.getRecycledUp({"t"}, recycledIn, recycledOut);
    BOOST_CHECK_EQUAL(recycledIn, USD(0));
    accounts.getRecycledUp({"t", "s"}, recycledIn, recycledOut);
    BOOST_CHECK_EQUAL(recycledOut, USD(3));

    // s.setBalance(8)
    // t.recycledOut == 1 but t.recycledOut(up) == 0
    // s.recycledIn == 1 and s.recycledIn(up) == 1
    accounts.setBalance({"t", "s"}, USD(8), AT_NONE);
    accounts.getRecycledUp({"t"}, recycledIn, recycledOut);
    BOOST_CHECK_EQUAL(recycledOut, USD(0));
    accounts.getRecycledUp({"t", "s"}, recycledIn, recycledOut);
    BOOST_CHECK_EQUAL(recycledIn, USD(1));

    // sp.setBalance(5)
    // sp.recycleX untouched and sp.budgetIncreases increased
    accounts.setBalance({"t", "s", "sp"}, USD(5), AT_SPEND);
    accounts.getRecycledUp({"t", "s", "sp"}, recycledIn, recycledOut);
    BOOST_CHECK_EQUAL(recycledIn, USD(0));
    BOOST_CHECK_EQUAL(recycledOut, USD(0));

    // sp.setBalance(4)
    // s.recycledIn == 2 but s.recycledIn(up) == 1
    // sp.recycledOut == 1 and sp.recycledOut(up) == 1
    accounts.setBalance({"t", "s", "sp"}, USD(4), AT_NONE);
    accounts.getRecycledUp({"t", "s"}, recycledIn, recycledOut);
    BOOST_CHECK_EQUAL(recycledIn, USD(1));
    accounts.getRecycledUp({"t", "s", "sp"}, recycledIn, recycledOut);
    BOOST_CHECK_EQUAL(recycledOut, USD(1));
}

/* ensure values of simple account summaries matches those in non-simple
   ones */
BOOST_AUTO_TEST_CASE( test_account_simple_summaries )
{
    Accounts accounts;

    /* NOTE: the accounts are not particularly consistent with one another */
    Json::Value jsonValue
        = Json::parse("{'adjustmentLineItems':{},"
                      "'adjustmentsIn':{},"
                      "'adjustmentsOut':{},"
                      "'allocatedIn':{},"
                      "'allocatedOut':{'USD/1M':46571708796},"
                      "'budgetDecreases':{},"
                      "'budgetIncreases':{'USD/1M':52947000000},"
                      "'commitmentsMade':{},"
                      "'commitmentsRetired':{},"
                      "'lineItems':{},"
                      "'md':{'objectType':'Account','version':1},"
                      "'recycledIn':{},"
                      "'recycledOut':{},"
                      "'spent':{},"
                      "'type':'budget'}");
    accounts.restoreAccount({"top"}, jsonValue);
    jsonValue = Json::parse("{'adjustmentLineItems':{},"
                            "'adjustmentsIn':{},"
                            "'adjustmentsOut':{},"
                            "'allocatedIn':{},"
                            "'allocatedOut':{'USD/1M':582053135},"
                            "'budgetDecreases':{},"
                            "'budgetIncreases':{'USD/1M':614502770},"
                            "'commitmentsMade':{},"
                            "'commitmentsRetired':{},"
                            "'lineItems':{},"
                            "'md':{'objectType':'Account','version':1},"
                            "'recycledIn':{},"
                            "'recycledOut':{},"
                            "'spent':{},"
                            "'type':'budget'}");
    accounts.restoreAccount({"top", "sub"}, jsonValue);
    jsonValue = Json::parse("{'adjustmentLineItems':{},"
                            "'adjustmentsIn':{},"
                            "'adjustmentsOut':{},"
                            "'allocatedIn':{},"
                            "'allocatedOut':{},"
                            "'budgetDecreases':{},"
                            "'budgetIncreases':{'USD/1M':582053135},"
                            "'commitmentsMade':{},"
                            "'commitmentsRetired':{},"
                            "'lineItems':{},"
                            "'md':{'objectType':'Account','version':1},"
                            "'recycledIn':{},"
                            "'recycledOut':{},"
                            "'spent':{'USD/1M':582053135},"
                            "'type':'spend'}");
    accounts.restoreAccount({"top", "sub", "spent"}, jsonValue);

    vector<string> aNames = { "top", "top:sub", "top:sub:spent" };
    for (const string & aName: aNames) {
        AccountSummary accountS
            = accounts.getAccountSummary(aName);
        Json::Value summary = accountS.toJson();
        BOOST_CHECK_EQUAL(summary["md"]["objectType"].asString(),
                          "AccountSummary");
        BOOST_CHECK_EQUAL(summary["md"]["version"].asInt(), 1);

        Json::Value simpleSummary = accountS.toJson(true);
        BOOST_CHECK_EQUAL(simpleSummary["md"]["objectType"].asString(),
                          "AccountSimpleSummary");
        BOOST_CHECK_EQUAL(simpleSummary["md"]["version"].asInt(), 1);
        vector<string> keys = {"budget", "spend", "available", "inFlight"};
        for (const string & key: keys) {
            BOOST_CHECK_EQUAL(summary[key], simpleSummary[key]);
        }
    }
}

/* ensure values of account summaries, both normal and simple, matches the
 * values of the account tree they represent */
BOOST_AUTO_TEST_CASE( test_account_summary_values )
{
    Accounts accounts;

    /* NOTE: the accounts are not particularly consistent with one another */
    Json::Value jsonValue
        = Json::parse("{'adjustmentLineItems':{},"
                      "'adjustmentsIn':{},"
                      "'adjustmentsOut':{},"
                      "'allocatedIn':{},"
                      "'allocatedOut':{'USD/1M':5000},"
                      "'budgetDecreases':{},"
                      "'budgetIncreases':{'USD/1M':10000},"
                      "'commitmentsMade':{},"
                      "'commitmentsRetired':{},"
                      "'lineItems':{},"
                      "'md':{'objectType':'Account','version':1},"
                      "'recycledIn':{},"
                      "'recycledOut':{},"
                      "'spent':{},"
                      "'type':'budget'}");
    accounts.restoreAccount({"top"}, jsonValue);
    jsonValue = Json::parse("{'adjustmentLineItems':{},"
                            "'adjustmentsIn':{},"
                            "'adjustmentsOut':{},"
                            "'allocatedIn':{},"
                            "'allocatedOut':{'USD/1M':2000},"
                            "'budgetDecreases':{},"
                            "'budgetIncreases':{'USD/1M':5000},"
                            "'commitmentsMade':{},"
                            "'commitmentsRetired':{},"
                            "'lineItems':{},"
                            "'md':{'objectType':'Account','version':1},"
                            "'recycledIn':{'USD/1M':1},"
                            "'recycledOut':{},"
                            "'spent':{},"
                            "'type':'budget'}");
    accounts.restoreAccount({"top", "sub"}, jsonValue);
    jsonValue = Json::parse("{'adjustmentLineItems':{},"
                            "'adjustmentsIn':{},"
                            "'adjustmentsOut':{},"
                            "'allocatedIn':{},"
                            "'allocatedOut':{},"
                            "'budgetDecreases':{},"
                            "'budgetIncreases':{'USD/1M':2000},"
                            "'commitmentsMade':{},"
                            "'commitmentsRetired':{},"
                            "'lineItems':{},"
                            "'md':{'objectType':'Account','version':1},"
                            "'recycledIn':{},"
                            "'recycledOut':{},"
                            "'spent':{'USD/1M':1000},"
                            "'type':'spend'}");
    accounts.restoreAccount({"top", "sub", "spent"}, jsonValue);

    AccountSummary summary = accounts.getAccountSummary({"top"}, 0);
    Json::Value normalValue = summary.toJson(false);
    Json::Value expected
        = Json::parse("{'account':"
                      " {'adjustmentLineItems':{},"
                      "  'adjustmentsIn':{},"
                      "  'adjustmentsOut':{},"
                      "  'allocatedIn':{},"
                      "  'allocatedOut': {'USD/1M':5000},"
                      "  'budgetDecreases':{},"
                      "  'budgetIncreases': {'USD/1M':10000},"
                      "  'commitmentsMade':{},"
                      "  'commitmentsRetired':{},"
                      "  'lineItems':{},"
                      "  'md': {'objectType':'Account','version':1},"
                      "  'recycledIn':{},"
                      "  'recycledOut':{},"
                      "  'spent':{},"
                      "  'status':'active',"
                      "  'type':'budget'},"
                      " 'available': {'USD/1M':9001},"
                      " 'adjustedSpent': {'USD/1M' : 1000},"
                      " 'adjustments' : {},"
                      " 'budget': {'USD/1M':10000},"
                      " 'effectiveBudget': {'USD/1M':10001},"
                      " 'inFlight':{},"
                      " 'md': {'objectType':'AccountSummary',"
                      "        'version':1},"
                      " 'spent': {'USD/1M':1000}}");
    BOOST_CHECK_EQUAL(normalValue, expected);

    Json::Value simpleValue = summary.toJson(true);
    expected = Json::parse("{'adjustedSpent': {'USD/1M' : 1000},"
                           " 'adjustments' : {},"
                           " 'available': {'USD/1M':9001},"
                           " 'budget': {'USD/1M':10000},"
                           " 'effectiveBudget': {'USD/1M':10001},"
                           " 'inFlight': {},"
                           " 'md': {'objectType':'AccountSimpleSummary',"
                           "        'version':1},"
                           " 'spent': {'USD/1M':1000}}");
    BOOST_CHECK_EQUAL(simpleValue, expected);
}

