var RTBkit = require('../../build/x86_64/bin/rtb.node'),
    services = require('../../build/x86_64/bin/services.node'),
    budgetController = require('./budget-controller.js'),
    zookeeperUri = "localhost:2181", // must point to same Zookeeper as routers
    services = new services.ServiceProxies(),
    accountAdded = false,
    interval,
    config = require('./agent-config.json'),
    accountParent = config.account[0],
    accountFullName = config.account.join(":");
     
// uri,install name and location from bootstrap.json
services.useZookeeper(zookeeperUri,"rtb-test", "mtl");
 
// yes, we want to log to carbon
services.logToCarbon('localhost:2003');
 
var addAccountHandler = function(err, res){
  if (err) {
    console.log("Error adding account "+accountFullName);
    console.log(err);
  }
}
 
var topupErrorHandler = function(err, res){
  if (err) {
    // TODO: Handle an error topping up the account.
    console.log("Error topping up "+accountFullName);
    // shutdown with an error
    shutdown(1);
  }
}
 
// Keep the budget for this subaccount topped up
var pace = function(){
  if (!accountAdded){
    budgetController.addAccount(accountParent, addAccountHandler);
    accountAdded = true;
  }
  budgetController.topupTransferSync(accountFullName, "USD/1M", 100, topupErrorHandler);
}
 
var agent = new RTBkit.BiddingAgent("node_350_250_1", services);
 
//---------------------
// Agent Event Handlers
//---------------------
// You can skip overriding some of these handlers by setting strictMode(false);
 
agent.onError = function(timestamp, description, message){
  console.log('Bidding Agent sent something invalid to the router.', description, message);
}
 
// the agent won a bid. secondPrice contains the win price
agent.onWin = function(timestamp, confidence, auctionId, spotNum, secondPrice, bidRequest, ourBid, accountInfo, metadata, augmentations, uids){
  console.log("----------------------------------------WON--------------------------");
}
 
// the auction was not won by this agent
agent.onLoss = function(timestamp, confidence, auctionId, spotNum, secondPrice, bidRequest, ourBid, accountInfo, metadata){
  console.log("---------------------------------------LOSS----------------------------");
}
 
// an invalid bid has been sent back to the router
agent.onInvalidBid = function(timestamp, confidence, auctionId, spotNum, secondPrice, bidRequest, ourBid, accountInfo, metadata, augmentations, uids){
  console.log("INVALIDBID");
}
 
// a bid was placed by this bid agent after the router had sent its bids back to the exchange
agent.onTooLate = function(timestamp, confidence, auctionId, spotNum, secondPrice, bidRequest, ourBid, accountInfo, metadata, augmentations, uids){
  console.log("TOOLATE");
}
 
// not sufficient budget available for this agent to bid the price it has chosen
agent.onNoBudget = function(timestamp, confidence, auctionId, spotNum, secondPrice, bidRequest, ourBid, accountInfo, metadata, augmentations, uids){
  console.log("NOBUDGET");
}
 
// the auction dropped this bid. usually happens if the auctionId is unknown
// or if the bid was delayed for too long.
agent.onDroppedBid = function(timestamp, confidence, auctionId, spotNum, secondPrice, bidRequest, ourBid, accountInfo, metadata, augmentations, uids){
  console.log("DROPPED");
}
 
// respond to the router when pinged.
agent.onPing = function(router,timesent,args){
  var timereceived = new Date();
  agent.doPong(router, timesent, timereceived, args);
  timereceived = null;
}
 
agent.onImpression = function(timestamp, auctionId, spotId, spotIndex, bidRequest, bidMeta, winMeta, impressionMeta, clickMeta, augmentations, visits){
  console.log("IMPRESSION");
}
 
agent.onVisit = function(timestamp, auctionId, spotId, spotIndex, bidRequest, bidMeta, winMeta, impressionMeta, clickMeta, augmentations, visits){
  console.log("VISIT");
}
 
agent.onClick = function(timestamp, auctionId, spotId, spotIndex, bidRequest, bidMeta, winMeta, impressionMeta, clickMeta, augmentations, visits){
  console.log("CLICK");
}
 
 
agent.onBidRequest = function(timestamp, auctionId, bidRequest, bids, timeAvailableMs, augmentations, wcm){
   console.log("Inside Bid Request :Agent : nod_300_250_1");
     console.log(wcm);
  var amountArray = [100,200,300,400,500,600,700,800,900,1000];
   var bidAmount = amountArray[Math.floor(Math.random()*amountArray.length)];
  var amount = new RTBkit.MicroUSD(bidAmount);
   console.log("My Bid :Agent : nod_300_250_1 " +bidAmount);
  for (var i=0; i<bids.length; i++){
    // TODO: validate a bid before deciding to put an amount on it
    bids.bid(i,1,amount,1); // spotId, creativeIndex, price, priority
  }
  agent.doBid(auctionId, bids, {}, wcm); // auction id, collection of bids, meta, win cost model.
  amount = null;
};
 
// END Agent Event Handlers
 
 
agent.init();
agent.start();
 
agent.doConfig(config);
// Start pacing the budget inflow for this bid agent
pace();
interval = setInterval(pace,10000);
 
// code is 1 for fail, 0 for success
var shutdown = function(code){
  clearInterval(interval);
  agent.close();
  process.exit(code);
}
