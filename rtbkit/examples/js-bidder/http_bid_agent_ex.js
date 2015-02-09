// --- Copyright (c) 2014 Datacratic. All rights reserved.
// ----------------------------------------
//
//  Just a simple http bidder agent based on node.js httpapi
//
//  It does include a minimalist openrtb class that implements basic
//  messaging.
//
//  This implementation uses nodejs http.create server for each
//  port we need to listen, and by passing the appropriate request handler
//  to node httpserver as a CB. So we created a special
//  class to deal with the bid request and passed one of its methods
//  as CB to the httpserver. Since Node.JS already has implemented
//  asynchronous request management we leave that to the http server to
//  deal with.
//
//  There are a budget control class that uses an httprequest object to
//  talk to the the banker, and at start up also register the bidder
//  with the ACS
//  
//  Other few helper functions are defined at the beginning of the
//  module.


// --- imports modules
var http = require("http");
var fs = require("fs");

// --- simple alias function to do JSON objects deep copy
var jsonDeepCopy = function(jsonObj)
{
  var newObj = {};
  if(jsonObj !== undefined)
  {
    newObj = JSON.parse(JSON.stringify( jsonObj ));
  }
  return newObj;
};

// --- JSON marshaller: tries to read the given keys recursivelly and orderly
//     if any key fails return an undefined object
var jsonMarshaller = function(jsonObj, keysArray)
{
  var jsonItem = jsonObj;
  var currKey;
  for (currKey in keysArray)
  {
    jsonItem = jsonItem[keysArray[currKey]];
    if(jsonItem === undefined)
    {
      break;
    }
  };
  return jsonItem;
};


// --- helper function to read config file
var readConfig = function(cfgFile)
{

  var cfgJson;
  // this can be done sync'ly read because this is done only at startup
  // we don't want to have to deal with call backs just for this...
  if (fs.existsSync(cfgFile))
  {
    var contents = fs.readFileSync(cfgFile).toString();
    cfgJson = JSON.parse(contents);
  }

  return cfgJson;
};


// ----------------------------------------
// minimalistic OpenRtb response message object
// ----------------------------------------
var OpenRtbResponse = function ()
{
  // default template response object
  this.keyId = "id";
  this.keyBid = "bid";
  this.keyCrid = "crid";
  this.keyExt = "ext";
  this.keyExtId = "external-id";
  this.keyExtCreatives = "creative-indexes";
  this.keyPriority = "priority";
  this.keyImpId = "impid";
  this.keyPrice = "price";
  this.keySeatBid = "seatbid";
  this.extObj = {};
  this.bidObj = {};
  this.seatBidObj = {};
  this.bidResponseObj = {};
};

// --- returns an object with the scafolding of an rtb response
// and fills some fields based on the request provided
OpenRtbResponse.prototype.getResponse = function(req)
{
  // reset response to default values
  this.extObj[this.keyPriority] = 1.0;
  this.bidObj[this.keyId] = "1";
  this.bidObj[this.keyImpId] = "1";
  this.bidObj[this.keyPrice] = 1.0;
  this.bidObj[this.keyCrid] = "";
  this.bidObj[this.keyExt] = this.extObj;
  this.seatBidObj[this.keyBid] = [this.bidObj];
  this.bidResponseObj[this.keyId]= "1";
  this.bidResponseObj[this.keySeatBid] = [this.seatBidObj];

    // make a deep copy to return
  var resp = jsonDeepCopy(this.bidResponseObj);

  if (this.validateReq(req))
  // if we got a valid 'req' arg, use it to populate some fields
  {
    // set the Id of the response to the same as the request
    resp[this.keyId] = req[this.keyId];

    // empty the bid list (we assume only one seat for simplicity of the example)
    resp[this.keySeatBid][0][this.keyBid] = [];

     // default values for some of the fields of the bid response
    var idCounter = 0;
    var newBid = jsonDeepCopy(this.bidObj);

    // iterate over impressions array from request and
    // populate bid list
    // # NOTE: as an example we are bidding on all the impressions,
    // # usually that is not what one real/smarter bidder would do!!!
    var impNdx;
    for (impNdx in req["imp"])
    {

      var imp = req["imp"][impNdx];
      // -> imp is the field name @ the req

      // dumb bid id, just a counter
      idCounter = idCounter + 1;
      newBid[this.keyId] = idCounter.toString();

      // copy impression id as impId for this bid
      newBid[this.keyImpId] = imp[this.keyId];

      // try to copy external id to the response
      var externalId = jsonMarshaller(imp, [this.keyExt, "external-ids", 0]);
      if(externalId !== undefined)
      {
	newBid[this.keyExt][this.keyExtId] = externalId;
      }

      // will keep the defaul price as it'll be changed by bidder
      // and append this bid into the bid response
      var ref2bidList = resp[this.keySeatBid][0][this.keyBid];
      ref2bidList.push(jsonDeepCopy(newBid));
    }
  }
  
  return resp;
};

// --- validate request
OpenRtbResponse.prototype.validateReq = function(req)
{
  var isValid = false;

  if(req !== undefined)
  // to start with is this a defined variable?
  {
    if(typeof req === 'object')
    // is this a json object
    {
      // TODO:
      // here we should check if the json object is a valid
      // openRtb request, but for the sake of simplicity we
      // will just assume it is valid
      isValid = true;      
    };
  };

  return isValid;
};



// ----------------------------------------
// bidding logic object: Fixed Price
// ----------------------------------------
var FixedPriceBidder = function()
{
  this.bidConfig = {};
  this.openRtb = new OpenRtbResponse();
};

// --- do the bidder config
FixedPriceBidder.prototype.doConfig = function(cfgJson)
{

  if (cfgJson !== undefined)
  {
    this.bidConfig.probability = cfgJson["ACS"]["Config"]["bidProbability"]; 
    this.bidConfig.creatives = cfgJson["ACS"]["Config"]["creatives"]; 
  }
  else
  {
    this.bidConfig.probability = 0.5; 
    this.bidConfig.creatives = [{
      id: 1,
      width: 300,
      height: 250
    }];
  }
  this.bidConfig.price = 1.0;
};

// --- perform a bid based on the request
FixedPriceBidder.prototype.doBid = function(req)
{

  var bidReq;
  if(req.length > 0)
  {
    try {
      bidReq = JSON.parse(req);
    } catch(e) {
      console.log("Malformed JSON!");
    }
  }

  var resp;
  
  if (bidReq !== undefined)
  {
    // assemble default response
    resp = this.openRtb.getResponse(bidReq);

    // ---
    // update bid with creatives and price
    // ---

    // first we need to buid a dictionary
    // that correlates impressions from the request
    // to creative lists
    // FORMAT: dict[extId][impId] = [creat1..creatN]
    var impDict = {};
    var impList = bidReq["imp"];
    var impNdx;
    for (impNdx in impList)
    {
      // current impression
      var imp = impList[impNdx];

      // list of external ids from this impression
      var extIdsList = imp[this.openRtb.keyExt]["external-ids"];
      var extIdsNdx;
      for (extIdsNdx in extIdsList)
      {
	var extId = extIdsList[extIdsNdx];
	var tempDict = {};
	tempDict[imp[this.openRtb.keyId]] =
	  imp[this.openRtb.keyExt][this.openRtb.keyExtCreatives][extId.toString()];
	impDict[extId] = jsonDeepCopy(tempDict);      
      }
    }

    // then we iterate over all bids and choose a random creative for each bid
    // NOTE: we are just doing this fot the first seatbid for simplicity's sake
    var seatBid0 = resp[this.openRtb.keySeatBid][0];
    var bidNdx;
    for (bidNdx in seatBid0[this.openRtb.keyBid])
    {
      // get bid
      var bid = seatBid0[this.openRtb.keyBid][bidNdx];
      // update price
      bid[this.openRtb.keyPrice] = this.bidConfig.price;

      // gets the list of creatives from the ext field in the request 
      var extId = bid[this.openRtb.keyExt][this.openRtb.keyExtId];
      var impId = bid[this.openRtb.keyImpId];
      var creativeList = impDict[extId][impId];

      // gets one of the creative indexes randomly
      var ndx = Math.floor(Math.random() * creativeList.length);
      var creatNdx  = creativeList[ndx];

      // get creative id
      var creativeId = this.bidConfig["creatives"][creatNdx]["id"].toString();

      // set the cretive id to the bid
      bid[this.openRtb.keyCrid] = creativeId;
    };  
  }
  
  // return response
  return resp;
};


// ----------------------------------------
// basic request handler object
// ----------------------------------------
var BaseRequestHandler = function(bidderObj)
{
  this.bidder = bidderObj;
  this.body = "";
  this.response = "";
};

// --- send Error response
BaseRequestHandler.prototype.sendErr = function(resp)
{
  // write HTTP code 204
  resp.writeHead(204);
  // end request
  resp.end();
  // clear the message body
  this.body = "";
  this.response = "";
};

// --- send Ok the response
BaseRequestHandler.prototype.sendOk = function(resp)
{
  // write HTTP header
  resp.writeHead(200, {"Content-Type": "application/json", "x-openrtb-version": "2.1"});
  // write response
  // NOTE: since the answer is not large we are fitting the body in the end() call
  // if write() is used instead, the response time increases dramatically.
  // it seems that writting and then ending makes the TCP conection send 2 chuncks,
  // which in the end causes 5 packets exchange insteaad of just 2 exchanges
  // then on the bidder side the TCP dump shows that the time goes from 1ms
  // to around 40ms for the entire transaction to take place
  // so dont use: "resp.write(this.response + "\n");" !!!
  resp.end(this.response + "\n");
  // clear the message body
  this.body = "";
  this.response = "";
};

// --- process Bid request
BaseRequestHandler.prototype.processBid = function()
{
  var success = true;
  
  if (this.bidder !== undefined)
  {
    var resp = JSON.stringify(this.bidder.doBid(this.body));

    if(resp !== undefined)
    {
      this.response = resp;
    }
    else
    {
      success = false;
    }   
  }
  
  return success;
};

// --- process POST request
BaseRequestHandler.prototype.onPost = function(req, resp)
{
  // process request
  if (this.processBid())
  { // print DEBUG
    // console.log(this.body);
    // console.log(this.response);
    //send response
    this.sendOk(resp);
  }
  else
  {
    this.sendErr(resp);
  }
};

// --- process PUT requests
BaseRequestHandler.prototype.onPut = function(req, resp)
{
  // process request
  if (this.processBid())
  {
    //send response
    this.sendOk(resp);
  }
  else
  {
    this.sendErr(resp);
  }
};

// --- process GET requests
BaseRequestHandler.prototype.onGet = function(req, resp)
{
  // process request
  if (this.processBid())
  {
    //send response
    this.sendOk(resp);
  }
  else
  {
    this.sendErr(resp);
  }
};

// --- dispatcher called by the http server
BaseRequestHandler.prototype.reqDispatcher = function(req, resp)
{
  // which kind of request
  var httpVerb = req.method;
  // work around that deals with the closure of the anonymous functions 
  var self = this; 

  // function call table to reduce repetition of code.
  var callTable = {
    "POST": function(req, resp) {self.onPost(req, resp);},
    "PUT" : function(req, resp) { self.onPut(req, resp);},
    "GET" : function(req, resp) { self.onget(req, resp);}
  };

  // deals with the common types of requests
  if (httpVerb == "POST" || httpVerb == "PUT")
  {
    // recovers the entire message body
    req.on('data', function(chunk) {
      self.body = self.body + (chunk.toString());
    }); 

    // call the specific handler when the body is done
    req.on('end', function() {
      callTable[httpVerb](req, resp);
    });

    // error callback
    req.on('error', function() {
      self.sendErr(resp);
      console.log("received failed request!");
    });
   
  }
  else if (httpVerb == "GET")
  {
    // Get do not have boddy, so just calls the handler
    callTable[httpVerb](req, resp);
  }
  else
  {
    // otherwise just send an error message
    self.sendErr(resp);
  }
};



// ----------------------------------------
// Bank budget control object
// ----------------------------------------
var BudgetControl = function(cfgObj)
{
  this.body = JSON.stringify({ "USD/1M" : cfgObj["Banker"]["Budget"] });
  this.headers = {"Accept": "application/json"};
  this.host = cfgObj["Banker"]["Ip"];
  this.port = cfgObj["Banker"]["Port"];
  this.path = "/v1/accounts/" + cfgObj["ACS"]["Config"]["account"].join(":");
  this.path = this.path + "/balance";
  this.acsIP = cfgObj["ACS"]["Ip"];
  this.acsPort = cfgObj["ACS"]["Port"];
  this.acsCfgJson = cfgObj["ACS"]["Config"];
  this.response = "";
};

// --- called back from http request to banker
BudgetControl.prototype.reqCallback = function(resp)
{
  self = this;
  
  resp.on('data', function(chunk) {
    self.response = self.response + (chunk.toString());
  }); 

  // call the specific handler when the body is done
  resp.on('end', function() {
    console.log("Response to request:");
    console.log(self.response);
    self.response = ""; // reset response
  });

};

// --- make request to the banker 
BudgetControl.prototype.budgetRequest = function()
{
  var self = this; // closure issue work around

  // request options
  var options = {
    host: this.host,
    port: this.port,
    path: this.path,
    headers: jsonDeepCopy(this.headers),
    method: "POST"
  };

  // set the body (POST) length
  options.headers['Content-Length'] = this.body.length;
  
  // request call
  var request = http.request(
    options,
    function (resp) {self.reqCallback(resp);}
  );
  // write our post data to the request
  console.log("budgetRequest: " + this.body);
  request.write(this.body);
  request.end();
};


// --- make request to the Agent Config Server
BudgetControl.prototype.acsRequest = function()
{
  var self = this; // closure issue work around

  // most request options are hard coded for simplicity
  // they must be adjusted to specific environments
  var options = {
    host: this.acsIP,
    port: this.acsPort,
    path: "/v1/agents/my_http_config/config",
    headers: jsonDeepCopy(this.headers),
    method: "POST"
  };

  var body;
  // if there is a config file
  if (this.acsCfgJson !== undefined)
  {
    var body = JSON.stringify(this.acsCfgJson);
    
    // set the body (POST) length
    options.headers['Content-Length'] = body.length;

    // request call
    var request = http.request(
      options,
      function (resp) {self.reqCallback(resp);}
    );
    // write our post data to the request
    console.log("acsRequest: " + body);
    request.write(body);
    request.end();
  }
  
};


// --- start requesting budget and periodically send top up requests to banker
BudgetControl.prototype.start = function(interval)
{
  self = this;

  // ACS call to ensure the account exist is configured
  self.acsRequest();
  
  // send the first budget request
  self.budgetRequest();

  // then repeat it periodically  
  this.timer = setInterval(function() {self.budgetRequest();}, interval);
};

// ----------------------------------------


///////////////////////////////////////////
// set example test variables
///////////////////////////////////////////


// -----
// config file name
var cfgFileName = "../http_config.json";
// read global config object
var cfgObj = readConfig(cfgFileName);
// -----

// ----------------------------------------

// just create a budget control object and start pacing the budget
var budgetController = new BudgetControl(cfgObj);
budgetController.start(cfgObj["Banker"]["Period"]);


// ----------------------------------------


// these are just dummy listeners to receive ad server events
// no action is taken on the events received

var adWinHandler = new BaseRequestHandler();
// --- create server and start listening at the win port
var adWinHttpServer = http.createServer(
  // this binds the callback to the handler object method not its prototype
  function(req, resp) {
    adWinHandler.reqDispatcher(req, resp);
  }
);
adWinHttpServer.listen(cfgObj["Bidder"]["Win"]);

// ---

var adEvtHandler = new BaseRequestHandler();
// --- create server and start listening at the event port
var adEvtHttpServer = http.createServer(
  // this binds the callback to the handler object method not its prototype
  function(req, resp) {
    adEvtHandler.reqDispatcher(req, resp);
  }
);
adEvtHttpServer.listen(cfgObj["Bidder"]["Event"]);


// ----------------------------------------


// --- instantiate a fixed price  bidder
var fixPrice = new FixedPriceBidder();
fixPrice.doConfig(cfgObj); // configuration from json file

// --- http bid request handler object
var bidHandler = new BaseRequestHandler(fixPrice);

// --- create server and start listening at the bid port
var bidHttpServer = http.createServer(
  // this binds the callback to the handler object method not its prototype
  function(req, resp) {
    bidHandler.reqDispatcher(req, resp);
  }
);
bidHttpServer.listen(cfgObj["Bidder"]["Port"]);


// ----------------------------------------
