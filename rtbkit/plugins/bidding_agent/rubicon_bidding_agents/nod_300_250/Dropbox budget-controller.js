var http = require("http");
var querystring = require("querystring");
 
BudgetController = {};
module.exports = BudgetController;
 
var host = "localhost",
    root_path = "/v1/accounts",
    default_port = 9985;
 
var req_options = {
  host: host,
  path: root_path,
  port:default_port
};
 
// TODO: better error handling for requests
 
// addAccount does a POST to /vi/accounts?accountType=budget&accountName=<accountName>
BudgetController.addAccount = function(account, callback){
  var req = http.request({
    port : default_port,
    method : "POST",
    path : root_path+"?accountName="+account+"&accountType=budget"
  },
  function(res){
    if (res.statusCode == "400"){
      console.log("Add account ERROR 400");
    }
    callback(null, res);
  });
 
  req.on("error", function(e){
    console.log(e.message);
  });
 
  req.end();
}
 
// topupTransferSync does a PUT {currency:amount}
// to /v1/accounts/<account>/balance?accountType=budget
BudgetController.topupTransferSync = function(account, currency, amount, callback){
  console.log('topping up');
  var put_data = '{"'+currency+'":'+amount+'}';
  var req = http.request({
    port : default_port,
    method : "PUT",
    path :  "/v1/accounts/"+account+"/balance?accountType=budget",
    headers : {
      "Content-Type":"application/json",
      "Content-Length":put_data.length
    }
  },
  function(res){
    if (res.statusCode == "400"){
      console.log("Top up ERROR 400");
    }
    callback(null, res);
  });
   
  req.on("error", function(e){
    console.log("ERROR with topupTransferSync in budget-controller.js", e);
    callback(e);
  });
 
  req.write(put_data);
  req.end();
}
