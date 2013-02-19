var zeromq  = require('zeromq');
var sys     = require('sys');

/* The zeromq socket we connect on */
var socket;

/* The date we got the last message from the server */
var lastMessage = new Date();

var inLastSecond = 0;
var lastTick = new Date();
var maxBacklog = 0;

function onTick()
{
    //gc();

    socket.send("HEARTBEAT");

    var secsSinceLastMessage = (new Date() - lastMessage) / 1000;
    if (secsSinceLastMessage > 2) {
        console.log("server down for", secsSinceLastMessage,
                    'seconds');
    }

    var thisTick = new Date();
    var secsSinceLastTick = (thisTick - lastTick) / 1000;
    lastTick = thisTick;

    console.log(inLastSecond, "in", secsSinceLastTick, "seconds =",
                inLastSecond/secsSinceLastTick, "req/second");
    inLastSecond = 0;

    console.log("maxBacklog", maxBacklog);
    maxBacklog = 0;
}

var intervalId = setInterval(onTick, 1000);

function gotError(err)
{
    console.log("got error: ", err);
    console.log(err.stack);
    console.log(err.message);
}

/* Generate the tag for the given placement. */
function generateTag(placement)
{
    var response =
"<script type=\"text/javascript\" language=\"JavaScript\">\
(function() {\
SOURCE_CLICKTRACKER = \"__CLICK_TRACKER_URL__\";\
SOURCE_CLICKTRACKER_EXPECTS_ENCODED = false;\
var randomnum = \"__RANDOM_NUMBER__\";\
var proto = \"http:\";\
if (window.location.protocol == \"https:\") proto = \"https:\";\
if (randomnum.substring(0,2) == \"__\") randomnum = String(Math.random());\
document.writeln('<scr' + 'ipt type=\"text/ja' + 'vascr' + 'ipt\" s' + 'rc=\"' +\
proto + '//www.tagdomain.com' + '/impressions/ext/p=' + '" + placement +
"' + '.js?rand=' + randomnum + '\"></scr' + 'ipt>');\
})();\
</script>";

    return response;
}

/* Send over the client's configuration as a JSON blob.  This will eventually
 * give information about the query load, the tags to serve, any kinds
 * of restrictions on the bids to send over and anything else that is
 * pertinent for the server to know about.
 */
function doConfig()
{
    var config = {

        // How we round-robin between clients.  Only a maximum of one of the
        // clients in a round-robin group will win each impression.
        roundRobin: {
            group: "Clients",   // Group with which it's round robined
            weight: 1           // This client's weight within the group
        },

        // Probability that we bid once it matches
        bidProbability: 0.80,

        // Max number of bids in flight for this client
        maxInFlight: 5,

        // We use this to filter based on host.  The values passed should
        // be regular expressions written as strings.
        hostFilter: {
            include: [ "*" ],
            exclude: [ "icanhazcheezburger.com" ]
        },

        // We use this to indicate what our creatives are; they are used
        // to filter the ads to the ones that we could win.  When we return
        // a creative, it should be an index into this list.
        creatives: [
            {
                name: "bb",
                width: 300,
                height: 250,
                tag: generateTag(8330),
                clickUrl: "http://www.datacratic.com/"
            },
            {
                name: "lb",
                width: 728,
                height: 90,
                tag: generateTag(8331),
                clickUrl: "http://www.datacratic.com/"
            },
        ]
    };

    socket.send("CONFIG", JSON.stringify(config));

    var backlog = socket.currentSendBacklog();
    if (backlog > maxBacklog)
        maxBacklog = backlog;
}

function doAuction(type, id, reqstr, spots)
{
    try {
        var request = JSON.parse(reqstr);
    } catch (e) {
        console.log("error parsing JSON request: ", e);
        return;
    }
    try {
        spots = JSON.parse(spots);
    } catch (e) {
        console.log("error getting spots", e);
        return;
    }
    //console.log("auction: type ", type, " id: ", id, " reqstr: ", reqstr, " spots: ", spots);
    var response = [];

    //console.log("spots", sys.inspect(spots));

    function bidSpot(spot)
    {
        //console.log("spot ", sys.inspect(spot));

        var spotnum = spot.spot;
        var creative = spot.creatives[0];
        var price = 500000;
        var surplus = 50000;


        response[spotnum] = { creative: creative, price: price, surplus: surplus };
    }

    // Bid on each of the spots
    for (var i = 0;  i < spots.length;  ++i) {
        //console.log("spots[i]", spots[i]);
        bidSpot(spots[i]);
    }

    socket.send("BID", "" + id, JSON.stringify(response));

    inLastSecond += 1;
}

function doShutdown()
{
    console.log("shutting down");
    socket.send("SHUTDOWN");
    socket.close();
    clearInterval(intervalId);
}

function doUnknownMessage(type, arg1, arg2, arg3, arg4, arg5, arg6)
{
    var args = ["unknown message: "];
    if (type !== undefined) args.push(type.toString());
    if (arg1 !== undefined) args.push(arg1.toString());
    if (arg2 !== undefined) args.push(arg2.toString());
    if (arg3 !== undefined) args.push(arg3.toString());
    if (arg4 !== undefined) args.push(arg4.toString());
    if (arg5 !== undefined) args.push(arg5.toString());

    console.log.apply(null, args);
}

function doError(arg1, arg2, arg3, arg4, arg5, arg6)
{
    var args = ["ERROR from server: "];
    if (arg1 !== undefined) args.push(arg1.toString());
    if (arg2 !== undefined) args.push(arg2.toString());
    if (arg3 !== undefined) args.push(arg3.toString());
    if (arg4 !== undefined) args.push(arg4.toString());
    if (arg5 !== undefined) args.push(arg5.toString());

    console.log.apply(null, args);
}

// What function we call on each kind of message
var routes = {
    NEEDCONFIG: doConfig,
    AUCTION: doAuction,
    SHUTDOWN: doShutdown,
    ERROR: doError,
    ACKHEARTBEAT: function () {}
};

function gotMessage(type)
{
    var secsSinceLastMessage = (new Date() - lastMessage) / 1000;
    if (secsSinceLastMessage > 2)
        console.log("server back");

    lastMessage = new Date();
    if (type in routes)
        return routes[type].apply(this, arguments);
    else return doUnknownMessage.apply(this, arguments);
}

function run()
{
    var serverUri = 'ipc://tst.ipc';
    socket = zeromq.createSocket('xreq');
    socket["identity"] = "HelloFromJS" + process.pid;
    socket.connect(serverUri);
    socket.on("message", gotMessage);
    socket.on("error", gotError);
    doConfig();
}

run();

