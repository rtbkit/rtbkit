/* rtb_logger_js_test.js
 * Jeremy Barnes, 20 May 2011
 * Copyright (c) 2011 Datacratic.  All rights reserved.
 */

var logger = require('logger');
var sys = require('sys');
var net = require('net');

//console.log(sys.inspect(logger));


var f = new logger.FileOutput("./testdir/dir2/test_file_output.log.gz");

f.logMessage("MESSAGES", ["hello", "this", "is", "a", "message"]);

var pubsubUri = "ipc://test-logger-publish.ipc";

var p = new logger.PublishOutput();
p.bind(pubsubUri);

var l = new logger.Logger();
l.subscribe(pubsubUri);
l.addOutput(f);
l.start();

l.logMessage("MESSAGES2", "this", "is", "also", "a", "message");

p.logMessage("MESSAGE3", ["a", "published", "message"]);

l.logMessage("MESSAGES4", "this", "is", "also", "a", "message", "as", "well");

f.close();

var l2 = new logger.Logger();

var g = new logger.JSOutput();

function logMessage(channel, message)
{
    console.log("got message on channel ", channel, ": ", message);
    l.shutdown();
    l2.shutdown();
}

g.logMessage = logMessage;

console.log(sys.inspect(g));

g.logMessage("channelxxx", "hello world");

l2.addOutput(g);

l2.start();

l2.logMessage("MESSAGES2xxx", "this", "is", "also", "a", "message");



var port = 1301;

var r;

var m = 1;

var tcpserver;

var i;

function handleData(sock, data)
{
    console.log("got socket for m = " + m + " data " + data);
    if (m == 10) {
        console.log("closing");
        sock.end();
        tcpserver.close();
        r.shutdown();
        clearInterval(i);
    }
}

function handleResponse(socket)
{
    console.log("got socket from " + socket.remoteAddress + ":"
                + socket.remotePort);
    socket.on("data", function(data) { handleData(socket, data); });



    //console.log(sys.inspect(socket));
    socket.write("hello");
    //socket.end();
    //tcpserver.close();

    //r.shutdown();
    //delete r;
}

function onListenDone()
{
    console.log("listen done");
    r = new logger.RemoteOutput();
    r.connect(port, "localhost");
    r.logMessage("hello", "how are you today");
}

tcpserver = net.createServer(handleResponse);
tcpserver.listen(port, onListenDone);

function onInterval()
{
    console.log("interval");
    r.logMessage("this is message", m++);
}

i = setInterval(onInterval, 200);



//l.shutdown();



