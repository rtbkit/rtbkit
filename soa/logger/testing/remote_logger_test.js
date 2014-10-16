/* remote_logger_test.js
 *
 */

var logger = require('logger');
var sys = require('sys');
var net = require('net');

var input = new logger.RemoteInput();
input.listen();
var port = input.port();

console.log("port = ", port);

var output = new logger.RemoteOutput();
output.connect(port, "localhost");


for (var i = 0;  i < 1000;  ++i) {
    output.logMessage("channelname", "blah blah this is another message");
}

output.shutdown();

//for (i = 0;  i < 1000;  ++i) {
//    output.logMessage("channelname", "blah blah this is another message");
//}


input.shutdown();