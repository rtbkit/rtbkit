/* filter_js_test.js
 * Jeremy Barnes, 30 May 2011
 * Copyright (c) 2011 Datacratic.  All rights reserved.
 *
 * Test program for filter.
 */

require('sync').makeSynchronous();

var logger = require('logger');
var sys = require('sys');
var net = require('net');
var assert = require('assert');
var fs = require('fs');
var child_process = require('child_process');

console.log(sys.inspect(logger));


var id = new logger.IdentityFilter();
console.log(id);
console.log(id.onOutput);
console.log(id.onError);


var id2 = logger.Filter.create("identity");
console.log(id2);


var id_output = [];
id.onOutput = function (str, flush, cb) { id_output.push(str);  if (cb) cb(); };
console.log(id.onOutput);
console.log(id);

id.process("hello", logger.Filter.FLUSH_NONE);

assert.deepEqual(id_output, ["hello"]);



var z = logger.Filter.create("z", "COMPRESS");
console.log(z.toString());

var dez = logger.Filter.create("z", "DECOMPRESS");
console.log(dez.toString());

//process.exit();

z.onOutput = function(str, flush, cb) { if (str != "") { console.log("passing through " + str.length + " bytes");  dez.process(str, flush, cb);} else if (cb) cb(); };

var zdez_output = [];
dez.onOutput = function(str, flush, cb) { console.log("got output ", str);  zdez_output.push(str);  if (cb) cb(); };

z.process("hello", logger.Filter.FLUSH_NONE);
z.process(" world", logger.Filter.FLUSH_SYNC);

assert.deepEqual(zdez_output, [ "hello world" ]);






var xz = logger.Filter.create("xz", "COMPRESS");
console.log(xz.toString());

var dexz = logger.Filter.create("xz", "DECOMPRESS");
console.log(dexz.toString());

xz.onOutput = function(str, flush, cb) { if (str != "") { console.log("passing through " + str.length + " bytes");  dexz.process(str, flush, cb);} else if (cb) cb(); };

var xzdez_output = [];
dexz.onOutput = function(str, flush, cb) { if (str != "") { console.log("got output ", str);  xzdez_output.push(str); } if (cb) cb(); };

xz.process("hello", logger.Filter.FLUSH_NONE);
xz.process(" world", logger.Filter.FLUSH_SYNC);

assert.deepEqual(xzdez_output, [ "hello world" ]);

var auctions = [
    '{"id":"93796339-bca4-4001-15e8-deff6c06441e","timestamp":1307028230.469642,"is_test":false,"inventory_source":"DBLCLK_ADX","user":{"ip_address":"74.65.115.54","user_agent":"Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0),gzip(gfe),gzip(gfe)","country_code":"US","region_code":"NY","city_name":"Cooperstown","postal_code":"607","dma":526},"inventory":{"url":"http://www.quiltedparadise.com/n/2011-6-2.jsp","attributes":[{"source":"DBLCLK_ADX","type":"google_uid","value":"CAESEHVC3_42JpS4uOCcq8FaktY"},{"source":"DBLCLK_ADX","type":"1230","value":"0.6806696057319641"},{"source":"DBLCLK_ADX","type":"284","value":"0.16486994922161102"},{"source":"DBLCLK_ADX","type":"11","value":"0.15446043014526367"}]},"restrictions":[{"source":"DBLCLK_ADX","type":"category_id","value":"10"},{"source":"DBLCLK_ADX","type":"category_id","value":"18"},{"source":"DBLCLK_ADX","type":"category_id","value":"3"},{"source":"DBLCLK_ADX","type":"category_id","value":"8"},{"source":"DBLCLK_ADX","type":"category_id","value":"4"},{"source":"DBLCLK_ADX","type":"category_id","value":"5"},{"source":"DBLCLK_ADX","type":"category_id","value":"24"},{"source":"DBLCLK_ADX","type":"category_id","value":"23"},{"source":"DBLCLK_ADX","type":"category_id","value":"31"},{"source":"DBLCLK_ADX","type":"category_id","value":"19"},{"source":"DBLCLK_ADX","type":"click_url","value":"imvu.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"nextag.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"plentyoffish.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"zoosk.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"ancestry.co.uk"},{"source":"DBLCLK_ADX","type":"click_url","value":"fatburningfurnace.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"justanswer.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"aol.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"phoenix.edu"},{"source":"DBLCLK_ADX","type":"click_url","value":"gay.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"made-in-china.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"blurtit.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"kayak.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"fullsail.edu"},{"source":"DBLCLK_ADX","type":"click_url","value":"autotrader.com"}],"spots":[{"id":"1","width":160,"height":600,"reserve_price":0}]}',
    '{"id":"b47d6339-bca4-4001-15e8-deff6c064421","timestamp":1307028230.470767,"is_test":false,"inventory_source":"DBLCLK_ADX","user":{"ip_address":"173.174.59.11","user_agent":"Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Trident/4.0; GTB7.0; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.30729; .NET CLR 3.5.30729; .NET4.0C),gzip(gfe)","country_code":"US","region_code":"TX","city_name":"Austin","postal_code":"512","dma":635},"inventory":{"attributes":[{"source":"DBLCLK_ADX","type":"google_uid","value":"CAESEBVZLNi5-s6yI-yr9ehL9Js"},{"source":"DBLCLK_ADX","type":"507","value":"0.6086142659187317"},{"source":"DBLCLK_ADX","type":"16","value":"0.20674879848957062"},{"source":"DBLCLK_ADX","type":"409","value":"0.1846369355916977"}]},"restrictions":[{"source":"DBLCLK_ADX","type":"category_id","value":"4"},{"source":"DBLCLK_ADX","type":"category_id","value":"5"},{"source":"DBLCLK_ADX","type":"category_id","value":"10"},{"source":"DBLCLK_ADX","type":"category_id","value":"24"},{"source":"DBLCLK_ADX","type":"category_id","value":"31"},{"source":"DBLCLK_ADX","type":"click_url","value":"homestead.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"imvu.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"yahoo.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"starware.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"scientology.org"},{"source":"DBLCLK_ADX","type":"click_url","value":"tinyurl.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"smarter.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"muslima.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"plentyoffish.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"mate1.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"match.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"russianeuro.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"true.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"smileycentral.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"youtube.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"fitness-singles.com"}],"spots":[{"id":"1","width":300,"height":250,"reserve_price":0}]}',
    '{"id":"95756339-bca4-4001-15e8-deff6c06441d","timestamp":1307028230.4687,"is_test":false,"inventory_source":"DBLCLK_ADX","user":{"ip_address":"76.242.183.94","user_agent":"Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; BRI/2; .NET CLR 2.0.50727),gzip(gfe),gzip(gfe)","country_code":"US","region_code":"OK","city_name":"Oklahoma City","postal_code":"405","dma":650},"inventory":{"url":"http://www.playlist.com","attributes":[{"source":"DBLCLK_ADX","type":"google_uid","value":"CAESEA-EIOzG4BbTNLrd9LsQlxY"},{"source":"DBLCLK_ADX","type":"529","value":"0.4397984445095062"},{"source":"DBLCLK_ADX","type":"220","value":"0.40002569556236267"},{"source":"DBLCLK_ADX","type":"592","value":"0.1601758450269699"}]},"restrictions":[{"source":"DBLCLK_ADX","type":"click_url","value":"napster.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"rhapsody.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"buzznet.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"tunecore.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"mog.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"nevadaskideals.com"}],"spots":[{"id":"1","width":728,"height":90,"reserve_price":0}]}',
    '{"id":"2f766339-bca4-4001-15e8-deff6c06441c","timestamp":1307028230.468866,"is_test":false,"inventory_source":"DBLCLK_ADX","user":{"ip_address":"173.166.155.129","user_agent":"Mozilla/5.0 (Windows NT 5.1) AppleWebKit/534.24 (KHTML, like Gecko) Chrome/11.0.696.71 Safari/534.24,gzip(gfe),gzip(gfe)","country_code":"US","region_code":"DC","city_name":"Washington","postal_code":"202","dma":511},"inventory":{"url":"http://www.playlist.com","attributes":[{"source":"DBLCLK_ADX","type":"google_uid","value":"CAESEN06VGy7E-3wmJYEqP-9J4Q"},{"source":"DBLCLK_ADX","type":"529","value":"0.4397984445095062"},{"source":"DBLCLK_ADX","type":"220","value":"0.40002569556236267"},{"source":"DBLCLK_ADX","type":"592","value":"0.1601758450269699"}]},"restrictions":[{"source":"DBLCLK_ADX","type":"click_url","value":"napster.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"rhapsody.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"buzznet.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"tunecore.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"mog.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"nevadaskideals.com"}],"spots":[{"id":"1","width":728,"height":90,"reserve_price":0}]}',
    '{"id":"1e776339-bca4-4001-15e8-deff6c06441f","timestamp":1307028230.469107,"is_test":false,"inventory_source":"DBLCLK_ADX","user":{"ip_address":"76.24.18.210","user_agent":"Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0.1) Gecko/20100101 Firefox/4.0.1,gzip(gfe),gzip(gfe)","country_code":"US","region_code":"MA","city_name":"Cambridge","postal_code":"617","dma":506},"inventory":{"url":"http://www.thesuperficial.com/tag/blake-lively","attributes":[{"source":"DBLCLK_ADX","type":"google_uid","value":"CAESEIgdWnd2wDHYKZHw7Az23F8"},{"source":"DBLCLK_ADX","type":"184","value":"0.8093111515045166"},{"source":"DBLCLK_ADX","type":"1222","value":"0.190688818693161"}]},"restrictions":[{"source":"DBLCLK_ADX","type":"category_id","value":"3"},{"source":"DBLCLK_ADX","type":"category_id","value":"5"},{"source":"DBLCLK_ADX","type":"click_url","value":"ask.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"yahoo.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"google.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"msn.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"aol.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"bing.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"att.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"nagr.org"},{"source":"DBLCLK_ADX","type":"click_url","value":"abc.go.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"jillianmichaels.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"gilt.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"htc.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"buysub.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"vgamenetwork.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"usmagazine.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"pbteen.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"middlemenmovie.com"},{"source":"DBLCLK_ADX","type":"click_url","value":"couponnetwork.com"}],"spots":[{"id":"1","width":160,"height":600,"reserve_price":0}]}'
];

function testCompressDecompress(compressor, decompressor, input, flush)
{
    var compressedBytes = 0;

    function onCompressedData(str, flush, cb)
    {
        compressedBytes += str.length;
        if (str != "") {
            console.log("passing through " + str.length + " bytes");
            decompressor.process(str, flush, cb);
        }
        else if (cb) cb();
    }

    compressor.onOutput = onCompressedData;

    var decompressedBytes = 0;
    var decompressor_output = [];

    function onDecompressedData(str, flush, cb)
    {
        decompressedBytes += str.length;
        if (str != "") {
            console.log("got output ", str);
            decompressor_output.push(str);
        }
        if (cb) cb();
    }

    decompressor.onOutput = onDecompressedData;

    for (line in input) {
        compressor.process(input[line], flush);
    }

    compressor.process("", logger.Filter.FLUSH_FINISH);

    console.log("decompressed " + decompressedBytes + " compressed "
                + compressedBytes + " ratio " + (compressedBytes / decompressedBytes));

    //assert.equal(input.length, output.length);

    //assert.deepEqual(decompressor_output, input);
}

testCompressDecompress(logger.Filter.create("z", "COMPRESS"),
                       logger.Filter.create("z", "DECOMPRESS"),
                       auctions,
                       logger.Filter.FLUSH_SYNC);

testCompressDecompress(logger.Filter.create("xz", "COMPRESS"),
                       logger.Filter.create("xz", "DECOMPRESS"),
                       auctions,
                       logger.Filter.FLUSH_SYNC);


function testToolCompatibility(compressor, decompressor, input, flush,
                               tool, extension)
{
    extension = extension || tool;

    var compressedFname = tool + "-compressed.txt." + extension;
    var uncompressedFname = tool + "-uncompressed.txt";

    var raw = fs.openSync(uncompressedFname, "w");
    var cmp = fs.openSync(compressedFname, "w");

    process.on("exit", function () { fs.unlinkSync(compressedFname); });
    process.on("exit", function () { fs.unlinkSync(uncompressedFname); });

    var compressedBytes = 0;

    function onCompressedData(str, flush, cb)
    {
        compressedBytes += str.length;
        fs.writeSync(cmp, new Buffer(str, 'ascii'), 0, str.length);
        if (str != "") {
            decompressor.process(str, flush, cb);
        }
        else if (cb) cb();
    }

    compressor.onOutput = onCompressedData;

    var decompressedBytes = 0;
    var decompressor_output = [];

    function onDecompressedData(str, flush, cb)
    {
        decompressedBytes += str.length;
        //console.log('str = ', str);
        fs.writeSync(raw, new Buffer(str, 'ascii'), 0, str.length);
        if (cb) cb();
    }

    decompressor.onOutput = onDecompressedData;

    var inputLength = 0;

    for (line in input) {
        compressor.process(input[line] + '\n', flush);
        inputLength += input[line].length + 1;
    }

    compressor.process("", logger.Filter.FLUSH_FINISH);

    console.log("decompressed " + decompressedBytes + " compressed "
                + compressedBytes + " ratio " + (compressedBytes / decompressedBytes));

    var stats = fs.fstatSync(raw);

    //console.log(sys.inspect(stats));

    fs.closeSync(raw);
    fs.closeSync(cmp);

    assert.equal(stats.size, inputLength);

    //assert.equal(input.length, output.length);
    //assert.deepEqual(decompressor_output, input);

    function onFinishedChild(error, stdout, stderr)
    {
        console.log("finished child: error = ", error);
        console.log("stdout: ", stdout);
        console.log("stderr: ", stderr);

        assert.strictEqual(error, null);
        assert.equal(stdout, "");

    }

    child_process.exec("cat " + compressedFname + " | "
                       + tool + " -d | diff - "
                        + uncompressedFname, null, onFinishedChild);

}

testToolCompatibility(logger.Filter.create("xz", "COMPRESS"),
                      logger.Filter.create("xz", "DECOMPRESS"),
                      auctions,
                      logger.Filter.FLUSH_SYNC,
                      "xz");

//testToolCompatibility(logger.Filter.create("bzip2", "COMPRESS"),
//                      logger.Filter.create("bzip2", "DECOMPRESS"),
//                      auctions,
//                      logger.Filter.FLUSH_SYNC,
//                      "bzip2");

//testToolCompatibility(logger.Filter.create("gz", "COMPRESS"),
//                      logger.Filter.create("gz", "DECOMPRESS"),
//                      auctions,
//                      logger.Filter.FLUSH_SYNC,
//                      "gzip");

