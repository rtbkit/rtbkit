/* logger_spam_test.cc
   Remi Attab, 5 December 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Tests to flush out any potential deadlocks with the log output threads.
   Note that fo here is shorthand for FileOutput.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/logger/logger.h"
#include "soa/logger/file_output.h"
#include "jml/arch/atomic_ops.h"

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <tuple>
#include <string>


using namespace Datacratic;
using namespace ML;
using namespace std;
namespace fs = boost::filesystem;



struct TestFolderFixture {

    TestFolderFixture () : 
	testFolder("./logtest-dl")
    {
	if (fs::exists(testFolder)) {
	    fs::remove_all(testFolder);
	}
	BOOST_REQUIRE(fs::create_directory(testFolder));
    }

    ~TestFolderFixture () {
	BOOST_WARN(fs::remove_all(testFolder) > 0);
    }
    
    fs::path testFolder;
};



struct SimpleNextMessage {
    map<int,string> msgCache;

    tuple<string, string> operator () (int index) {
	string message;	
	int msgSize = index % 100;

	if (index == msgSize) {
	    message = "asd";
	    for (int i = 0; i < msgSize; ++i)
		message += "dsa";
	    msgCache[msgSize] = message;
	}
	else
	    message = msgCache[msgSize];

	return make_tuple("TEST", message);
    };

} simpleNextMessage;



BOOST_FIXTURE_TEST_CASE(test_fo_logging, TestFolderFixture) {
    const int messagesToSend = 100000;

    Logger logger;
    boost::barrier barrier(2);

    // Setup the output
    std::shared_ptr<FileOutput> fo (new FileOutput((testFolder / "a").string()));
    fo->onFileWrite = [&](string channel, size_t bytesWritten) {
        if (channel == "KILL") barrier.wait();
    };
    logger.addOutput(fo); // Accepts everything.


    logger.start();

    // Start spamming
    for (int i = 0; i < messagesToSend; ++i) {
	string channel, message;
	tie(channel, message) = simpleNextMessage(i);
	logger.logMessage(channel, message);
    }

    logger.logMessage("KILL", "");
    barrier.wait();

    logger.waitUntilFinished();
    logger.shutdown();
}



BOOST_FIXTURE_TEST_CASE(test_rotatingfo_logging, TestFolderFixture) {
    const int messagesToSend = 100;

    Logger logger;
    boost::barrier barrier(2);

    // Setup the output
    std::shared_ptr<RotatingFileOutput> rfo (new RotatingFileOutput());
    rfo->onFileWrite = [&](string channel, size_t bytesWritten) {
        if (channel == "KILL") barrier.wait();
    };
    rfo->open(testFolder.string() + "/rfo-deadlock-%s.log", "200x"); // rotate every 200ms.
    logger.addOutput(rfo); // Accepts everything.


    logger.start();

    // Start spamming
    for (int i = 0; i < messagesToSend; ++i) {
	string channel, message;
	tie(channel, message) = simpleNextMessage(i);
	logger.logMessage(channel, message);
    }

    logger.logMessage("KILL", "");
    barrier.wait();

    logger.waitUntilFinished();
    logger.shutdown();
}



BOOST_FIXTURE_TEST_CASE(test_multi_rotatingfo_logging, TestFolderFixture) {
    const int messagesToSend = 1000000;
    const int outputThreadCount = 64;

    vector<string> channelList = {
        "CHAN_A", "CHAN_B", "CHAN_C", "CHAN_D"
    };
    vector<string> compressionList = { "", "xz", "gz" };
    vector<int> levelList = { -1, 2, -1 } ;

    // Overkill for our simple rand needs but fun none-the-less
    mt19937 engine;
    uniform_int_distribution<int> regexDist(1, channelList.size()-1);
    uniform_int_distribution<int> compressionDist(0, compressionList.size()-1);

    vector< tuple<string, string, int>> outputConfigList;
    for (int i = 0; i < outputThreadCount; ++i) {

        int randRegex = regexDist(engine);
        string allowRegex = "KILL|CHAN_A|" + channelList[randRegex];

        int randComp = compressionDist(engine);

        auto thConfig = make_tuple(allowRegex, compressionList[randComp], levelList[randComp]);
        outputConfigList.push_back(thConfig);
    }

    Logger logger;
    boost::barrier barrier(outputConfigList.size() + 1);


    // Setup the output
    for (int i = 0; i < outputConfigList.size(); ++i) {
        std::shared_ptr<RotatingFileOutput> rfo (new RotatingFileOutput());
        rfo->onFileWrite = [&](string channel, size_t bytesWritten) {
            if (channel == "KILL") barrier.wait();
        };

        int level;
        string allowRegex, compression;
        tie(allowRegex, compression, level) = outputConfigList[i];

        string path = testFolder.string();
        path += "/" + boost::lexical_cast<string>(i) + "-" + compression;
        path += "/rfo-deadlock-%s.log";

        rfo->open(path, "2x", compression, level); // rotate every 2ms
        logger.addOutput(rfo, boost::regex(allowRegex));
    }
    

    logger.start();

    auto nextMessage = [&](int index) -> tuple<string, string> {
        string message, channel;
        tie(channel, message) = simpleNextMessage(index);

        int messagesPerRound = messagesToSend /2;

        // Round robin message distribution for the first half of the test.
        if (index < messagesPerRound)
            channel = channelList[index % channelList.size()];

        // Spam one channel at the time for the second half of the test.
        else {
            int adjIndex = index - messagesPerRound;
            int messagesForChannel = (messagesPerRound / channelList.size()) + 1;
            channel = channelList[adjIndex / messagesForChannel];
        }

        return make_tuple(channel, message);
    };

    // Start spamming
    for (int i = 0; i < messagesToSend; ++i) {
	string channel, message;
	tie(channel, message) = nextMessage(i);
	logger.logMessage(channel, message);
    }

    // Cleanup
    logger.logMessage("KILL", "");
    barrier.wait();

    logger.waitUntilFinished();
    logger.shutdown();
}

