/* syslog_trace.cc
   Mathieu Stefani, 23 September 2013

   Utility to collect RTBKit traces from syslog
*/

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <array>

#include <syslog.h>
#include <iostream>
#include <string>

#include "soa/jsoncpp/json.h"

static const size_t MaxMessage = 50;

void usage() {
    std::cout << "usage: syslog_aggregator fifo-path" << std::endl;
}

bool handleMessage(const std::string &message) {
    std::cout << "Received message -> " << message << std::endl;

    Json::Value root;
    Json::Reader reader;
    bool ok = reader.parse(message, root);
    if (!ok) {
        return false;
    }

    const auto tag = root["tag"].asString();
    std::cout << "tag = " << tag << std::endl;
    return true;
}

int main(int argc, const char *argv[]) {
    if (argc == 1) {
        usage();
        return 0;
    }

    int fd = open(argv[1], O_RDONLY);

    if (fd == -1) {
        ::perror("open");
        return 1;
    }

    char c;
    ssize_t bytes { 0 };
    std::string message;
    bool inMessage { false };
    for (;;) {
        if ((bytes = read(fd, &c, 1)) == -1) {
            ::perror("read");
            return 1;
        }
        
        if (c == '}') {
            message += c;
            if (!handleMessage(message)) 
                std::cerr << "Failed to handle message: " << message << std::endl;
            message.clear();
            inMessage = false;
        } else if (c == '{') {
            inMessage = true;
        }

        if (inMessage) {
            message += c;
        }

    }

}
