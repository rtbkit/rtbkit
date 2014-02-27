#include <vector>
#include <string>


namespace Datacratic {

struct RunnerTestHelperCommands : std::vector<std::string>
{
    RunnerTestHelperCommands()
        : std::vector<std::string>(),
          active_(0)
    {}

    void reset() { active_ = 0; }

    std::string nextCommand()
    {
        if (active_ < size()) {
            int active = active_;
            active_++;
            return at(active);
        }
        else {
            return "";
        }
    }

    void sendOutput(bool isStdOut, const std::string & data)
    {
        char cmdBuffer[16384];
        int len = data.size();
        int totalLen = len + 3 + sizeof(int);
        if (totalLen > 16384) {
            throw ML::Exception("message too large");
        }
        ::sprintf(cmdBuffer, (isStdOut ? "out" : "err"));
        ::memcpy(cmdBuffer + 3, &len, sizeof(int));
        ::memcpy(cmdBuffer + 3 + sizeof(int), data.c_str(), len);
        push_back(std::string(cmdBuffer, totalLen));
    }

    void sendSleep(int tenthSecs)
    {
        char cmdBuffer[8];
        ::sprintf(cmdBuffer, "slp%.4x", tenthSecs);
        push_back(std::string(cmdBuffer, 7));
    }

    void sendExit(int code)
    {
        char cmdBuffer[1024];
        int totalLen = 3 + sizeof(int);
        ::sprintf(cmdBuffer, "xit");
        ::memcpy(cmdBuffer + 3, &code, sizeof(int));
        push_back(std::string(cmdBuffer, totalLen));
    };

    void sendAbort()
    {
        push_back("abt");
    }

    int active_;
};

} // namespace Datacratic
