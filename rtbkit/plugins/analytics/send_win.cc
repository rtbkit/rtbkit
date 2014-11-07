#include <string>
#include "analytics.h"
#include "jml/arch/timers.h"

using namespace std;
int main() {

    auto an = std::make_shared<AnalyticsClient>();
    an->init("http://127.0.0.1:40000");
    an->start();
    const std::string type("win");
    const std::string event("a Win");
    an->sendEvent(type, event);
    an->sendEvent(type, event);
    an->sendEvent(type, event);
    an->publish(type, string("this"), string("uses"), string("a"), string("variadic"));
    ML::sleep(2);
    an->shutdown();
    return 1;

}
