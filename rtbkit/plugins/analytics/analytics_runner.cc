
#include <string>
#include "analytics.h"


int main() {
    const std::string host("http://127.0.0.1");
    AnalyticsClient an(1246, host);
    const std::string win("a Win");
    an.sendWin(win);
    return 1;
}
