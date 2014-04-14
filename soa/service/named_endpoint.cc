/* named_endpoint.cc
   Jeremy Barnes, 24 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Endpoint named so that you can connect to it by name (via a discovery
   service).
*/

#include "named_endpoint.h"
#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "jml/utils/guard.h"
#include "jml/arch/info.h"
#include <sys/utsname.h>

using namespace std;

namespace Datacratic {


/*****************************************************************************/
/* NAMED ENDPOINT                                                             */
/*****************************************************************************/

void
NamedEndpoint::
init(std::shared_ptr<ConfigurationService> config,
          const std::string & endpointName)
{
    this->config = config;
    this->endpointName = endpointName;
}

void
NamedEndpoint::
publishAddress(const std::string & address,
               const Json::Value & addressConfig)
{
    ExcAssert(config);

    //cerr << "publishing " << address << " with " << addressConfig << endl;
    config->setUnique(endpointName + "/" + address,
                      addressConfig);

    for (auto & address: addressConfig) {
        if (address.isMember("transports")) {
            for (auto & transport: address["transports"]) {
                if (transport.isMember("uri")) {
                    publishedUris.emplace_back(transport["uri"].asString());
                }
            }
        }
    }
}
    
vector<string>
NamedEndpoint::
getPublishedUris()
    const
{
    return publishedUris;
}

std::string
NamedEndpoint::
addrToString(const sockaddr * addr)
{
    switch (addr->sa_family) {
    case AF_INET:
    case AF_INET6: {
        char host[NI_MAXHOST];
        int s = getnameinfo(addr,
                            (addr->sa_family == AF_INET)
                            ? sizeof(struct sockaddr_in)
                            : sizeof(struct sockaddr_in6),
                            host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
        if (s != 0)
            throw ML::Exception("getnameinfo: %s",
                                gai_strerror(s));
        return host;
    }
    default:
        return "";
    }
}
    
std::string
NamedEndpoint::
addrToIp(const std::string & addr)
{
    addrinfo * info = 0;
    addrinfo hints = { 0, AF_INET, SOCK_STREAM, 0, 0, 0, 0, 0 };
    int res = getaddrinfo(addr.c_str(), 0, &hints, &info);
    if (res != 0)
        throw ML::Exception("addrToIp(%s): %s", addr.c_str(),
                            gai_strerror(res));
    if (info == 0)
        throw ML::Exception("no addresses");
    ML::Call_Guard guard([&] () { freeaddrinfo(info); });
        
    // Now we have it as a sockaddr_t.  Convert it back to a numeric
    // address
    return addrToString(info->ai_addr);
}

std::vector<NamedEndpoint::Interface>
NamedEndpoint::
getInterfaces(const std::set<int> & families,
              int flagsRequired,
              int flagsExcluded)
{
    using namespace std;

    ifaddrs * addrs = 0;

    int res = getifaddrs(&addrs);
    if (res != 0)
        throw ML::Exception(errno, "getifaddrs failed");
    ML::Call_Guard guard([&] () { freeifaddrs(addrs); });

    vector<Interface> result;

    for (ifaddrs * p = addrs;  p;  p = p->ifa_next) {

        if (!p->ifa_addr)
            continue;
        if (!families.count(p->ifa_addr->sa_family))
            continue;
        if ((p->ifa_flags & flagsRequired) != flagsRequired)
            continue;
        if ((p->ifa_flags & flagsExcluded) != 0)
            continue;

        Interface iface;
        iface.family = p->ifa_addr->sa_family;
        iface.name = p->ifa_name;
        iface.addr = addrToString(p->ifa_addr);
        iface.netmask = addrToString(p->ifa_netmask);
        iface.broadcast = addrToString(p->ifa_broadaddr);
        iface.flags = p->ifa_flags;

        iface.up = p->ifa_flags & IFF_UP;
        iface.running = p->ifa_flags & IFF_RUNNING;
        iface.loopback = p->ifa_flags & IFF_LOOPBACK;
        iface.pointtopoint = p->ifa_flags & IFF_POINTOPOINT;
        iface.noarp = p->ifa_flags & IFF_NOARP;

        // TODO: better way of detecting non-routable addresses
        if (iface.loopback || iface.addr == "127.0.0.1") {
            iface.hostScope = ML::fqdn_hostname("");
        }
        else {
            iface.hostScope = "*";
            // Other scopes...
        }

#if 0
        if (iface.loopback)
            iface.type = "loopback";
        else if (iface.pointtopoint)
            iface.type = "p2p";
#endif
        result.push_back(iface);
    }

    return result;
}


} // namespace Datacratic
