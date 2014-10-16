/* named_endpoint.h                                                 -*- C++ -*-
   Jeremy Barnes, 14 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#ifndef __service__named_endpoint_h__
#define __service__named_endpoint_h__

#include "service_base.h"
#include "jml/utils/exc_assert.h"
#include <net/if.h>
#include <arpa/inet.h>
#include <set>


namespace Datacratic {


/*****************************************************************************/
/* NAMED ENDPOINT                                                             */
/*****************************************************************************/

/** A endpoint that publishes itself in such a way that other things can
    connect to it by name.

    When we bind the interface, we record where the interface is listening
    in the configuration service.  That way, when something wants to connect
    to the service it can look up the configuration in the object.
*/

struct NamedEndpoint {

    /** The endpoint exists.  Publish that to anyone who may care. */
    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & endpointName);
    
    /** Publish an address for the endpoint. */
    void publishAddress(const std::string & address,
                        const Json::Value & addressConfig);
    
    /** Publish that the address is active.  Atomically causes the endpoint
        to be reconfigured.
    */
    void activate();

    struct Interface {
        int family;
        uint64_t flags;

        /// See man 7 netdevice for these
        uint32_t up:1;            ///< Interface is up
        uint32_t running:1;       ///< Interface is running
        uint32_t loopback:1;      ///< Loopback interface
        uint32_t pointtopoint:1;  ///< Point-to-point link
        uint32_t noarp:1;         ///< no ARP on address

        std::string name;
        std::string addr;
        std::string netmask;
        std::string broadcast;
        std::string hostScope;    ///< Which hosts can connect to this iface
    };

    /** Convert a network address to a string */
    static std::string addrToString(const sockaddr * addr);

    /** Convert a network address to an IP address. */
    static std::string addrToIp(const std::string & addr);

    /** List all of the interfaces on this machine. */
    static std::vector<Interface>
    getInterfaces(const std::set<int> & families = std::set<int>({ AF_INET }),
                  int flagsRequired = IFF_UP,
                  int flagsExcluded = 0);
    
    std::vector<std::string> getPublishedUris() const;

private:
/** Configuration service for this endpoint. */
    std::shared_ptr<ConfigurationService> config;
    std::string endpointName;

    std::vector<std::string> publishedUris;
};



} // namespace Datacratic

#endif /* __service__named_endpoint_h__ */
