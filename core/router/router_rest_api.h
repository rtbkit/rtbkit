/* router_rest_api.h                                               -*- C++ -*-
   Jeremy Barnes, 7 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Class to add an http monitor to a router.
*/

#ifndef __router__router_rest_api_h__
#define __router__router_rest_api_h__

#include "soa/service/http_monitor.h"
#include "soa/service/service_base.h"
#include <string>

namespace RTBKIT {

struct Router;

/******************************************************************************/
/* ROUTER REST API CONNECTION                                                 */
/******************************************************************************/

struct RouterRestApiConnection
    : public HttpMonitorHandler<RouterRestApiConnection, Router*> {

    RouterRestApiConnection(const std::string& name,
                            Router* router);

    Router * router;

    virtual void doGet(const std::string& resource);

    virtual void doPost(const std::string& resource,
                        const std::string& payload);
};

typedef HttpMonitor<RouterRestApiConnection, Router*> RouterRestApi;

} // namespace RTBKIT


#endif /* __router__router_rest_api_h__ */
