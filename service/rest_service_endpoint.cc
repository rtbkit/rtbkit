/* rest_service_endpoint.cc
   Jeremy Barnes, 11 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Endpoint to talk with a REST service.
*/

#include "rest_service_endpoint.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/pair_utils.h"

using namespace std;

namespace Datacratic {

std::ostream & operator << (std::ostream & stream, const RestRequest & request)
{
    return stream << request.verb << " " << request.resource << endl
                  << request.params << endl
                  << request.payload;
}


} // namespace Datacratic
