/* openssl_threading.h                                             -*- C++ -*-
   Wolfgang Sourdeau, 15 July 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.

   This module registers OpenSSL-specific callbacks required to ensure proper
   functioning of the library when used in multi-threaded programs. See
   manpage "CRYPTO_set_dynlock_create_callback" for further details.
*/

namespace Datacratic {

/* Init the openssl multi-threading environment. Should be called from any
 * module potentially making use of the OpenSSL crypto functions, directly or
 * not. It is safe to call this function multiple times. */
void initOpenSSLThreading();

}
