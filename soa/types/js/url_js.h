/* url_js.h                                                         -*- C++ -*-
   JS encoding/decoding for URLs.
*/

#pragma once

#include "soa/js/js_utils.h"
#include "soa/types/url.h"

namespace Datacratic {

class Url;

namespace JS {

inline void to_js(JS::JSValue & value, const Url & url)
{
    to_js(value, url.toString());
}

inline Url from_js(const JSValue & value, Url * = 0)
{
    std::string s;
    s = from_js(value, (std::string *)0);
    return Url(s);
}

inline Url from_js_ref(const JSValue & value, Url * = 0)
{
    return from_js(value, (Url *)0);
}

} // namespace JS
} // namespace Datacratic
