/* value_description_fwd.h                                         -*- C++ -*-
   Jeremy Barnes, 29 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Code for description and introspection of values and structures.  Used
   to allow for automated formatters and parsers to be built.
*/

#pragma once

namespace Datacratic {

struct JsonParsingContext;
struct JsonPrintingContext;

template<typename T>
struct ValueDescription;

template<typename T, typename Enable = void>
struct DefaultDescription;


} // namespace Datacratic
