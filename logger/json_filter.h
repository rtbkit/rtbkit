/* json_filter.h                                                   -*- C++ -*-
   Jeremy Barnes, 5 June 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Pre-compression filter for JSON data.
*/

#ifndef __logger__json_filter_h__
#define __logger__json_filter_h__

#include "filter.h"

namespace Datacratic {

/*****************************************************************************/
/* JSON COMPRESSOR                                                           */
/*****************************************************************************/

/** Pre-compressor for JSON data. */

struct JsonCompressor: public Filter {

    JsonCompressor();
    ~JsonCompressor();

    using Filter::process;

    virtual void process(const char * src_begin, const char * src_end,
                         FlushLevel level,
                         boost::function<void ()> onMessageDone);

private:
    struct Itl;
    std::shared_ptr<Itl> itl;
};



/*****************************************************************************/
/* JSON DECOMPRESSOR                                                         */
/*****************************************************************************/

/** Decompressor for JSON data. */

struct JsonDecompressor: public Filter {

    JsonDecompressor();
    ~JsonDecompressor();

    using Filter::process;

    virtual void process(const char * src_begin, const char * src_end,
                         FlushLevel level,
                         boost::function<void ()> onMessageDone);

private:
    struct Itl;
    std::shared_ptr<Itl> itl;
};

} // namespace Datacratic


#endif /* __logger__json_filter_h__ */

