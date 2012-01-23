/* wakeup_fd.h                                                     -*- C++ -*-
   Jeremy Barnes, 23 January 2012
   Copyright (c) 2012 Recoset.  All rights reserved.

   Simple class that provides an FD that we can use to wake something
   up.  A generalization of the self-pipe trick.
*/

#ifndef __jml__arch__wakeup_fd_h__
#define __jml__arch__wakeup_fd_h__

#include <sys/eventfd.h>
#include "exception.h"

namespace ML {

struct Wakeup_Fd {
    Wakeup_Fd()
    {
        fd_ = eventfd(0, 0);
        if (fd_ == -1)
            throw ML::Exception(errno, "eventfd");
    }

    ~Wakeup_Fd()
    {
        close(fd_);
    }

    int fd() const { return fd_; }

    void signal()
    {
        //cerr << "wakeup signal" << endl;
        eventfd_t val = 1;
        int res = eventfd_write(fd_, val);
        if (res == -1)
            throw ML::Exception(errno, "eventfd write()");
    }

    eventfd_t read()
    {
        eventfd_t val = 0;
        int res = eventfd_read(fd_, &val); 
        if (res == -1)
            throw ML::Exception(errno, "eventfd read()");
        return val;
    }

    int fd_;
};



} // namespace Recoset



#endif /* __jml__arch__wakeup_fd_h__ */
