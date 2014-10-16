/* rotating_output.h                                               -*- C++ -*-
   Jeremy Barnes, 19 September 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Output that writes to an output source and rotates it.
*/

#ifndef __logger__rotating_output_h__
#define __logger__rotating_output_h__

#include "logger.h"
#include "soa/types/date.h"
#include <boost/thread/recursive_mutex.hpp>
#include "soa/types/periodic_utils.h"
#include "soa/gc/rcu_protected.h"

namespace Datacratic {


/*****************************************************************************/
/* ROTATING OUTPUT                                                           */
/*****************************************************************************/

/** An output sink that rotates what it's writing to periodically. */

struct RotatingOutput : public LogOutput {

    RotatingOutput();

    virtual ~RotatingOutput();
    
    void open(const std::string & periodPattern);

    virtual void performRotation();

    virtual void close();

protected:
    typedef boost::recursive_mutex Lock;
    mutable Lock lock;
    typedef boost::unique_lock<Lock> Guard;

    TimeGranularity granularity;
    int number;

    Date currentPeriodStart;  ///< Start of current logging period
    double interval;   ///< How many seconds from period start until we rotate

    /** Subclass must override this to open a sink for the given
        date and make it so that any record() calls write to that new
        sink.
    */
    virtual void openSubordinate(Date newDate) = 0;
    
    /** Subclass must override this to atomically open a sink for the given
        date and replace the current sink so that any record() calls write
        to the new sink.  When the function returns the old sink must not
        be written to anymore.

        Note that due to the atomic requirement, this is NOT the same as
        just calling close on the old one and open on the new one.
    */
    virtual void rotateSubordinate(Date newDate) = 0;

    /** Subclass must override to close the current sink. */
    virtual void closeSubordinate() = 0;
    
private:
    /// Thread to do the rotating
    boost::scoped_ptr<boost::thread> rotateThread;

    /// Flag to indicate that we need to shutdown
    volatile int shutdown_;

    /// Flag to indicate that we're up
    volatile int up_;

    /// Thread to run the logging
    void runRotateThread();
};


/*****************************************************************************/
/* ROTATING OUTPUT ADAPTOR                                                   */
/*****************************************************************************/

/** A rotating output that writes its output to a given logger. */

struct RotatingOutputAdaptor : public RotatingOutput {
    typedef std::function<LogOutput * (std::string)>
        LoggerFactory;

    RotatingOutputAdaptor(LoggerFactory factory);

    virtual ~RotatingOutputAdaptor();

    
    /** Open the file for rotation. */
    void open(const std::string & filenamePattern,
              const std::string & periodPattern);
    
    virtual void logMessage(const std::string & channel,
                            const std::string & message);
    
    virtual Json::Value stats() const;

    virtual void clearStats();

    virtual void close();

    boost::function<void (std::string)> onPreFileOpen;
    boost::function<void (std::string)> onPostFileOpen;
    boost::function<void (std::string)> onPreFileClose;
    boost::function<void (std::string)> onPostFileClose;
    boost::function<void (std::string, std::size_t)> onFileWrite;

    /** Function to be called before and after a log rotation occurs. */
    boost::function<void (std::string, std::string)> onBeforeLogRotation;
    boost::function<void (std::string, std::string)> onAfterLogRotation;

private:
    /** Factory function */
    LoggerFactory loggerFactory;

    std::string filenamePattern;

    GcLock gcLock;

    /** Object that does the actual logging. */
    RcuProtected<LogOutput> logger;

    std::string currentFilename;

    virtual void openSubordinate(Date newDate);

    virtual void rotateSubordinate(Date newDate);

    virtual void closeSubordinate();
};


} // namespace Datacratic

#endif /* __logger__rotating_output_h__ */
