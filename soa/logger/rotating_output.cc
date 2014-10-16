/* rotating_output.cc
   Jeremy Barnes, 19 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#include "rotating_output.h"
#include "jml/arch/futex.h"

using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* ROTATING OUTPUT                                                           */
/*****************************************************************************/

RotatingOutput::
RotatingOutput()
{
}

RotatingOutput::
~RotatingOutput()
{
}
    
void
RotatingOutput::
open(const std::string & periodPattern)
{
    close();

    std::tie(granularity, number)
        = parsePeriod(periodPattern);

    std::tie(currentPeriodStart, interval)
        = findPeriod(Date::now(), granularity, number);

    shutdown_ = false;
    up_ = false;

    // NOTE: we can pass by reference since the log thread never touches
    // sem until this function has exited
    rotateThread
        .reset(new boost::thread([&](){ this->runRotateThread(); }));

    // Wait until we're ready
    while (!up_)
        futex_wait(up_, false);
}
    
void
RotatingOutput::
close()
{
    //cerr << "rfo close" << endl;
    if (rotateThread) {
        shutdown_ = true;
        futex_wake(shutdown_);
        rotateThread->join();
        //cerr << "rfo joined" << endl;
        rotateThread.reset();
        shutdown_ = false;
    }
    else closeSubordinate();
}

void
RotatingOutput::
performRotation()
{
    Guard guard(lock);
    
    currentPeriodStart.addSeconds(interval);

    rotateSubordinate(currentPeriodStart);
}

void
RotatingOutput::
runRotateThread()
{
    openSubordinate(currentPeriodStart);

    up_ = true;
    futex_wake(up_);

    while (!shutdown_) {
        Date now = Date::now();
        Date nextRotation = currentPeriodStart.plusSeconds(interval);
        double secondsUntilRotation = now.secondsUntil(nextRotation);

        if (secondsUntilRotation <= 0) {
            performRotation();
        }
        else {
            futex_wait(shutdown_, false, secondsUntilRotation);
        }
    }
    
    cerr << "rfo shutdown acknowledged" << endl;

    closeSubordinate();

    cerr << "file closed" << endl;
}


/*****************************************************************************/
/* ROTATING OUTPUT ADAPTOR                                                   */
/*****************************************************************************/

RotatingOutputAdaptor::
RotatingOutputAdaptor(LoggerFactory factory)
    : loggerFactory(factory),
      logger(0, gcLock)
{
}

RotatingOutputAdaptor::
~RotatingOutputAdaptor()
{
    close();
}
    
void
RotatingOutputAdaptor::
open(const std::string & filenamePattern,
     const std::string & periodPattern)
{
    close();

    this->filenamePattern = filenamePattern;
    RotatingOutput::open(periodPattern);
}

void
RotatingOutputAdaptor::
close()
{
    RotatingOutput::close();
}
    
void
RotatingOutputAdaptor::
logMessage(const std::string & channel,
           const std::string & message)
{
    if (!logger)
        throw ML::Exception("logging message with no logger");
    logger()->logMessage(channel, message);
}
    
Json::Value
RotatingOutputAdaptor::
stats() const
{
    Guard guard(lock);
    return logger()->stats();
}

void
RotatingOutputAdaptor::
clearStats()
{
    Guard guard(lock);
    logger()->clearStats();
}

void
RotatingOutputAdaptor::
rotateSubordinate(Date currentPeriodStart)
{
    string oldFilename = currentFilename;
    string newFilename = filenameFor(currentPeriodStart, filenamePattern);

    if (onBeforeLogRotation)
        onBeforeLogRotation(oldFilename, newFilename);

    // We don't close, as calling openSubordinate() will automatically close
    // the old one once it's not in use anymore
    openSubordinate(currentPeriodStart);
    
    if (onAfterLogRotation)
        onAfterLogRotation(oldFilename, newFilename);
}

void
RotatingOutputAdaptor::
closeSubordinate()
{
    if (logger)
        logger()->close();
    logger.replace(0);
}

void
RotatingOutputAdaptor::
openSubordinate(Date currentPeriodStart)
{
    currentFilename = filenameFor(currentPeriodStart, filenamePattern);
    logger.replace(loggerFactory(currentFilename));
}


} // namespace Datacratic
