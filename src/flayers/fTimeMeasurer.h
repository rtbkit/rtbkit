#ifndef TIMEMEASURER
#define TIMEMEASURER

#include "fLayersGeneral.h"
#ifdef WIN32
#include <time.h>
#define CLK CLK_TCK
#define FACTOR_CLK 1000
#endif
#ifndef WIN32
#include <sys/times.h>
#define CLK CLOCKS_PER_SEC
#define FACTOR_CLK 10000
#endif
///
///  \class fTimeMeasurer fTimeMeasurer.h "Utils/fTimeMeasurer.h" 
///  \brief fTimeMeasurer class do all related to time measurement.
///
/// -why ? : need experimentation time measurement.
/// author : Francis Pieraut and J-S Senecal
///
class fTimeMeasurer
{
 protected:
  bool started;
    time_t t_before,t_after;
    long cpu_t_before, cpu_t_after;
 public:
   static long getRuntime();
    fTimeMeasurer()
      {t_before=0;t_after=0;started=false;};
	/// Restarts the timer and return current time
    std::string startTimer(bool with_host=false);
	/// Stop the timer and returns current time
    std::string stopTimer(bool cpu_time=false);
    /// Restart the timer, but keep in memory the last running time i.e. difference between time at startTimer (or resumeTimer) and last call
    /// to stopTimer (or getStopRunningTime); returns current time
    std::string getRunningTime(bool only_time=true, bool cpu_time=true);// Returns the running time between start and stop
    std::string getStopRunningTime(bool only_time=true, bool cpu_time=true);// Stops and returns the running time between start and stop (i.e. now)
    double getStopTime();
};

#endif
