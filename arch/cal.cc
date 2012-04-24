/* cuda.cc
   Jeremy Barnes, 10 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   CUDA implementation.  dlopens and initializes CUDA if it's available on the
   system.
*/

#include <cal.h>
#include <iostream>
#include "format.h"
#include "jml/utils/guard.h"
#include <boost/bind.hpp>
#include "jml/utils/environment.h"


using namespace std;


namespace ML {

namespace {

Env_Option<bool> debug("DEBUG_CAL_INIT", false);

} // file scope

std::string printTarget(CALtarget target)
{
    switch (target) {
    case CAL_TARGET_600: return "R600 GPU ISA";
    case CAL_TARGET_610: return "RV610 GPU ISA";
    case CAL_TARGET_630: return "RV630 GPU ISA";
    case CAL_TARGET_670: return "RV670 GPU ISA";
    case CAL_TARGET_7XX: return "R700 class GPU ISA";
    case CAL_TARGET_770: return "RV770 GPU ISA";
    case CAL_TARGET_710: return "RV710 GPU ISA";
    case CAL_TARGET_730: return "RV730 GPU ISA";
    default:
        return format("unknown CALtarget(%d)", target);
    }
}

/** Device Kernel ISA */
typedef enum CALtargetEnum {
} CALtarget;


struct Register_CAL {
    
    Register_CAL()
    {
        if (debug)
            cerr << "registering CAL" << endl;
        
        CALresult r = calInit();
        if (r != CAL_RESULT_OK) {
            if (debug)
                cerr << "calInit() returned " << r << ": "
                     << calGetErrorString()
                     << endl;
            return;
        }

        CALuint major, minor, imp;
        r = calGetVersion(&major, &minor, &imp);
        if (r != CAL_RESULT_OK) {
            if (debug)
                cerr << "calGetVersion returned " << r
                     << ": " << calGetErrorString()
                     << endl;
            return;
        }
        if (debug)
            cerr << "CAL version " << major << "." << minor << "." << imp
                 << " detected" << endl;
        
        CALuint num_devices = 0;
        r = calDeviceGetCount(&num_devices);
        if (r != CAL_RESULT_OK) {
            if (debug)
                cerr << "calDeviceGetCount returned " << r
                     << ": " << calGetErrorString()
                     << endl;
            return;
        }

        if (debug)
            cerr << num_devices << " CAL devices" << endl;
        
        for (unsigned i = 0; i < num_devices; ++i) {
            CALdeviceinfo info;
            if ((r = calDeviceGetInfo(&info, i)) != CAL_RESULT_OK) {
                if (debug)
                    cerr << "calDeviceGetInfo for device " << i
                         << " returned " << r << ": "
                         << calGetErrorString() << endl;
                continue;
            }
            
            CALdeviceattribs attribs;
            attribs.struct_size = sizeof(CALdeviceattribs);
            if ((r = calDeviceGetAttribs(&attribs, i)) != CAL_RESULT_OK) {
                if (debug)
                    cerr << "calDeviceGetAttribs for device " << i
                         << " returned " << r << ": "
                         << calGetErrorString() << endl;
                continue;
            }

            if (debug) {
                cerr << "CAL device " << i << ":" << endl;
                cerr << "  target:              "
                     << printTarget(info.target) << endl;
                cerr << "  max width 1D:        "
                     << info.maxResource1DWidth << endl;
                cerr << "  max width 2D:        "
                     << info.maxResource2DWidth << endl;
                cerr << "  max height 2D:       "
                     << info.maxResource2DHeight << endl;
                cerr << "  local ram:           "
                     << attribs.localRAM << "mb" << endl;
                cerr << "  uncached remote ram: "
                     << attribs.uncachedRemoteRAM << "mb"
                     << endl;
                cerr << "  cached remote ram:   "
                     << attribs.cachedRemoteRAM << "mb"
                     << endl;
                cerr << "  engine clock speed:  "
                     << attribs.engineClock << "mhz"
                     << endl;
                cerr << "  memory clock speed:  "
                     << attribs.memoryClock << "mhz"
                     << endl;
                cerr << "  wavefront size:      "
                     << attribs.wavefrontSize << endl;
                cerr << "  number of SIMDs:     "
                     << attribs.numberOfSIMD << endl;
                cerr << "  double precision:    "
                     << attribs.doublePrecision << endl;
                cerr << "  local data share:    "
                     << attribs.localDataShare << endl;
                cerr << "  global data share:   "
                     << attribs.globalDataShare << endl;
                cerr << "  globalGPR:           " << attribs.globalGPR << endl;
                cerr << "  compute shader:      "
                     << attribs.computeShader << endl;
                cerr << "  memory export:       "
                     << attribs.memExport << endl;
                cerr << "  pitch alignment:     "
                     << attribs.pitch_alignment << " bytes" << endl;
                cerr << "  surface alignment:   "
                     << attribs.surface_alignment << " bytes" << endl;
            }
            
            CALdevice device;
            r = calDeviceOpen(&device, i);
            if ((r = calDeviceGetAttribs(&attribs, i)) != CAL_RESULT_OK) {
                if (debug)
                    cerr << "calDeviceOpen for device " << i
                         << " returned " << r << ": "
                         << calGetErrorString() << endl;
                continue;
            }
            
            Call_Guard guard(boost::bind(&calDeviceClose, device));
            
            CALdevicestatus status;
            status.struct_size = sizeof(CALdevicestatus);
            if ((r = calDeviceGetStatus(&status, device)) != CAL_RESULT_OK) {
                if (debug)
                    cerr << "calDeviceGetStatus for device " << i
                         << " returned " << r << ": "
                         << calGetErrorString() << endl;
                continue;
            }
            
            if (debug) {
                cerr << "  AVAILABLE: " << endl;
                cerr << "  local ram:           "
                     << status.availLocalRAM << "mb" << endl;
                cerr << "  uncached remote ram: "
                     << status.availUncachedRemoteRAM << "mb"
                     << endl;
                cerr << "  cached remote ram:   "
                     << status.availCachedRemoteRAM << "mb"
                     << endl;
            }
        }
    }
    
    ~Register_CAL()
    {
        if (debug)
            cerr << "unregistering CAL" << endl;
        calShutdown();
    }
} register_cal;

} // namespace ML
