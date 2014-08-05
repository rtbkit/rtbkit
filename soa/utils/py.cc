#include <Python.h>
#include "py.h"

namespace Datacratic {

void initPythonEnv() {
    if (!Py_IsInitialized()) {
        setenv("PYTHONPATH", "build/x86_64/bin", 1);
        char pyhome[] = "virtualenv";
        Py_SetPythonHome(pyhome);
        Py_Initialize();
    }
}

} //namespace Datacratic
