// Copyright (C) 2009 Jeremy Barnes                                -*- C++ -*-

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

%module jml 

%feature("autodoc", "1");

%{
#include "jml/arch/demangle.h"
#include <cxxabi.h>
%}

// Cause things that call out to jml to handle exceptions so that we don't
// crash the python interpreter
%exception {
    try {
        $action
    }
    catch (const std::exception & exc) {
        PyErr_SetString(PyExc_Exception, ("JML library threw exception of type " + ML::demangle(typeid(exc).name()) + ": " + exc.what()).c_str());
        return NULL;
    }
    catch (...) {
        PyErr_SetString(PyExc_Exception, ("JML library threw exception of type " + ML::demangle(abi::__cxa_current_exception_type()->name())).c_str());
        return NULL;
        
    }
}

namespace ML {
namespace DB {
struct Store_Reader;
struct Store_Writer;

} // namespace DB

struct Parse_Context;
};

%include "distribution.i"
%include "feature.i"
%include "feature_set.i"
%include "feature_info.i"
%include "feature_space.i"
%include "training_data.i"
%include "classifier.i"



