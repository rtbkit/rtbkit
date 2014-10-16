/* jml_wrap_python.h                                               -*- C++ -*-
   Jeremy Barnes, 17 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Wrapper classes for python interface.
*/

#ifndef __jml_wrap_python_h__
#define __jml_wrap_python_h__

#include <string>

class Classifier {
public:
    Classifier();
    void load(const char * filename);
    std::string print() const;
};

void classifierTrainingTool(int argc, char** argv);
Classifier * getClassifier();


#endif /* __jml__wrap_python_h__ */

