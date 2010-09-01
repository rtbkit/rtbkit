/* jml_wrap_python.cc                                              -*- C++ -*-
   Jeremy Barnes, 17 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include "jml_wrap_python.h"
//#include "jml/boosting/classifier.h"

Classifier::
Classifier()
{
}

void
Classifier::
load(const char * filename)
{
}


std::string
Classifier::
print() const
{
    return "hello";
}

void classifierTrainingTool(int argc, char** argv)
{
}

Classifier * getClassifier()
{
    return new Classifier();
}
