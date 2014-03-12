/* xml_helpers.cc                                                  -*- C++ -*-
   Jeremy Barnes, 12 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.
   
   Helper functions to deal with XML.   
*/

#include <memory>

#include "jml/utils/string_functions.h"

#include "xml_helpers.h"


using namespace std;


namespace Datacratic {

const tinyxml2::XMLNode *
extractNode(const tinyxml2::XMLNode * element, const string & path)
{
    using namespace std;

    vector<string> splitPath = ML::split(path, '/');
    const tinyxml2::XMLNode * p = element;
    for (unsigned i = 0;  i < splitPath.size();  ++i) {
        p = p->FirstChildElement(splitPath[i].c_str());
        if (!p) {
            //element->GetDocument()->Print();
            throw ML::Exception("required key " + splitPath[i]
                                + " not found on path " + path);
        }
    }

    return p;
}

}
