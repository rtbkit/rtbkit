/* parameters.h                                                    -*- C++ -*-
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Desciption of parameters.  Used to allow polymorphic updates of
   parameters.
*/

#ifndef __neural__parameters_h__
#define __neural__parameters_h__

#define JML_NEURAL_PARAMETERS_KNOWN float double

namespace ML {

struct Parameters {
};

template<class Value>
struct Value_Vector : public Parameters {
    std::string name;
    

    const std::string & name() const;
};

template<class Value>
struct Value_Matrix : public Parameters {
};


} // namespace ML

#endif /* __neural__parameters_h__ */
