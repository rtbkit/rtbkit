/* dataset_functions.h                                             -*- C++ -*-
   Jeremy Barnes, 15 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

*/


namespace ML {

struct Dataset_Function {
    virtual size_t input_vars() const = 0;
    virtual size_t output_vars() const = 0;
    virtual void train(const Training_Data & data);
};

/** Decorrelate the variables and output a list of decorrelated but
    equivalent variables.  Any which are perfectly correlated may be
    removed.
*/
struct Decorrelate_Function : public Dataset_Function {
};

struct Min_Function : public Dataset_Function {
};

struct Max_Function : public Dataset_Function {
};

struct Avg_Function : public Dataset_Function {
};

struct Plus_Function : public Dataset_Function {
};

} // namespace ML

