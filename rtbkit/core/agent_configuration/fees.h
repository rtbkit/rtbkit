/* fees.h                                               -*- C++ -*-
   JS Bejeau, 13 June 2015
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "soa/service/service_base.h"
#include "rtbkit/common/plugin_interface.h"
#include "rtbkit/common/currency.h"
#include "soa/jsoncpp/json.h"

using namespace std;

namespace RTBKIT {

/***************************/
// FEES
/***************************/
struct Fees {
    Fees();
    ~Fees();

    typedef function<shared_ptr<Fees> (Json::Value const &)> FeesFactory;
    static void RegisterFees(string name, FeesFactory factory);

    static shared_ptr<Fees> createFees(Json::Value const & json);

    virtual Amount applyFees(Amount) const = 0;
    virtual Json::Value toJson() const = 0;
};

/***************************/
// NULL FEES
/***************************/
struct NullFees : public Fees {
    NullFees(Json::Value const &);
    ~NullFees();

    static shared_ptr<Fees> RegisterSpecificFees(Json::Value const & json) {
        shared_ptr<Fees> ret(new NullFees(json));
        return ret;
    };

    static shared_ptr<Fees> createNullFees();

    virtual Amount applyFees(Amount) const ;
    virtual Json::Value toJson() const ;

};


/***************************/
// FLAT FEES
/***************************/
struct FlatFees : public Fees {
    FlatFees(Json::Value const &);
    ~FlatFees();

    static shared_ptr<Fees> RegisterSpecificFees(Json::Value const & json) {
        shared_ptr<Fees> ret(new FlatFees(json));
        return ret;
    };

    virtual Amount applyFees(Amount) const ;
    virtual Json::Value toJson() const ;

private :
    Amount a;

};

/***************************/
// LINEAR FEES
/***************************/
struct LinearFees : public Fees {
    LinearFees(Json::Value const &);
    ~LinearFees();

    static shared_ptr<Fees> RegisterSpecificFees(Json::Value const & json) {
        shared_ptr<Fees> ret(new LinearFees(json));
        return ret;
    };

    virtual Amount applyFees(Amount) const ;
    virtual Json::Value toJson() const ;

private :
    double a;
    double b;
};

} // namespace RTBKIT
