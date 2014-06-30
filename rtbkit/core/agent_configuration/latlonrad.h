/*
 * latlonrad.h
 *
 *  Created on: 22/05/2014
 *      Author: pablin
 */

#ifndef LATLONRAD_H_
#define LATLONRAD_H_

#include "jml/arch/exception.h"
#include "soa/jsoncpp/value.h"
#include <vector>

#include "soa/types/string.h"
#include <iostream>


namespace RTBKIT {

struct LatLonRad {

    /**
     * This component has a latitude, longitude and a radius. The json value
     * for this object has the form:
     * { "lat" : float, "long":float, "radius":float}.
     */

    LatLonRad(float la, float lo, float r);

    LatLonRad();

    static LatLonRad createFromJson(const Json::Value & val);

    void fromJson(const Json::Value & val);

    Json::Value toJson() const ;

    bool empty() const ;

    float lat;
    float lon;
    float radius; // MUST be in kilometers

};

struct LatLonRadList {

    /**
     * Stores many values of LatLonRad, in the form of [<latlongrad>, ...].
     */

    static LatLonRadList createFromJson(const Json::Value & val);

    void fromJson(const Json::Value & val);

    Json::Value toJson() const ;

    bool empty() const ;

    std::vector<LatLonRad> latlonrads;
};


}  // namespace RTBKIT


#endif /* LATLONRAD_H_ */
