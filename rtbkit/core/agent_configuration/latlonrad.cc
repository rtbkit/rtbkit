/*
 * latlonrad.cc
 *
 *  Created on: 22/05/2014
 *      Author: pablin
 */
#include <boost/lexical_cast.hpp>
#include <rtbkit/core/agent_configuration/latlonrad.h>
#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm/remove_if.hpp>
#include <stdlib.h>
#include "jml/arch/exception.h"

using namespace RTBKIT;

LatLonRad::LatLonRad(float la, float lo, float r)
: lat(la), lon(lo), radius(r){ }

LatLonRad::LatLonRad()
: lat(0), lon(0), radius(-1){ }

LatLonRad
LatLonRad::createFromJson(const Json::Value & val) {

    LatLonRad llr;

    if (val.isMember("lat") && val.isMember("long") && val.isMember("radius")){
        llr.lat = val["lat"].asDouble();
        llr.lon = val["long"].asDouble();
        llr.radius = val["radius"].asDouble();
    } else {
        throw ML::Exception("Error parsing LatLonRad. It should have lat, "
                            "long and radius. There was given: %s",
                            val.asCString());
    }

    return llr;
}

void LatLonRad::fromJson(const Json::Value & val) {
    *this = createFromJson(val);
}

Json::Value LatLonRad::toJson() const {
    Json::Value result;
    result["lat"] = lat;
    result["long"] = lon;
    result["radius"] = radius;
    return result;
}

bool LatLonRad::empty() const {
    return radius == 0; // as no point could be inside a 0 radius circle/square
}


LatLonRadList
LatLonRadList::createFromJson(const Json::Value & val) {

    LatLonRadList llrl;

    for (auto jt = val.begin(), jend = val.end();  jt != jend;  ++jt) {
        LatLonRad llr = LatLonRad::createFromJson(*jt);
        llrl.latlonrads.push_back(llr);
    }

    return llrl;
}

void LatLonRadList::fromJson(const Json::Value & val) {
    *this = createFromJson(val);
}

Json::Value LatLonRadList::toJson() const {
    Json::Value result;
    for (unsigned i = 0;  i < latlonrads.size();  ++i)
        result[i] = latlonrads[i].toJson();
    return result;
}

bool LatLonRadList::empty() const {
    return latlonrads.empty();
}
