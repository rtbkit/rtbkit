/** augmentation.h                                 -*- C++ -*-
    Rémi Attab, 05 Dec 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Structures used to manipulate augmentations.

*/

#ifndef __rtb__augmentation_h__
#define __rtb__augmentation_h__

#include "rtbkit/common/account_key.h"
#include "soa/jsoncpp/value.h"

#include <set>
#include <string>

namespace RTBKIT {

/******************************************************************************/
/* AUGMENTATION                                                               */
/******************************************************************************/

/** Dynamic filtering rules and augmentation data to be associated with an
    account scope.
 */
struct Augmentation
{
    Augmentation() {}

    Augmentation(
            const std::set<std::string>& tags,
            const Json::Value& data = Json::Value()) :
        tags(tags), data(data)
    {}

    Augmentation(const Json::Value& data) : tags(), data(data) {}

    std::set<std::string> tags;
    Json::Value data;

    void mergeWith(const Augmentation& other);

    Json::Value toJson() const;
    static Augmentation fromJson(const Json::Value& json);
};

/** Agent name to stringified augemntation.
    In other words, it's a collapsed version of the AumgnetationList structure.
*/
typedef std::map<std::string, std::string> AgentAugmentations;


/******************************************************************************/
/* AUGMENTATION LIST                                                          */
/******************************************************************************/

// \todo may want to change this to unordered_map.
typedef std::map<AccountKey, Augmentation> AugmentationListBase;

/** Aggregation of the filtering rules on a per account prefix basis. */
struct AugmentationList : public AugmentationListBase
{

    void insertGlobal(const Augmentation& aug)
    {
        insert(std::make_pair(AccountKey(), aug));
    }

    /** Merge the given augmentation data with our current augmentation data. */
    void mergeWith(const AugmentationList& other);


    /** Returns all the tags and data associated with the given account. */
    Augmentation filterForAccount(AccountKey account) const;


    /** Returns all the tags associated with the given account. This functions
        is essentially the same as filterForAccount except that it doesn't merge
        the json data. Useful for filtering.
     */
    std::vector<std::string> tagsForAccount(AccountKey account) const;


    Json::Value toJson() const;
    static AugmentationList fromJson(const Json::Value& json);
};


} // namespace RTBKIT

#endif // __rtb__augmentation_h__
