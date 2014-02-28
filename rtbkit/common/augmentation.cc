/** augmentation.cc                                 -*- C++ -*-
    Rémi Attab, 05 Dec 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Struct to manipulate the result of augmentors.

*/

#include "rtbkit/common/augmentation.h"
#include "jml/arch/format.h"

#include <iostream>
#include <algorithm>

using namespace std;

namespace RTBKIT {


/******************************************************************************/
/* UTILITIES                                                                  */
/******************************************************************************/

namespace {

void mergeAugmentationData(Json::Value& lhs, const Json::Value& rhs)
{
    if (lhs.isNull()) {
        lhs = rhs;
        return;
    }

    ExcCheckEqual(lhs.type(), rhs.type(),
            "Augmentation data must be of the same type");

    if (lhs.isObject()) {
        vector<string> members = rhs.getMemberNames();
        for (auto it = members.begin(), end = members.end(); it != end; ++it) {
            ExcCheck(!lhs.isMember(*it), "Duplicated augmentation data.");
            lhs[*it] = rhs[*it];
        }
    }

    else if (lhs.isArray())
        for (size_t i = 0; i < rhs.size(); ++i) lhs.append(rhs[i]);

    // Last wins.
    else lhs = rhs;
}

} // namespace anonymous


/******************************************************************************/
/* AUMGMENTATION                                                              */
/******************************************************************************/

void
Augmentation::
mergeWith(const Augmentation& other)
{
    tags.insert(other.tags.begin(), other.tags.end());
    mergeAugmentationData(data, other.data);
}

Json::Value
Augmentation::
toJson() const
{
    Json::Value json(Json::objectValue);

    json["data"] = data;

    if (!tags.empty()) {
        Json::Value jsonTags(Json::arrayValue);

        for (auto it = tags.begin(), end = tags.end(); it != end; ++it)
            jsonTags.append(*it);

        json["tags"] = jsonTags;
    }

    return json;
}

Augmentation
Augmentation::
fromJson(const Json::Value& json)
{
    Augmentation aug;

    aug.data = json["data"];

    const Json::Value& jsonTags = json["tags"];
    for (auto it = jsonTags.begin(), end = jsonTags.end(); it != end; ++it)
        aug.tags.insert(it->asString());

    return aug;
}


/******************************************************************************/
/* AUGMENTATION LIST                                                          */
/******************************************************************************/

void
AugmentationList::
mergeWith(const AugmentationList& other)
{
    for (auto it = other.begin(), end = other.end(); it != end; ++it)
        (*this)[it->first].mergeWith(it->second);
}

Augmentation
AugmentationList::
filterForAccount(AccountKey account) const
{
    Augmentation result;

    while(true) {
        auto it = find(account);
        if (it != end()) result.mergeWith(it->second);

        if (account.empty()) break;
        account.pop_back();
    }

    return result;
}

vector<string>
AugmentationList::
tagsForAccount(AccountKey account) const
{
    vector<string> tags;

    while(true) {
        auto it = find(account);
        if (it != end()) {
            const auto& aug = it->second;
            tags.insert(tags.end(), aug.tags.begin(), aug.tags.end());
        }

        if (account.empty()) break;
        account.pop_back();
    }

    sort(tags.begin(), tags.end());

    auto it = unique(tags.begin(), tags.end());
    tags.erase(it, tags.end());

    return tags;
}


Json::Value
AugmentationList::
toJson() const
{
    Json::Value result(Json::arrayValue);

    for (auto it = begin(), last = end(); it != last; ++it) {
        Json::Value augJson (Json::objectValue);
        augJson["account"] = it->first.toJson();
        augJson["augmentation"] = it->second.toJson();
        result.append(augJson);
    }

    return result;
}

AugmentationList
AugmentationList::
fromJson(const Json::Value& json)
{
    AugmentationList list;

    for (auto it = json.begin(), end = json.end(); it != end; ++it) {
        AccountKey acc = AccountKey::fromJson((*it)["account"]);
        Augmentation aug = Augmentation::fromJson((*it)["augmentation"]);
        list.insert(make_pair(acc, aug));
    }

    return list;
}

} // namespace RTBKIT
