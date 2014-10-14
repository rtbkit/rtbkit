/** submission_info.cc                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Implementation for submission info.

*/

#include "submission_info.h"

using namespace std;
using namespace ML;

namespace RTBKIT {

/*****************************************************************************/
/* SUBMISSION INFO                                                           */
/*****************************************************************************/

std::string
SubmissionInfo::
serializeToString() const
{
    ostringstream stream;
    ML::DB::Store_Writer writer(stream);
    int version = 5;
    writer << version
           << bidRequestStr
           << bidRequestStrFormat
           << augmentations.toString()
           << pendingWinEvents
           << earlyCampaignEvents;
    bid.serialize(writer);
    return stream.str();
}

void
SubmissionInfo::
reconstituteFromString(const std::string & str)
{
    istringstream stream(str);
    ML::DB::Store_Reader store(stream);
    int version;
    store >> version;
    if (version < 1 || version > 5)
        throw ML::Exception("bad version %d", version);
    store >> bidRequestStr;
    if (version == 5)
    {
        store >> bidRequestStrFormat ;
    }
    if (version > 1) {
        string s;
        store >> s;
        augmentations = s;
    }
    else augmentations.clear();
    if (version == 3) {
        vector<vector<string> > msg1, msg2;
        store >> msg1 >> msg2;
        if (!msg1.empty() || !msg2.empty())
            cerr << "warning: discarding early events from old format"
                 << endl;
        pendingWinEvents.clear();
        earlyCampaignEvents.clear();
    }
    else if (version > 3) {
        store >> pendingWinEvents >> earlyCampaignEvents;
    }
    else {
        pendingWinEvents.clear();
        earlyCampaignEvents.clear();
    }
    bid.reconstitute(store);

    if (!bidRequestStr.empty())
        bidRequest.reset(BidRequest::parse(bidRequestStrFormat, bidRequestStr));
    else bidRequest.reset();
}



} // namepsace RTBKIT
