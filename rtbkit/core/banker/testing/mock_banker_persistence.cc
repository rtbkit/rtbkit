/* mock_banker_persistence.cc
   Wolfgang Sourdeau, 14 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Mock banker persistence for testing classes that depend on persistence
   storage.
*/

#include <iostream>
#include <jml/utils/exc_assert.h>

#include "mock_banker_persistence.h"

using namespace std;

namespace RTBKIT {

MockBankerPersistence::
MockBankerPersistence()
    : disableSaves(false)
{
}

void
MockBankerPersistence::
prepareLoad(shared_ptr<Accounts> accounts,
            BankerPersistence::PersistenceCallbackStatus status,
            std::string info)
{
    accountsQ.push_back(accounts);
    statusQ.push_back(status);
    infoQ.push_back(info);
    opsQ.push_back(2);
}

void
MockBankerPersistence::
prepareSave(BankerPersistence::PersistenceCallbackStatus status,
            std::string info)
{
    ExcAssertEqual(disableSaves, false);
    statusQ.push_back(status);
    infoQ.push_back(info);
    opsQ.push_back(1);
}

void
MockBankerPersistence::
loadAll(const std::string & topLevelKey, OnLoadedCallback onLoaded)
{
    int op = opsQ.front();
    if (op == 2) {
        opsQ.pop_front();
        onLoaded(accountsQ.front(), statusQ.front(), infoQ.front());
        accountsQ.pop_front();
        statusQ.pop_front();
        infoQ.pop_front();
    }
    else {
        throw ML::Exception("operation code must be 2");
    }
}

void
MockBankerPersistence::
saveAll(const Accounts & toSave, OnSavedCallback onSaved)
{
    if (disableSaves) {
        onSaved(BankerPersistence::SUCCESS, "");
        cerr << __FUNCTION__ << ": invocation ignored" << endl;
        return;
    }
    int op = opsQ.front();
    if (op == 1) {
        opsQ.pop_front();
        onSaved(statusQ.front(), infoQ.front());
        statusQ.pop_front();
        infoQ.pop_front();
    }
    else {
        throw ML::Exception("operation code must be 1");
    }
}

} // namespace RTBKIT
