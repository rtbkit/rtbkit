/* mock_banker_persistence.cc                                      -*- C++ -*-
   Wolfgang Sourdeau, 14 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Mock banker persistence for testing classes that depend on persistence
   storage.
*/

#ifndef MOCK_BANKER_PERSISTENCE_H
#define MOCK_BANKER_PERSISTENCE_H

#include <deque>
#include <memory>
#include <string>

#include "rtbkit/core/banker/master_banker.h"

namespace RTBKIT {

struct MockBankerPersistence : public BankerPersistence {
    MockBankerPersistence();

    bool disableSaves; /* make "saveAll" void */

    std::deque<BankerPersistence::PersistenceCallbackStatus> statusQ;
    std::deque<std::string> infoQ;
    std::deque<std::shared_ptr<Accounts>> accountsQ;

    void prepareLoad(std::shared_ptr<Accounts> accounts,
                     BankerPersistence::PersistenceCallbackStatus status,
                     std::string info);
    void prepareSave(BankerPersistence::PersistenceCallbackStatus status,
                     std::string info);

    void loadAll(const std::string & topLevelKey, OnLoadedCallback onLoaded);
    void saveAll(const Accounts & toSave, OnSavedCallback onSaved);

private:
    std::deque<int> opsQ; // 1 = save; 2 = load
};

} // namespace RTBKIT

#endif /* MOCK_BANKER_PERSISTENCE_H */
