/** value_instance.h                                 -*- C++ -*-
    Rémi Attab, 12 Mar 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Typeless reflected value.

*/

#pragma once

#include "value_description.h"

namespace Datacratic {


/******************************************************************************/
/* VALUE INSTANCE                                                             */
/******************************************************************************/

/** Proxy object to manipulate values through the reflection mechanism.

    This object does not own the manipulated object and it's up to the user to
    keep the referenced object valid throughought the lifetime of this object.

 */
struct ValueInstance
{
    template<typename T>
    explicit ValueInstance(T* instance) :
        instance(instance),
        description(getDefaultDescriptionShared<T>())
    {}

    ValueInstance(void* instance, std::shared_ptr<const ValueDescription> desc) :
        instance(instance), description(std::move(desc))
    {}


    const std::shared_ptr<const ValueDescription>&
    getDescription() const
    {
        return description;
    }

    void* getInstance() const
    {
        return instance;
    }


    template<typename T>
    T* cast()
    {
        auto desc = getDefaultDescriptionShared<T>();
        ExcCheck(description->isChildOf(desc.get()), "invalid cast");
        return static_cast<T*>(instance);
    }

    template<typename T>
    const T* cast() const
    {
        auto desc = getDefaultDescriptionShared<T>();
        ExcCheck(description->isChildOf(desc.get()), "invalid cast");
        return static_cast<T*>(instance);
    }

    void set(ValueInstance value) const
    {
        description->set(instance, value.instance, value.description.get());
    }

    ValueInstance operator[] (const std::string& name) const
    {
        auto field = description->getField(name);
        return ValueInstance(field.getFieldPtr(instance), field.description);
    }

private:
    void* instance;
    std::shared_ptr<const ValueDescription> description;
};


} // namespace Datacratic
