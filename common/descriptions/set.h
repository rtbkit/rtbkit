/* set.h                                                  -*- C++ -*-
   Janusz Lewandowski, 25 July 2013
   Copyright (c) 2013 Daftcode.

   Default description of std::set<>.
*/

#pragma once

#include "soa/types/value_description.h"
#include <set>
#include "../cpp11.h"

// TODO check compiler version here
#define STD_SET_HAS_EMPLACE 0

namespace Datacratic
{
	template<class Inner>
	struct DefaultDescription<std::set<Inner>>: public ValueDescriptionI<std::set<Inner>, ValueKind::ARRAY>
	{
		typedef std::set<Inner> T;

		DefaultDescription(ValueDescriptionT<Inner>* inner = getDefaultDescription((Inner*)0)) : inner(inner) {}

		std::unique_ptr<ValueDescriptionT<Inner>> inner;

		virtual void parseJsonTyped(T* valp, JsonParsingContext & context) const override
		{
			auto& val = *valp;

			val.clear();

			if (!context.isArray())
				context.exception("expected array of " + inner->typeName);

			context.forEachElement([&]()
			{
				Inner el;
				inner->parseJsonTyped(&el, context);
				#if STD_SET_HAS_EMPLACE
					val.emplace(std::move(el));
				#else
					val.insert(std::move(el));
				#endif
			});
		}

		virtual void printJsonTyped(const T* valp, JsonPrintingContext & context) const override
		{
			const auto& val = *valp;

			context.startArray(val.size());

			for (const auto& v : val)
			{
				context.newArrayElement();
				inner->printJsonTyped(&v, context);
			}

			context.endArray();
		}

		virtual bool isDefaultTyped(const T* valp) const override
		{
			return valp->empty();
		}

		virtual size_t getArrayLength(void* valp) const override
		{
			const auto& val = *reinterpret_cast<const T*>(valp);
			return val.size();
		}

		// They don't make any sense here.
		// virtual void* getArrayElement(void* valp, uint32_t element) const override
		// virtual const void* getArrayElement(const void* valp, uint32_t element) const override
		// virtual void setArrayLength(void* valp, size_t newLength) const override

		virtual const ValueDescription& contained() const override
		{
			return *this->inner;
		}
	};
}
