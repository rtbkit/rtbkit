/* optional.h                                             -*- C++ -*-
   Janusz Lewandowski, 25 July 2013
   Copyright (c) 2013 Daftcode.

   Default description of std::optional<>.
*/

#pragma once

#include "soa/types/value_description.h"
#include "../optional.h"
#include "../cpp11.h"
#include <boost/algorithm/string.hpp>

namespace Datacratic
{
	template<class Inner>
	struct DefaultDescription<std14::optional<Inner>>: public ValueDescriptionI<std14::optional<Inner>, ValueKind::ANY>
	{
		typedef std14::optional<Inner> T;

		DefaultDescription(ValueDescriptionT<Inner>* inner = getDefaultDescription((Inner*)0)) : inner(inner) {}

		std::unique_ptr<ValueDescriptionT<Inner>> inner;

		virtual void parseJsonTyped(T* valp, JsonParsingContext& context) const override
		{
			auto& val = *valp;
			try
			{
				Inner v;
				inner->parseJsonTyped(&v, context);
				val = v;
			}
			catch (ML::Exception& e)
			{
				std::cerr << context.printPath() << " (optional) is invalid: " << boost::algorithm::trim_copy(context.expectJson().toString()) << " (" << e.what() << ")." << std::endl;
			}
			catch (std::exception& e)
			{
				std::cerr << context.printPath() << " (optional) is invalid (" << e.what() << ")." << std::endl;
			}
		}

		virtual void printJsonTyped(const T* valp, JsonPrintingContext& context) const override
		{
			const auto& val = *valp;

			if (val)
				inner->printJsonTyped(&*val, context);
			else
				context.writeNull();
		}

		virtual bool isDefaultTyped(const T* valp) const override
		{
			const auto& val = *valp;
			return !val;
		}

		virtual void setDefaultTyped(T* valp) const override
		{
			auto& val = *valp;
			val = std14::nullopt;
		}

		virtual void* optionalMakeValueTyped(T* valp) const override
		{
			auto& val = *valp;
			if (!val)
				val.emplace();
			return &*val;
		}

		virtual const void* optionalGetValueTyped(const T* valp) const override
		{
			const auto& val = *valp;
			if (!val)
				throw ML::Exception("no value in optional field");
			return &*val;
		}

		virtual const ValueDescription& contained() const override
		{
			return *this->inner;
		}
	};
}
