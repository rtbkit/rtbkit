/* defaulted.h                                            -*- C++ -*-
   Janusz Lewandowski, 25 July 2013
   Copyright (c) 2013 Daftcode.

   Default description of RTBKIT::defaulted<>.
*/

#pragma once

#include "soa/types/value_description.h"
#include "../defaulted.h"
#include "../cpp11.h"
#include <boost/algorithm/string.hpp>

namespace Datacratic
{
	template<class Inner, Inner def>
	struct DefaultDescription<RTBKIT::defaulted<Inner, def>>: public ValueDescriptionI<RTBKIT::defaulted<Inner, def>, ValueKind::ANY>
	{
		typedef RTBKIT::defaulted<Inner, def> T;

		DefaultDescription(ValueDescriptionT<Inner>* inner = getDefaultDescription((Inner*)0)) : inner(inner) {}

		std::unique_ptr<ValueDescriptionT<Inner>> inner;

		virtual void parseJsonTyped(T* val, JsonParsingContext& context) const override
		{
			try
			{
				Inner v;
				inner->parseJsonTyped(&v, context);
				*val = v;
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

		virtual void printJsonTyped(const T* val, JsonPrintingContext& context) const override
		{
			inner->printJsonTyped((Inner*) val, context);
		}

		virtual bool isDefaultTyped(const T* valp) const override
		{
			const auto& val = *valp;
			return val == def;
		}

		virtual void setDefaultTyped(T* valp) const override
		{
			auto& val = *valp;
			val = def;
		}

		virtual const ValueDescription& contained() const override
		{
			return *this->inner;
		}
	};

	template<class Inner, class D, D def>
	struct DefaultDescription<RTBKIT::defaulted3<Inner, D, def>>: public ValueDescriptionI<RTBKIT::defaulted3<Inner, D, def>, ValueKind::ANY>
	{
		typedef RTBKIT::defaulted3<Inner, D, def> T;

		DefaultDescription(ValueDescriptionT<Inner>* inner = getDefaultDescription((Inner*)0)) : inner(inner) {}

		std::unique_ptr<ValueDescriptionT<Inner>> inner;

		virtual void parseJsonTyped(T* val, JsonParsingContext& context) const override
		{
			try
			{
				Inner v;
				inner->parseJsonTyped(&v, context);
				*val = v;
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

		virtual void printJsonTyped(const T* val, JsonPrintingContext& context) const override
		{
			inner->printJsonTyped((Inner*) val, context);
		}

		virtual bool isDefaultTyped(const T* valp) const override
		{
			const auto& val = *valp;
			return val == def;
		}

		virtual void setDefaultTyped(T* valp) const override
		{
			auto& val = *valp;
			val = def;
		}

		virtual const ValueDescription& contained() const override
		{
			return *this->inner;
		}
	};
}
