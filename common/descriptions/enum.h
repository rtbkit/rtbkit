/* enum.h                                                 -*- C++ -*-
   Janusz Lewandowski, 25 July 2013
   Copyright (c) 2013 Daftcode.

   Premade classes for describing enums.
*/

#pragma once

#include "soa/types/value_description.h"
#include "../cpp11.h"
#include <string>

namespace Datacratic
{
	template<class T>
	struct IntegerEnumDescription: public ValueDescriptionI<T, ValueKind::INTEGER>
	{
		virtual void parseJsonTyped(T* val, JsonParsingContext& context) const override
		{
			*val = T(context.expectInt());
		}

		virtual void printJsonTyped(const T* val, JsonPrintingContext& context) const override
		{
			context.writeInt(int(*val));
		}
	};

	template<class T, T (*toEnum)(const std::string&), std::string (*toString)(T)>
	struct StringEnumDescription: public ValueDescriptionI<T, ValueKind::STRING>
	{
		virtual void parseJsonTyped(T* val, JsonParsingContext& context) const override
		{
			*val = toEnum(context.expectString());
		}

		virtual void printJsonTyped(const T* val, JsonPrintingContext& context) const override
		{
			context.writeString(toString(*val));
		}
	};
}
