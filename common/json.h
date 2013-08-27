/* json.h                                                 -*- C++ -*-
   Janusz Lewandowski, 25 July 2013
   Copyright (c) 2013 Daftcode.

   JSON parsing/printing utils
*/

#pragma once

#include "soa/types/json_parsing.h"
#include "soa/types/json_printing.h"
#include "soa/types/value_description.h"
#include <sstream>
#include <cxxcompat/memory>

namespace RTBKIT
{
	template <class T> void fillFromJson(T& v, const std::string& json)
	{
		static const Datacratic::DefaultDescription<T> desc;

		Datacratic::StringJsonParsingContext jsonContext(json);
		desc.parseJsonTyped(&v, jsonContext);
	}

	template <class T> void fillFromJson(T& v, const Json::Value& json)
	{
		static const Datacratic::DefaultDescription<T> desc;

		Datacratic::StructuredJsonParsingContext jsonContext(json);
		desc.parseJsonTyped(&v, jsonContext);
	}

	template <class T> void fillFromJson(T& v, ML::Parse_Context& json)
	{
		static const Datacratic::DefaultDescription<T> desc;

		Datacratic::StreamingJsonParsingContext jsonContext(json);
		desc.parseJsonTyped(&v, jsonContext);
	}

	template <class T, class J> T fromJson(J&& json)
	{
		T v;
		fillFromJson(v, std::forward<J>(json));
		return v;
	}

	template <class T, class J> std::unique_ptr<T> newFromJson(J&& json)
	{
		auto v = std::make_unique<T>();
		fillFromJson(*v, std::forward<J>(json));
		return v;
	}

	template <class T> std::string toJson(const T& v)
	{
		static const Datacratic::DefaultDescription<T> desc;

		std::stringstream strstr;

		Datacratic::StreamJsonPrintingContext ctx(strstr);
		desc.printJsonTyped(&v, ctx);

		return strstr.str();
	}

	template <class T> Json::Value toJsonValue(const T& v)
	{
		static const Datacratic::DefaultDescription<T> desc;

		Datacratic::StructuredJsonPrintingContext ctx;
		desc.printJsonTyped(&v, ctx);

		return ctx.output;
	}

	template <class T> std::string toPrettyJson(const T& v)
	{
		std::stringstream strstr;
		strstr << toJsonValue(v);
		return strstr.str();
	}
}
