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

namespace RTBKIT
{
	template <class T> void fillFromJson(T& v, const std::string& json)
	{
		static const Datacratic::DefaultDescription<T> desc;

		Datacratic::StringJsonParsingContext jsonContext(json);
		desc.parseJson(&v, jsonContext);
	}

	template <class T> void fillFromJson(T& v, const Json::Value& json)
	{
		static const Datacratic::DefaultDescription<T> desc;

		Datacratic::StructuredJsonParsingContext jsonContext(json);
		desc.parseJson(&v, jsonContext);
	}

	template <class T> T fromJson(const std::string& json)
	{
		T v;
		fillFromJson(v, json);
		return v;
	}

	template <class T> T fromJson(const Json::Value& json)
	{
		T v;
		fillFromJson(v, json);
		return v;
	}

	template <class T> std::string toJson(const T& v)
	{
		static const Datacratic::DefaultDescription<T> desc;

		std::stringstream strstr;

		Datacratic::StreamJsonPrintingContext ctx(strstr);
		desc.printJson(&v, ctx);

		return strstr.str();
	}

	template <class T> Json::Value toJsonValue(const T& v)
	{
		static const Datacratic::DefaultDescription<T> desc;

		Datacratic::StructuredJsonPrintingContext ctx;
		desc.printJson(&v, ctx);

		return ctx.output;
	}

	template <class T> std::string toPrettyJson(const T& v)
	{
		std::stringstream strstr;
		strstr << toJsonValue(v);
		return strstr.str();
	}
}
