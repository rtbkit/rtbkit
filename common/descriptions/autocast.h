/* autocast.h                                             -*- C++ -*-
   Janusz Lewandowski, 25 July 2013
   Copyright (c) 2013 Daftcode.

   Automatically type casting value descriptions.
*/

#pragma once

#include "all.h"

namespace Datacratic
{
	using std::string;
	using std14::optional;
	using RTBKIT::defaulted;
	using std::set;
	using std::vector;

	template<class T>
	struct AutoCastingDescription: public DefaultDescription<T> {};

	template<class Struct>
	struct AutoCastingStructureDescription: public StructureDescription<Struct>
	{
		AutoCastingStructureDescription(bool nullAccepted = false) : StructureDescription<Struct>(nullAccepted) {}

		template<typename V, typename Base>
		void addField(const string& name, V Base::* field, const string& comment = string(), ValueDescriptionT<V>* description = new AutoCastingDescription<V>())
		{
			StructureDescription<Struct>::addField(name, field, comment, description);
		}
	};

	template<class T>
	struct AutoCastingDescription<optional<T>>: public DefaultDescription<optional<T>>
	{
		AutoCastingDescription(ValueDescriptionT<T>* inner = new AutoCastingDescription<T>()) : DefaultDescription<optional<T>>(inner) {}
	};

	template<class T, T def>
	struct AutoCastingDescription<defaulted<T, def>>: public DefaultDescription<defaulted<T, def>>
	{
		AutoCastingDescription(ValueDescriptionT<T>* inner = new AutoCastingDescription<T>()) : DefaultDescription<defaulted<T, def>>(inner) {}
	};

	template<class T>
	struct AutoCastingDescription<set<T>>: public DefaultDescription<set<T>>
	{
		AutoCastingDescription(ValueDescriptionT<T>* inner = new AutoCastingDescription<T>()) : DefaultDescription<set<T>>(inner) {}
	};

	template<class T>
	struct AutoCastingDescription<vector<T>>: public DefaultDescription<vector<T>>
	{
		AutoCastingDescription(ValueDescriptionT<T>* inner = new AutoCastingDescription<T>()) : DefaultDescription<vector<T>>(inner) {}
	};

	template<class T> inline T expect(JsonParsingContext&);
	template<class T> inline T cast_to(const string&);

	template<class T>
	struct NumberDescription: public DefaultDescription<T>
	{
		virtual void parseJsonTyped(T* val, JsonParsingContext& context) const override
		{
			if (context.isString())
				*val = cast_to<T>(context.expectStringAscii());
			else
				*val = expect<T>(context);
		}
	};

	template<> inline int expect<int>(JsonParsingContext& ctx) { return ctx.expectInt(); }
	template<> inline uint expect<uint>(JsonParsingContext& ctx) { return ctx.expectUnsignedInt(); }
	template<> inline int64_t expect<int64_t>(JsonParsingContext& ctx) { return ctx.expectLongLong(); }
	template<> inline uint64_t expect<uint64_t>(JsonParsingContext& ctx) { return ctx.expectUnsignedLongLong(); }
	template<> inline float expect<float>(JsonParsingContext& ctx) { return ctx.expectFloat(); }
	template<> inline double expect<double>(JsonParsingContext& ctx) { return ctx.expectDouble(); }

	template<> inline int cast_to<int>(const string& s) { return stoi(s); }
	template<> inline uint cast_to<uint>(const string& s) { return stoul(s); }
	template<> inline int64_t cast_to<int64_t>(const string& s) { return stoll(s); }
	template<> inline uint64_t cast_to<uint64_t>(const string& s) { return stoul(s); }
	template<> inline float cast_to<float>(const string& s) { return stof(s); }
	template<> inline double cast_to<double>(const string& s) { return stod(s); }

	template<> struct AutoCastingDescription<int>: public NumberDescription<int> {};
	template<> struct AutoCastingDescription<uint>: public NumberDescription<uint> {};
	template<> struct AutoCastingDescription<int64_t>: public NumberDescription<int64_t> {};
	template<> struct AutoCastingDescription<uint64_t>: public NumberDescription<uint64_t> {};
	template<> struct AutoCastingDescription<float>: public NumberDescription<float> {};
	template<> struct AutoCastingDescription<double>: public NumberDescription<double> {};

	template <>
	struct AutoCastingDescription<bool>: public DefaultDescription<bool>
	{
		virtual void parseJsonTyped(bool* val, JsonParsingContext& context) const override
		{
			if (context.isString())
				*val = stoi(context.expectStringAscii());
			else if (context.isNumber())
				*val = context.expectInt();
			else
				*val = context.expectBool();
		}
	};
}
