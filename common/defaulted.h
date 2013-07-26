/* defaulted.h                                            -*- C++ -*-
   Janusz Lewandowski, 25 July 2013
   Copyright (c) 2013 Daftcode.

   Templates simulating C++11's non-static data member initializers.
*/

#pragma once

namespace RTBKIT
{
	template <class T, T def>
	class defaulted
	{
		T val;

	public:
		explicit inline defaulted(): val(def) {}
		explicit inline defaulted(T&& val): val(std::move(val)) {}
		explicit inline defaulted(const T& val): val(val) {}

		inline operator T&() { return val; }
		inline operator const T&() const { return val; }

		inline T& operator*() { return val; }
		inline const T& operator*() const { return val; }

		inline defaulted<T, def>& operator=(T&& v) { val = std::move(v); return *this; }
		inline defaulted<T, def>& operator=(const T& v) { val = v; return *this; }
	};

	// Useful for types that can't be template parameters
	template <class T, class D, D def>
	class defaulted3
	{
		T val;

	public:
		explicit inline defaulted3(): val(def) {}
		explicit inline defaulted3(T&& val): val(std::move(val)) {}
		explicit inline defaulted3(const T& val): val(val) {}

		inline operator T&() { return val; }
		inline operator const T&() const { return val; }

		inline T& operator*() { return val; }
		inline const T& operator*() const { return val; }

		inline defaulted3<T, D, def>& operator=(T&& v) { val = std::move(v); return *this; }
		inline defaulted3<T, D, def>& operator=(const T& v) { val = v; return *this; }
	};

}
