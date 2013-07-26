/* cpp14.h                                                -*- C++ -*-
   Janusz Lewandowski, 25 July 2013
   Copyright (c) 2013 Daftcode.

   C++14 features implementable today.
*/

#pragma once

#include "cpp11.h"
#include "optional.h"
#include <memory>

namespace std14
{
	template<typename T, typename ...Args>
	std::unique_ptr<T> make_unique(Args&& ...args)
	{
		return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
	}
}
