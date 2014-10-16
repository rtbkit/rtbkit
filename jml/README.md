Jeremy's Machine Learning Library

This library contains two main parts:
* A system layer that exposes much functionality that wasn't readily available at the time the
  library was first built (1999) for things like multi threading, linear algebra, and much else.
  Most of this functionality is now available in modern, maintained packages (like Boost) and
  it is mostly of historical interest.
* A Machine Learning library containing implementations of a significant number of machine
  learning algorithms.  This code is quite old but extremely efficient and performant, and is
  used at the core of several machine-learning startups.

Currently, many of the source files mention the Affero GPL version 3.  These references will
eventually be changed; the library as a whole is made available under the Apache License,
version 2 (the text of the library is below).

Note that the library incorporates the Judy Arrays library from HP; that library is made
available under the LGPL license by HP and is *not* covered by the Apache License.

---

   Copyright 1999-2013 Jeremy Barnes, Idilia Inc, Datacratic Inc

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
