/* training_params.h                                               -*- C++ -*-
   Jeremy Barnes, 16 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Structure with training parameters.
*/

#ifndef __boosting__training_params_h__
#define __boosting__training_params_h__


#include <map>
#include <boost/any.hpp>
#include <string>



namespace ML {


/*****************************************************************************/
/* TRAINING_PARAMS                                                           */
/*****************************************************************************/

/** This is a class that describes training parameters.  It is done using
    boost::any fields, so that the parameters can be passed through
    without requiring anyone to know what they are.
*/

struct Training_Params : public std::map<std::string, boost::any> {

    /** Returns an extra named parameter, given by the key.  Returns an
        empty \p any if not found. */
    const boost::any & operator [] (const std::string & key) const
    {
        const_iterator loc = find(key);
        static const boost::any EMPTY;
        if (loc == end()) return EMPTY;
        else return loc->second;
    }
    
    /** Returns an extra named parameter, given by the key.  Returns an
        empty \p any if not found. */
    boost::any & operator [] (const std::string & key)
    {
        return std::map<std::string, boost::any>::operator [] (key);
    }
    
    /** Gets an extra named parameter of the specified type.
        Should be called like
        \code
            context.get<type>("key")
        \endcode
           
        Example:
 
        \code
            ML_Trainer_Context context;
            try {
                string trainer_name = context.get<string>("trainer");
                cout << "trainer used was " << trainer_name << endl;
            } catch (...) {
                cout << "unknown trainer used" << endl;
            }
        \endcode
           
        \param Obj        the type of the object to use
        \param key        the key of the object
       
        Throws an exception if not found or if type conversion failed.
    */
    template<class Obj> Obj get(const std::string & key) const
    {
        std::map<std::string, boost::any>::const_iterator loc = find(key);
        if (loc == end())
            throw ML::Exception("key \"" + key + "\" not found in "
                                    "context object");
        try {
            return boost::any_cast<Obj>(loc->second);
        }
        catch (const boost::bad_any_cast & exc) {
            throw Exception("Training_Params: attempting to read key '"
                            + key + ": param type "
                            + demangle(loc->second.type().name())
                            + " doesn't match requested type "
                            + demangle(typeid(Obj).name()));
        }
    }
    
    /** Returns true if the context contains an object with the given
        key. */
    bool contains(const std::string & key) const
    {
        return count(key);
    }
    
    template<class Obj> void set(const std::string & key, const Obj & obj)
    {
        (*this)[key] = obj;
    }
};


} // namespace ML



#endif /* __boosting__training_params_h__ */

