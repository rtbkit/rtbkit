/* py_feature_set.cc
   Jeremy Barnes, 11 April 2005
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

   Bindings for python into the Feature_Set and associated classes.
*/

#include <boost/python.hpp>
#include "jml/boosting/feature_set.h"

using namespace boost::python;
using namespace ML;
using namespace std;


namespace ML {
namespace Python {

struct Feature_Wrapper {

    static std::string str(const Feature * feat)
    {
        return feat->print();
    }

    static std::string repr(const Feature * feat)
    {
        return format("Feature(%d, %d, %d)", feat->type(),
                      feat->arg1(), feat->arg2());
    }
};

struct Feature_Set_Wrapper : Feature_Set, wrapper<Feature_Set> {

    /* Pure virtual function get_data */
    virtual boost::tuple<const Feature *, const float *, int, int, size_t>
    get_data(bool need_sorted = false) const
    {
        return this->get_override("get_data")(need_sorted);
    }
    
    /* Pure virtual function make_copy() */

    virtual Feature_Set_Wrapper * make_copy() const
    {
        return this->get_override("make_copy")();
    }

    /* Pure virtual function sort() */
    virtual void sort()
    {
        this->get_override("sort")();
    }

    static tuple
    getitemA(const Feature_Set * fs, int index)
    {
        if (index < 0) index = fs->size() - index;
        if (index < 0 || index >= fs->size())
            throw Exception("invalid index");
        pair<Feature, float> result = (*fs)[index];
        return make_tuple(result.first, result.second);
    }

    static float getitemB(const Feature_Set * fs, const Feature & feature)
    {
        pair<const_iterator, const_iterator> range
            = fs->find(feature);
        if (range.first == range.second)
            throw Exception("No feature found");
        else if (range.first + 1 != range.second)
            throw Exception("More than one feature found");
        else return (*range.first).second;
    }

    static string str(const Feature_Set * fs)
    {
        string result = "(\n";
        for (Feature_Set::const_iterator it = fs->begin();
             it != fs->end();  ++it) {
            result += " ( " + (*it).first.print() + ", "
                + format("%g )\n", (*it).second);
        }
        result += ")";
        return result;
    }

    static string repr(const Feature_Set * fs)
    {
        string result = "( ";
        for (Feature_Set::const_iterator it = fs->begin();  it != fs->end();
             ++it) {
            result += " ( " + Feature_Wrapper::repr(&(*it).first) + ", "
                + format("%g )\n", (*it).second);
        }
        result += " )";
        return result;
    }
};


struct Mutable_Feature_Set_Wrapper
    : Mutable_Feature_Set, wrapper<Mutable_Feature_Set> {

    Mutable_Feature_Set_Wrapper()
    {
    }
    
    Mutable_Feature_Set_Wrapper(const Mutable_Feature_Set & other)
        : Mutable_Feature_Set(other)
    {
    }

    void add1(const Feature & feat)
    {
        add(feat);
    }

    tuple getitemA(int index) const
    {
        if (index < 0) index = size() + index;
        if (index < 0 || index >= size())
            throw Exception("invalid index");
        pair<Feature, float> result = operator [] (index);
        return make_tuple(result.first, result.second);
    }
    
    float getitemB(const Feature & feature) const
    {
        pair<const_iterator, const_iterator> range
            = find(feature);
        if (range.first == range.second)
            throw Exception("No feature found");
        else if (range.first + 1 != range.second)
            throw Exception("More than one feature found");
        else return range.first->second;
    }

    void setitemA(const Feature & feature, float val)
    {
        if (count(feature))
            replace(feature, val);
        else add(feature, val);
    }

    void setitemB(int index, const tuple & vals)
    {
        if (index < 0) index = size() + index;
        if (index < 0 || index >= size())
            throw Exception("invalid index");

        Feature feat = extract<Feature>(vals[0]);
        float value = extract<float>(vals[1]);

        operator [] (index) = make_pair(feat, value);
    }

    void delitemA(const Feature & feature)
    {
        pair<iterator, iterator> range = find(feature);
        features.erase(range.first, range.second);
    }

    void delitemB(int index)
    {
        if (index < 0) index = size() + index;
        if (index < 0 || index >= size())
            throw Exception("invalid index");
        features.erase(features.begin() + index);
    }

    struct iterator_wrap : public iterator {
        typedef boost::tuple<Feature, float> value_type;

        iterator_wrap(iterator it)
            : iterator(it)
        {
        }

        /* Coerce to a tuple to get the proper return type. */
        value_type & operator * () const
        {
            return reinterpret_cast<value_type &>(iterator::operator *());
        }
    };

    iterator_wrap beginA() { return begin(); }
    iterator_wrap endA() { return end(); }

    string str() const
    {
        string result = "(\n";
        for (const_iterator it = begin();  it != end();  ++it) {
            result += " ( " + it->first.print() + ", "
                + format("%g )\n", it->second);
        }
        result += ")";
        return result;
    }

    string repr() const
    {
        string result = "Mutable_Feature_Set ( ";
        for (const_iterator it = begin();  it != end();  ++it) {
            result += " ( " + Feature_Wrapper::repr(&it->first) + ", "
                + format("%g )\n", it->second);
        }
        result += " )";
        return result;
    }
};


void export_feature_set()
{
    class_<Feature>("Feature", init<>())
        .def(init<Feature_Id>())
        .def(init<Feature_Id, Feature_Id>())
        .def(init<Feature_Id, Feature_Id, Feature_Id>())
        .def_readwrite("type", &Feature::type_)
        .def_readwrite("arg1", &Feature::arg1_)
        .def_readwrite("arg2", &Feature::arg2_)
        .def("hash", &Feature::hash)
        .def(self < self)
        .def(self == self)
        .def(self != self)
        .def("print", &Feature::print)
        .def("__str__", &Feature_Wrapper::str)
        .def("__repr__", &Feature_Wrapper::repr)
        .def("__hash__", &Feature::hash);

    class_<Feature_Set_Wrapper, boost::noncopyable>("Feature_Set")
        .def("get_data", pure_virtual(&Feature_Set::get_data))
        .def("size", &Feature_Set::size)
        .def("__len__", &Feature_Set::size)
        .def("__getitem__", &Feature_Set_Wrapper::getitemA)
        .def("__getitem__", &Feature_Set_Wrapper::getitemB)
        .def("__contains__", &Feature_Set::contains)
#if 0
        .def("__iter__", range(&Feature_Set_Wrapper::begin,
                               &Feature_Set_Wrapper::end))
#endif
        .def("value", &Feature_Set::value)
        .def("count", &Feature_Set::count)
        .def("contains", &Feature_Set::contains)
#if 0
        .def("find", range(&Feature_Set_Wrapper::feature_begin,
                           &Feature_Set_Wrapper::feature_end))
#endif
        .def("sort", pure_virtual(&Feature_Set::sort))
        .def("make_copy", pure_virtual(&Feature_Set::make_copy),
             return_value_policy<manage_new_object>())
        .def("__str__", &Feature_Set_Wrapper::str)
        .def("__repr__", &Feature_Set_Wrapper::repr);

    class_<Mutable_Feature_Set_Wrapper,
           bases<Feature_Set> >("Mutable_Feature_Set", init<>())
        .def("add", &Mutable_Feature_Set_Wrapper::add)
        .def("add", &Mutable_Feature_Set_Wrapper::add1)
        .def("replace", &Mutable_Feature_Set::replace)
        .def("reserve", &Mutable_Feature_Set::reserve)
        .def("clear", &Mutable_Feature_Set::clear)
        .def("__getitem__", &Mutable_Feature_Set_Wrapper::getitemA)
        .def("__getitem__", &Mutable_Feature_Set_Wrapper::getitemB)
        .def("__setitem__", &Mutable_Feature_Set_Wrapper::setitemA)
        .def("__setitem__", &Mutable_Feature_Set_Wrapper::setitemB)
        .def("__delitem__", &Mutable_Feature_Set_Wrapper::delitemA)
        .def("__delitem__", &Mutable_Feature_Set_Wrapper::delitemB)
        .def("__iter__", range(&Mutable_Feature_Set_Wrapper::beginA,
                               &Mutable_Feature_Set_Wrapper::endA))
        .def("__str__", &Mutable_Feature_Set_Wrapper::str)
        .def("__repr__", &Mutable_Feature_Set_Wrapper::repr);

    def("escape_feature_name", escape_feature_name);
}

} // namespace Python
} // namespace ML
