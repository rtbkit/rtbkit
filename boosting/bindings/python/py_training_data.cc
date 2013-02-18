/* py_training_data.cc
   Jeremy Barnes, 10 April 2005
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

   Bindings for python into the Training_Data class.
*/

#include <boost/python.hpp>
#include "jml/boosting/training_data.h"
#include "jml/boosting/training_index.h"

using namespace boost::python;
using namespace ML;

namespace ML {
namespace Python {

struct Training_Data_Wrapper : Training_Data, wrapper<Training_Data> {

    /* Virtual function partition */

    virtual std::vector<std::shared_ptr<Training_Data> >
    partition(const std::vector<float> & sizes, bool random = true,
              const Feature & group_feature = MISSING_FEATURE) const
    {
        if (override ovr = this->get_override("partition"))
            return ovr(sizes, random, group_feature);
        return Training_Data::partition(sizes, random, group_feature);
    }

    std::vector<std::shared_ptr<Training_Data> >
    def_partition(const std::vector<float> & sizes, bool random,
                  const Feature & group_feature) const
    {
        return Training_Data::partition(sizes, random, group_feature);
    }

    std::vector<std::shared_ptr<Training_Data> >
    partition1(const std::vector<float> & sizes) const
    {
        return partition(sizes);
    }

    std::vector<std::shared_ptr<Training_Data> >
    partition2(const std::vector<float> & sizes, bool random) const
    {
        return partition2(sizes, random);
    }

    /* Virtual function make_copy() */

    virtual Training_Data * make_copy() const
    {
        if (override ovr = this->get_override("make_copy"))
            return ovr();
        return Training_Data::make_copy();
    }

    Training_Data * def_make_copy() const
    {
        return Training_Data::make_copy();
    }

    /* Virtual function make_type() */

    virtual Training_Data * make_type() const
    {
        if (override ovr = this->get_override("make_type"))
            return ovr();
        return Training_Data::make_type();
    }

    Training_Data * def_make_type() const
    {
        return Training_Data::make_type();
    }

    using Training_Data::data_;
    using Training_Data::index_;
    using Training_Data::feature_space_;
    using Training_Data::dirty_;
    using Training_Data::generate_index;
};

void export_training_data()
{
    class_<Training_Data_Wrapper, boost::noncopyable>("Training_Data")
        .def("init", &Training_Data::init)
        .def("clear", &Training_Data::clear)
        .def("swap", &Training_Data::swap)
        .def("all_features", &Training_Data::all_features)
        .def("example_count", &Training_Data::example_count)
        .def("empty", &Training_Data::empty)
        .def("feature_space", &Training_Data::feature_space)
        .def("dump", &Training_Data::dump)
        .def("serialize", &Training_Data::serialize)
        .def("save", &Training_Data::save)
        .def("reconstitute" ,&Training_Data::reconstitute)
        .def("load", &Training_Data::load)
        .def("__getitem__", &Training_Data::operator [],
             return_internal_reference<1>())
#if 0
        .def("__setitem__", &Training_Data::operator [],
             return_internal_reference<1>())
        .def("__getitem__", &Training_Data::operator [],
             return_internal_reference<1>())
#endif
        .def("get", &Training_Data::get)
        .def("share", &Training_Data::share)
        .def("modify", &Training_Data::modify,
             return_internal_reference<1>())
        .def("partition", &Training_Data::partition,
             &Training_Data_Wrapper::def_partition)
        .def("partition", &Training_Data_Wrapper::partition1)
        .def("partition", &Training_Data_Wrapper::partition2)
        .def("add", &Training_Data::add)
        .def("add_example", &Training_Data::add_example)
        .def("make_copy", &Training_Data::make_copy,
             &Training_Data_Wrapper::def_make_copy,
             return_value_policy<manage_new_object>())
        .def("make_type", &Training_Data::make_type,
             &Training_Data_Wrapper::def_make_type,
             return_value_policy<manage_new_object>())
        .def("label_count", &Training_Data::label_count)
        .def("index", &Training_Data::index,
             return_internal_reference<1>())
        .def_readwrite("data_", &Training_Data_Wrapper::data_)
        .def_readwrite("index_", &Training_Data_Wrapper::index_)
        .def_readwrite("feature_space_", &Training_Data_Wrapper::feature_space_)
        .def_readwrite("dirty_", &Training_Data_Wrapper::dirty_)
        .def("generate_index", &Training_Data_Wrapper::generate_index);
}

} // namespace Python
} // namespace ML
