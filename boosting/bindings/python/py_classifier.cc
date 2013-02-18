/* py_classifier.cc
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

   Bindings for python into the classifier.
*/

#include <boost/python.hpp>
#include "jml/boosting/classifier.h"
using namespace boost::python;
using namespace ML;

namespace ML {
namespace Python {

struct Classifier_Impl_Wrapper : Classifier_Impl, wrapper<Classifier_Impl> {

    /* Virtual function predict */

    virtual float predict(int label, const Feature_Set & features) const
    {
        if (override ovr = this->get_override("predict"))
            return ovr(label, features);
        return Classifier_Impl::predict(label, features);
    }

    float def_predict(int label, const Feature_Set & features) const
    {
        return Classifier_Impl::predict(label, features);
    }

    /* Virtual function predict_highest */

    virtual float predict_highest(const Feature_Set & features) const
    {
        if (override ovr = this->get_override("predict_highest"))
            return ovr(features);
        return Classifier_Impl::predict_highest(features);
    }
    
    float def_predict_highest(const Feature_Set & features) const
    {
        return Classifier_Impl::predict_highest(features);
    }
    
    /* Pure virtual function predict */

    virtual distribution<float>
    predict(const Feature_Set & features) const
    {
        return this->get_override("predict")(features);
    }

    /* Virtual function accuracy */

    virtual float accuracy(const Training_Data & data,
                           const distribution<float> & example_weights
                               = UNIFORM_WEIGHTS) const
    {
        if (override ovr = this->get_override("accuracy"))
            return ovr(data, example_weights);
        return Classifier_Impl::accuracy(data, example_weights);
    }

    float def_accuracy(const Training_Data & data,
                       const distribution<float> & example_weights
                           = UNIFORM_WEIGHTS) const
    {
        return Classifier_Impl::accuracy(data, example_weights);
    }

    float accuracyA1(const Training_Data & data) const
    {
        return accuracy(data);
    }

    /* Pure virtual function train */

    virtual float train(const Training_Data & data,
                        const Training_Params & params)
    {
        return this->get_override("train")(data, params);
    }

    /* Pure virtual function train_weighted */

    virtual float
    train_weighted(const Training_Data & data,
                   const Training_Params & params,
                   const fixed_array<float, 2> & weights)
    {
        override ovr = this->get_override("train_weighted");
        return ovr(data, params, weights);
    }
    
    /* Pure virtual function all_features */

    virtual std::vector<Feature> all_features() const
    {
        return this->get_override("all_features")();
    }

    /* Pure virtual function print */

    virtual std::string print() const
    {
        return this->get_override("print")();
    }

    /* Pure virtual function class_id */

    virtual std::string class_id() const
    {
        return this->get_override("class_id")();
    }

    /* Pure virtual function serialize */

    virtual void serialize(DB::Store_Writer & store) const
    {
        this->get_override("serialize")(store);
    }

    /* Pure virtual function reconstitute */

    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space)
    {
        this->get_override("reconstitute")(store, feature_space);
    }

    /* Pure virtual function make_copy */

    virtual Classifier_Impl * make_copy() const
    {
        return this->get_override("make_copy")();
    }

    /* Protected function swap */

    using Classifier_Impl::swap;
};

void export_classifier()
{

    float (Classifier_Impl::* predictA)
        (int, const Feature_Set &) const = &Classifier_Impl::predict;
    distribution<float> (Classifier_Impl::* predictB)
        (const Feature_Set &) const = &Classifier_Impl::predict;
    float (Classifier_Impl::* accuracyA)
        (const Training_Data &, const distribution<float> &) const
        = &Classifier_Impl::accuracy;

    float (* accuracyB) (const std::vector<distribution<float> > &,
                         const Training_Data &, const Feature &,
                         const distribution<float> &)
        = &Classifier_Impl::accuracy;
    float (* accuracyC) (const fixed_array<float, 2> &,
                         const Training_Data &,
                         const Feature &,
                         const distribution<float> &)
        = &Classifier_Impl::accuracy;

    void (Classifier_Impl::* initA)
        (const std::shared_ptr<const Feature_Space> &,
         const Feature &)
        = &Classifier_Impl::init;
    void (Classifier_Impl::* initB)
        (const std::shared_ptr<const Feature_Space> &,
         const Feature &, size_t)
        = &Classifier_Impl::init;

    class_<Classifier_Impl_Wrapper, boost::noncopyable>("Classifier")
        .def("label_count", &Classifier_Impl::label_count)
        .def("feature_space", &Classifier_Impl::feature_space)
        .def("predicted", &Classifier_Impl::predicted,
             return_internal_reference<1>())
        .def("set_feature_space", &Classifier_Impl::set_feature_space)
        .def("predict", predictA, &Classifier_Impl_Wrapper::def_predict)
        .def("predict_highest", &Classifier_Impl::predict_highest,
             &Classifier_Impl_Wrapper::def_predict_highest)
        .def("predict", pure_virtual(predictB))
        .def("accuracy", accuracyA, &Classifier_Impl_Wrapper::def_accuracy)
        .def("accuracy", &Classifier_Impl_Wrapper::accuracyA1)
        .def("accuracy", accuracyB)
        .def("accuracy", accuracyC)
        .def("train", pure_virtual(&Classifier_Impl::train))
        .def("train_weighted", &Classifier_Impl::train_weighted,
             &Classifier_Impl_Wrapper::train_weighted)
        .def("all_features", pure_virtual(&Classifier_Impl::all_features))
        .def("print", pure_virtual(&Classifier_Impl::print))
        .def("class_id", pure_virtual(&Classifier_Impl::class_id))
        .def("serialize", pure_virtual(&Classifier_Impl::serialize))
        .def("reconstitute", pure_virtual(&Classifier_Impl::reconstitute))
        .def("init", initA)
        .def("init", initB)
        .def("swap", &Classifier_Impl_Wrapper::swap);
}

} // namespace Python
} // namespace ML
