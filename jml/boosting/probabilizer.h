/* probabilizer.h                                                  -*- C++ -*-
   Jeremy Barnes, 13 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

*/

#ifndef __boosting__probabilizer_h__
#define __boosting__probabilizer_h__


#include "jml/db/persistent.h"
#include "jml/stats/distribution.h"
#include "decoder.h"
#include <boost/multi_array.hpp>
#include "jml/algebra/irls.h"


namespace ML {

class Training_Data;
class Classifier;
class Classifier_Impl;


/*****************************************************************************/
/* GLZ_PROBABILIZER                                                          */
/*****************************************************************************/

/** A probabilizer, implemented using a GLZ.  It learns a function of the
    form

    \f[
        p = \mathrm{logit}(a_0 x_0 + a_1 x_1 + ... + c)
    \f]

    which is learnt via a least squares procedure in the GLZ.
*/

class GLZ_Probabilizer : public Decoder_Impl {
public:    
    GLZ_Probabilizer();
    virtual ~GLZ_Probabilizer();

    virtual distribution<float>
    apply(const distribution<float> & input) const;

    void train(const Training_Data & data,
               const Classifier_Impl & classifier,
               const Optimization_Info & opt_info,
               int mode, const std::string & link_name);

    void train(const Training_Data & data,
               const Classifier_Impl & classifier,
               const Optimization_Info & opt_info,
               const distribution<float> & weights,
               int mode, const std::string & link_name);

    /** Initialize directly from parameters. */
    void init(const Classifier_Impl & classifier,
              int mode, const std::string & link_name,
              const std::vector<float> & params);

    virtual size_t domain() const;
    virtual size_t range() const;

    virtual Output_Encoding output_encoding(Output_Encoding input) const;

    virtual GLZ_Probabilizer * make_copy() const;

    /** Construct a probabilizer of type 2 (parameterized sparsely) given
        the output of a GLZ (3 values) and the number of labels for which we
        want this to apply.

        \param params       The parameters for the parameterizer.  These are
                            normally the result of the irls() (iteratively
                            restarted least squares) routine.  There should
                            be 3: the first for the label itself, the second
                            for the maximum over all labels, and the third
                            for the constant 1.0.
        \param label_count  The number of labels to construct the
                            parameterization for.  The 3 values in params will
                            be expanded into a nl x nl + 2 matrix.  The left
                            nl x nl square will be params[0] multiplied by the
                            identify matrix; the (nl + 1)st column will be
                            params[1], and the (nl + 2)nd params[2].

        \returns            The trained sparse probabilizer, for the given
                            number of labels.
    */
    static GLZ_Probabilizer
    construct_sparse(const distribution<double> & params,
                     size_t label_count, Link_Function link = LOGIT);

    /** Add some data to a dataset, ready to train a sparse probabilizer.  This
        one adds a single example. */
    static void add_data_sparse(std::vector<distribution<double> > & data,
                                const std::vector<float> & output,
                                int correct_label);

    /** Add some data to a dataset, ready to train a sparse probabilizer. 
        This one adds the result over the entire training data. */
    static void add_data_sparse(std::vector<distribution<double> > & data,
                                const Training_Data & training_data,
                                const Classifier_Impl & classifier,
                                const Optimization_Info & opt_info);

    /** Train the parameters for a sparse probabilizer.  Returns values which
        can be passed to construct_sparse to perform the actual construction.

        \param data         Training data built up by the add_data_sparse()
                            method.

        \param weights      Weight vector.  If empty, a uniform vector is
                            assumed.
    */
    static distribution<double>
    train_sparse(const std::vector<distribution<double> > & data,
                 Link_Function link = LOGIT,
                 const distribution<double> & weights
                     = distribution<double>());

    /** Interrogate whether it is positive or not.  This checks to make sure
        that the diagonals of the matrix has only positive entries.  If
        the result of this function is false, that means that a variable is
        negatively correlated (the higher the value, the less likely).  In
        some applications, this is a problem.
    */
    bool positive() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);
    GLZ_Probabilizer(DB::Store_Reader & store);

    virtual std::string class_id() const { return "GLZ_PROBABILIZER"; }

    /** Print a human-readable representation to the given string. */
    std::string print() const;

    /** Parameters of the fitted model. */
    std::vector<distribution<float> > params;  // constant + one for each class

    /** The link function to apply to the output. */
    Link_Function link;

    //private: // public for testing
    /** Training function for mode 0.  This learns a probabilizer over all
        outputs for all variables (very dense).  It requires a lot of
        training data, but is able to account for the relationship between
        different variables (eg, if output 3 is low, then the probability of
        class 2 being correct is also low).  Each element in the transformation
        matrix can vary independently.

        \param outputs      The output of the classifier over all of the
                            examples in the training data, with two extra
                            columns: the maximum of the row, and a constant
                            1.0 term.  Dimensions are (nl + 2) x nx.
        \param correct      A vector of distributions showing for which
                            examples label l was correct.  Dimensions are
                            [nl][nx].
        \param num_correct  A distribution of size nl giving the total number
                            of correct examples in each column.
        \param weights      The weight vector, of size nx.
        \param debug        True if we are operating in debug mode.  More
                            information will be printed in this case.
    */
    void train_mode0(boost::multi_array<double, 2> & outputs,
                     const std::vector<distribution<double> > & correct,
                     const distribution<int> & num_correct,
                     const distribution<float> & weights,
                     bool debug);

    /** Training function for mode 1.  This learns parameters individually
        for each output, based only upon its value, the maximum value, and
        the bias term.  The transformation matrix is diagonal, and the max
        and bias columns can vary independently.

        \verbatim
        x0 0 0 ... 0 y0 z0
        0 x1 0 ... 0 y1 z1
        0 0 x2 ... 0 y2 z2
        : : :      : : :
        0 0 0 ... xn yn zn
        \endverbatim


        \copydoc train_mode0
    */
    void train_mode1(const boost::multi_array<double, 2> & outputs,
                     const std::vector<distribution<double> > & correct,
                     const distribution<int> & num_correct,
                     const distribution<float> & weights,
                     bool debug);

    /** Training function for mode 2.  This learns 3 parameters: one for the
        output, one for the max column, and one for the bias column.  This
        is a very sparse representation, suitable for when there is not much
        data and the outputs are roughly distributed the same.

        \verbatim
        x 0 0 ... 0 y z
        0 x 0 ... 0 y z
        0 0 x ... 0 y z
        : : :     : : :
        0 0 0 ... x y z
        \endverbatim

        \copydoc train_mode0
    */
    void train_mode2(const boost::multi_array<double, 2> & outputs,
                     const std::vector<distribution<double> > & correct,
                     const distribution<int> & num_correct,
                     const distribution<float> & weights,
                     bool debug);

    /** Training function for mode 3.  This mode doesn't train at all; it
        applies the output transformation directly to the input.  Its
        transformation matrix looks like:

        \verbatim
        1 0 0 ... 0 0 0
        0 1 0 ... 0 0 0
        0 0 1 ... 0 0 0
        : : :     : : :
        0 0 0 ... 1 0 0
        \endverbatim

        \copydoc train_mode0
    */
    void train_mode3(const boost::multi_array<double, 2> & outputs,
                     const std::vector<distribution<double> > & correct,
                     const distribution<int> & num_correct,
                     const distribution<float> & weights,
                     bool debug);

    /** Training function for mode 4.  This mode is a combination of
        modes 0, 1 and 2.  It will use a different mode for each of
        the variables, depending upon the number of correct examples
        it has for each one (mode 0 for > 100 samples, mode 1 for > 50
        samples, mode 2 otherwise).  It can be used to get the
        "best of both worlds".

        \copydoc train_mode0
    */
    void train_mode4(const boost::multi_array<double, 2> & outputs,
                     const std::vector<distribution<double> > & correct,
                     const distribution<int> & num_correct,
                     const distribution<float> & weights,
                     bool debug);

    /** Training function for mode 5.  This mode is like mode 2 but for
        binary classifiers only.  It learns only one label and infers
        the other with 1 - thresh(prob).

        \copydoc train_mode0
    */
    void train_mode5(const boost::multi_array<double, 2> & outputs,
                     const std::vector<distribution<double> > & correct,
                     const distribution<int> & num_correct,
                     const distribution<float> & weights,
                     bool debug);

    /** Function to train a single classifier for mode 0.
        \param outputs      Outputs matrix.  Needs to already have had its
                            dependent columns removed.
        \param correct      Correctness vector, for the one we are doing.
        \param w            Weights vector.
        \param dest         Destination vector, giving which column number
                            really represents each label.

        Returns a vector with the parameters for the given label.
    */
    distribution<float>
    train_one_mode0(const boost::multi_array<double, 2> & outputs,
                    const distribution<double> & correct,
                    const distribution<double> & w,
                    const std::vector<int> & dest) const;

    /** Function to train a single classifier for mode 1.
        \param outputs      Outputs matrix.  We need this to have not been
                            altered (ie, no columns changed).
        \param correct      Correctness vector, for the one we are doing.
        \param w            Weights vector.
        \param l            The label number.  Needed to know where to put
                            the results.

        Returns a vector with the parameters for the given label.
    */
    distribution<float>
    train_one_mode1(const boost::multi_array<double, 2> & outputs,
                    const distribution<double> & correct,
                    const distribution<double> & w,
                    int l) const;

    /** Training function for regression.  This learns a probabilizer with
        two variables: one for the output, and one for the bias.

        \param outputs      The output of the classifier over all of the
                            examples in the training data, with one extra
                            column: a constant 1.0 term.  Dimensions are
                            2 x nx.
        \param correct      The target distribution, of length nx.  These
                            are the values we are trying to fit.
        \param weights      The weight vector, of size nx.
        \param debug        True if we are operating in debug mode.  More
                            information will be printed in this case.
    */
    void
    train_glz_regress(const boost::multi_array<double, 2> & outputs,
                      const distribution<double> & correct,
                      const distribution<float> & weights,
                      bool debug);

    /** Train an identity function for regression mode.
        \copydoc train_glz_regress
    */
    void
    train_identity_regress(const boost::multi_array<double, 2> & outputs,
                           const distribution<double> & correct,
                           const distribution<float> & weights,
                           bool debug);
};

/** Free serialization operator. */
DB::Store_Writer &
operator << (DB::Store_Writer & store, const GLZ_Probabilizer & prob);

/** Free reconstitution operator. */
DB::Store_Reader &
operator >> (DB::Store_Reader & store, GLZ_Probabilizer & prob);


} // namespace ML



#endif /* __boosting__probabilizer_h__ */

