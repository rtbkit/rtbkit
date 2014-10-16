/* classifier_mlp.cc
   Jeremy Barnes, 2 September 2005
   Copyright (c) 2005 Jeremy Barnes  All rights reserved.
   $Source$

   Implementation of classifier to MLP adaptor.
*/

#include "config.h"
#include "classifier_mlp.h"
#include "jml/utils/file_functions.h"
#include "dense_features.h"
#include "mlp.h"
#include "jml/mlp/mlp.h"
#include "jml/arch/exception.h"


using namespace std;

namespace ML {


/*****************************************************************************/
/* CLASSIFIER_MLP                                                            */
/*****************************************************************************/

Classifier_MLP::Classifier_MLP()
{
}

Classifier_MLP::Classifier_MLP(DB::Store_Reader& store)
{
    reconstitute(store);
}

Classifier_MLP::~Classifier_MLP()
{
}

vector<double>
Classifier_MLP::
test(const vector<double>&  fv) const
{
    /* Convert the vector to a feature set */
    Mutable_Feature_Set fs;

    fs.reserve(fv.size());

    for (unsigned i = 0;  i < fv.size();  ++i)
        fs.add(Feature(i + 1), fv[i]);
    
    distribution<float> output = classifier.predict(fs);

    //cerr << "returning " << output << endl;

    vector<double> result(output.begin(), output.end());
    return result;
}

void
Classifier_MLP::
read(const string & input_file)
{
    if (input_file.rfind(".mlp") == input_file.size() - 4) {

        Dense_Feature_Space feature_space;
        
        /* Load the classifier and guess a feature space from it. */
        dg::MlpNetwork mlp;
        mlp.read(input_file);
        
        size_t nfv = mlp.fvSize() + 1;  // +1 for label feature

        vector<string> names(nfv);
        names[0] = "LABEL";
        for (int i = 0;  i < nfv - 1;  ++i)
            names[i + 1] = format("FEAT%d", i);
        
        /* Feature types are all REAL, except for the label which is a
           classification label. */
        vector<Mutable_Feature_Info> info(nfv);
        for (unsigned i = 0;  i < nfv;  ++i)
            info[i].set_type(REAL);
        
        /* How many labels do we have? */
        size_t nl = mlp.nbLabels();

        /* Set up the feature info for the labels. */
        vector<string> label_names(nl);
        for (unsigned i = 0;  i < label_names.size();  ++i)
            label_names[i] = format("%d", i);
        
        info[0].set_type(CATEGORICAL);
        info[0].set_categorical(new Mutable_Categorical_Info(label_names));
        
        /* Initialize the feature space. */
        feature_space.init(names, info);

        bool decode = true;
    
        classifier.impl.reset
            (new MLP(input_file, decode, make_unowned_sp(feature_space),
                     Feature(0, 0, 0)));
    }
    else classifier.load(input_file);
}

void
Classifier_MLP::
reconstitute(DB::Store_Reader & store)
{
    classifier.reconstitute(store);
}

bool
Classifier_MLP::
valid() const
{
    return classifier;
}

vector<double>
Classifier_MLP::
compute(const vector<double> & fv) const
{
    throw Exception("Classifier_MLP::compute(): not implemented");
}

} // namespace ML
