#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <vector>
#include <iostream>

#include "jml/boosting/classifier.h"
#include "jml/boosting/dense_features.h"
#include "jml/stats/distribution.h"

using namespace ML;
using namespace std;

float rand_prob() {
    return float(rand()) / float(RAND_MAX);
}

BOOST_AUTO_TEST_CASE( weighted_training_test )
{
    string filename1 = "weighted_training_test_dataset1.txt";
    string filename2 = "weighted_training_test_dataset2.txt";
    string modelname1 = "weighted_training_test_model1.cls";
    string modelname2 = "weighted_training_test_model2.cls";
    string modelname3 = "weighted_training_test_model3.cls";
    string configname = "weighted_training_test_config.cls";
    
    ofstream filestream;
    filestream.open(filename1);

    filestream << "LABEL:k=BOOLEAN ExampleId:k=STRING/o=BIASED dim1:k=REAL dim2:k=REAL" << endl;
    filestream << "0 a 0.000000 0.000000" << endl;
    filestream << "0 b 0.100000 0.100000" << endl;
    filestream << "0 c 0.100000 0.400000" << endl;
    filestream << "0 d 0.400000 0.100000" << endl;
    filestream << "1 e 0.500000 0.500000" << endl;
    filestream << "1 u 1.000000 1.000000" << endl;
    filestream << "1 v 0.900000 0.600000" << endl;
    filestream << "1 x 0.600000 0.800000" << endl;
    filestream << "1 x 0.800000 0.700000" << endl;
    filestream << "1 x 0.900000 0.800000" << endl;

    filestream.close();
    filestream.open(filename2);

    filestream << "LABEL:k=BOOLEAN WEIGHT:k=REAL ExampleId:k=STRING/o=BIASED dim1:k=REAL dim2:k=REAL" << endl;
    filestream << "0 1.000 a 0.000000 0.000000" << endl;
    filestream << "0 1.000 b 0.100000 0.100000" << endl;
    filestream << "0 1.000 c 0.100000 0.400000" << endl;
    filestream << "0 1.000 d 0.400000 0.100000" << endl;
    filestream << "1 1.000 e 0.500000 0.500000" << endl;
    filestream << "1 1.000 u 1.000000 1.000000" << endl;
    filestream << "1 1.000 v 0.900000 0.600000" << endl;
    filestream << "1 1.000 x 0.600000 0.800000" << endl;
    filestream << "1 1.000 x 0.800000 0.700000" << endl;
    filestream << "1 1.000 x 0.900000 0.800000" << endl;

    filestream.close();
    filestream.open(configname);

    filestream << "glz {type=glz;\nverbosity=3;\nnormalize = true;\nridge_regression = true;\n}" << endl;

    filestream.close(); 

    char const * bin = getenv("BIN");
    if(!bin) {
        bin = "build/x86_64/bin";
    }

    string cmd1 = "make classifier_training_tool; " + std::string(bin) + "/classifier_training_tool " + filename1 + " -c " + configname + " -n glz glz.link_function=linear glz.ridge_regression=false -E 1 -p 1 -o " + modelname1 + " glz.verbosity=10  -W LABEL/F,ExampleId/F -z ExampleId";

    string cmd2 = std::string(bin) + "/classifier_training_tool " + filename2 + " -c " + configname + " -n glz glz.link_function=linear glz.ridge_regression=false -E 1 -p 1 -o " + modelname2 + " glz.verbosity=10  -W LABEL/F,ExampleId/F,WEIGHT/V -z ExampleId -z WEIGHT";

    string cmd3 = std::string(bin) + "/classifier_training_tool " + filename2 + " -c " + configname + " -n glz glz.link_function=linear glz.ridge_regression=false -E 1 -p 1 -o " + modelname3 + " glz.verbosity=10  -W LABEL/F,WEIGHT/V -z ExampleId -z WEIGHT";


    cout << cmd1 << endl;
    cout << cmd2 << endl;
    cout << cmd3 << endl;
    
    if ( system(cmd1.c_str()) == -1 )
        throw Exception("Call 1 to classifier_training_tool failed");
    
    if ( system(cmd2.c_str()) == -1 )
        throw Exception("Call 2 to classifier_training_tool failed");

    if ( system(cmd3.c_str()) == -1 )
        throw Exception("Call 2 to classifier_training_tool failed");
    
    Classifier class1;
    Classifier class2;
    Classifier class3;

    class1.load(modelname1);
    class2.load(modelname2);//"build/x86_64/tmp/pclick_predictor_test_model_ew.cls");
    class3.load(modelname3);

    for (int i=0; i<5; ++i){
        vector<float> point = {1.0, 0.0, rand_prob(), rand_prob()};
        distribution<float> point1(point);
        distribution<float> point2(point);
        point2.insert(point2.begin(), 1.0);

        // cerr << "point1: " << point1 << endl;
        float prediction1 = class1.impl->predict(0, *class1.impl->feature_space<Dense_Feature_Space>()->encode(point1));

        // cerr << "point2: " << point2 << endl;
        float prediction2 = class2.impl->predict(0, *class2.impl->feature_space<Dense_Feature_Space>()->encode(point2));

        BOOST_CHECK_EQUAL(prediction1, prediction2);

        float prediction3 = class3.impl->predict(0, *class3.impl->feature_space<Dense_Feature_Space>()->encode(point2));

        BOOST_CHECK_PREDICATE( std::not_equal_to<float>(), (prediction2)(prediction3) ); 
    }
   
    cout << "removing " << filename1 << endl;
    remove(filename1.c_str());
    cout << "removing " << filename2 << endl;
    remove(filename2.c_str());
    cout << "removing " << modelname1 << endl;
    remove(modelname1.c_str());
    cout << "removing " << modelname2 << endl;
    remove(modelname2.c_str());
    cout << "removing " << modelname3 << endl;
    remove(modelname3.c_str());
    cout << "removing " << configname << endl;
    remove(configname.c_str());
}
