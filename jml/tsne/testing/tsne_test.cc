/* tsne_test.cc
   Jeremy Barnes, 16 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Unit tests for the tsne software.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#undef NDEBUG

#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include <boost/tuple/tuple.hpp>
#include "jml/tsne/tsne.h"
#include <boost/assign/list_of.hpp>
#include <limits>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include "jml/utils/parse_context.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/environment.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

template<typename X>
X sqr(X x)
{
    return x * x;
}

BOOST_AUTO_TEST_CASE( test_vectors_to_distances )
{
    distribution<float> vecs[4] = {
        boost::assign::list_of<float>(1.0)(0.0)(-1.0)(0.0),
        boost::assign::list_of<float>(1.0)(1.0)(-1.0)(0.0),
        boost::assign::list_of<float>(-1.0)(2.0)(-2.0)(0.0),
        boost::assign::list_of<float>(1.0)(0.0)(-1.0)(4.0) };

    boost::multi_array<float, 2> vectors(boost::extents[4][4]);

    for (unsigned i = 0;  i < 4;  ++i)
        for (unsigned j = 0;  j < 4;  ++j) 
            vectors[i][j] = vecs[i].at(j);

    boost::multi_array<float, 2> distances
        = vectors_to_distances(vectors);

    double tolerance = 0.00001;

    for (unsigned i = 0;  i < 4;  ++i)
        for (unsigned j = 0;  j < 4;  ++j) 
            BOOST_CHECK_CLOSE(distances[i][j],
                              sqr(float((vecs[i] - vecs[j]).two_norm())),
                              tolerance);
}

BOOST_AUTO_TEST_CASE( test_perplexity_and_prob1 )
{
    static const double D [100] = {
        0.0,
        127.3646507617367, 200.79174789449758, 132.8431517841056,
        168.6695651092603, 149.83575026049175, 119.7804868444637,
        93.771630810081646, 127.87258643700642, 150.69475590453624,
        92.18950360869438, 194.7395450530293, 138.47392145410225,
        153.44382232639336, 124.75249546141032, 140.07600846946599,
        142.52422684695858, 112.53458818071468, 132.07673249366951,
        121.71444524900835, 230.76039706024162, 153.65195167028637,
        135.90623456637366, 141.46038866388847, 127.82008231999832,
        158.14226945533827, 139.16544230277188, 99.135028505834867,
        162.53574221087504, 155.54663216867959, 145.94914816004081,
        148.45282641646196, 138.28218261911837, 125.73692561668645,
        115.48238689570317, 145.41286371552036, 112.96762686302431,
        148.75725280363577, 171.79874154730805, 129.54100995517584,
        123.25696994589572, 97.417040308677002, 133.12667636291448,
        109.60150219382513, 124.55810949534977, 151.93098797851763,
        139.28486178870102, 137.81379575624848, 105.21261037047898,
        67.866856537412559, 118.63097885862219, 168.1890582000824,
        162.80820294133918, 148.42005954080824, 165.34621317818949,
        133.53390797717651, 204.34705568092269, 137.01483466564827,
        164.03120562137545, 142.33535643454547, 227.00761351009086,
        150.63686261349545, 105.9350761339563, 182.72297366429197,
        179.04700475611409, 194.59561522319746, 132.09520418925339,
        118.26178243062299, 201.85218056147639, 141.56241365805982,
        92.523368308752453, 156.57653221994249, 126.91063908664233,
        150.01792311135824, 79.968520873937621, 144.47464729743518,
        158.83234647816369, 127.11676853771658, 128.6179425062,
        146.47879679124785, 154.0507546968862, 112.05567878135535,
        170.68966818333143, 159.39572118149405, 144.86151660521938,
        100.14835072139124, 176.13397580871691, 150.15085498122815,
        188.92645325418638, 174.99387584699798, 157.00819864592739,
        157.34145157437979, 151.79701646949877, 132.40646769939468,
        99.790915408957133, 121.71261355515941, 131.10599107100731,
        97.803262730518327, 81.995893718966656, 123.95079073664236 };
    
    double H = 3.28491385;

    double betai = 0.0625;

    static const double P[100] = {
        0.0,
        0.0052076674027168356,
        5.2915364889354196e-05, 0.0036977544359491792, 0.00039399182973246058,
        0.0012785010300623912, 0.0083657190067658489, 0.042508148504905856,
        0.0050449415115928573, 0.0012116710848464084, 0.046926321177611842,
        7.7243009971482806e-05, 0.0026007571699126321, 0.0010203888879298211,
        0.0061312077070300444, 0.0023529554666564985, 0.0020191134617900336,
        0.013157750168155269, 0.0038791921836005059, 0.0074132570055467622,
        8.1307500422764546e-06, 0.0010072015405830908, 0.0030534834458931293,
        0.0021579278255610384, 0.0050615237164899915, 0.00076073582124040978,
        0.0024907467591474101, 0.030401241034895998, 0.0005780694999155823,
        0.00089472238083898722, 0.0016300341291158632, 0.0013939215887890984,
        0.0026321112979258493, 0.0057653442265850532, 0.010943804879502899,
        0.0016855951909598022, 0.012806412650215196, 0.0013676506482255075,
        0.00032400425388644667, 0.0045453720820282529, 0.0067319314333328375,
        0.033847247905790211, 0.0036328063182662685, 0.015805039195314789,
        0.0062061508276432002, 0.0011215772617581947, 0.0024722257318904426,
        0.0027103031116231672, 0.020793377873604087, 0.2145932077868026,
        0.00898886450492308, 0.00040600353001692433, 0.00056830901305673508,
        0.0013967791673110547, 0.00048494702979492846, 0.0035415109713178208,
        4.2371968004920509e-05, 0.0028490783217706693, 0.00052648752672150622,
        0.0020430891170220484, 1.0280019401301456e-05, 0.0012160632528851881,
        0.019875353654866756, 0.00016369290429846074, 0.00020597199118722314,
        7.794099298777404e-05, 0.0038747163141802515, 0.0091986920989935179,
        4.952198471339618e-05, 0.0021442114428565353, 0.045957276510967107,
        0.00083894464584652296, 0.0053575552973421673, 0.0012640268255567047,
        0.10072460989397887, 0.0017873914323957886, 0.00072862291645688039,
        0.0052889761308962131, 0.0048153142411132017, 0.0015769583760696156,
        0.00098240713789262573, 0.013557539169600952, 0.00034725998309463509,
        0.00070341385557516503, 0.0017446918065241767, 0.028535553260042926,
        0.00024710281464630741, 0.0012535684903119257, 0.00011108266604827809,
        0.00026535293000577465, 0.00081661322693345119, 0.00079978046262873235,
        0.0011310079014562836, 0.0038000661766847675, 0.0291802029929942,
        0.007414105730205808, 0.0041218342507203607, 0.03303999482592785,
        0.088737214435108819, 0.0064462482291298574 };

    distribution<double> Dv(D, D + (sizeof(D) / sizeof(D[0])));
    distribution<double> Pv(P, P + (sizeof(P) / sizeof(P[0])));

    distribution<double> resP;
    double resH;

    boost::tie(resH, resP)
        = perplexity_and_prob(Dv, betai, 0);

    double tolerance = 0.000001;

    BOOST_CHECK_CLOSE(resP.total(), 1.0, tolerance);
    BOOST_CHECK_CLOSE(resH, H, tolerance);

    for (unsigned i = 0;  i < 100;  ++i)
        BOOST_CHECK_CLOSE(resP[i], P[i], tolerance);

    //cerr << "Dv = " << Dv << endl;
    //cerr << "Pv = " << Pv << endl;
    //cerr << "resP = " << resP << endl;
    //cerr << "H = " << H << endl;
    //cerr << "resH = " << resH << endl;
}

BOOST_AUTO_TEST_CASE( test_small )
{
    string input_file = Environment::instance()["JML_TOP"]
        + "/tsne/testing/mnist2500_X_min.txt.gz";

    filter_istream stream(input_file);
    Parse_Context context(input_file, stream);

    int nd = 784;
    int nx = 100;

    boost::multi_array<float, 2> data(boost::extents[nx][nd]);

    cerr << "loading " << nx << " examples...";
    for (unsigned i = 0;  i < nx;  ++i) {
        for (unsigned j = 0;  j < nd;  ++j) {
            float f = context.expect_float();
            data[i][j] = f;
            context.expect_whitespace();
        }

        context.expect_eol();
    }
    cerr << "done." << endl;

    cerr << "converting to distances...";
    boost::multi_array<float, 2> distances
        = vectors_to_distances(data);
    cerr << "done." << endl;

    cerr << "converting to probabilities...";
    boost::multi_array<float, 2> probabilities
        = distances_to_probabilities(distances,
                                     1e-5 /* tolerance */,
                                     20.0 /* perplexity */);
    cerr << "done." << endl;

    // Obtained from the tsne.py program with bugs fixed
    float expected[100] = {
        0.00000000e+00, 2.68792541e-03, 2.43188624e-05, 2.30939754e-03,
        2.03611574e-04, 7.98121656e-04, 6.19396925e-03, 4.13000407e-02,
        4.23811520e-03, 6.85725943e-04, 6.03596575e-02, 2.62362954e-05,
        1.83916599e-03, 5.06189988e-04, 4.57227155e-03, 1.25841718e-03,
        1.46468136e-03, 9.76617696e-03, 3.37516482e-03, 5.32170173e-03,
        2.14410978e-06, 5.89158388e-04, 1.98417596e-03, 1.35763772e-03,
        4.57227155e-03, 3.73659924e-04, 1.58016490e-03, 3.54839505e-02,
        3.73659924e-04, 4.34905636e-04, 9.28939884e-04, 8.61049963e-04,
        1.46468136e-03, 4.23811520e-03, 8.39085226e-03, 1.25841718e-03,
        9.05243326e-03, 6.85725943e-04, 1.74937915e-04, 3.12849689e-03,
        5.74129401e-03, 3.54839505e-02, 2.49148316e-03, 1.42731844e-02,
        4.57227155e-03, 6.35610880e-04, 2.49148316e-03, 2.14061932e-03,
        2.61935846e-02, 2.55254059e-01, 7.77762173e-03, 1.74937915e-04,
        2.55670272e-04, 8.61049963e-04, 2.55670272e-04, 2.89985626e-03,
        1.22831552e-05, 1.35763772e-03, 2.75828725e-04, 1.25841718e-03,
        2.49554573e-06, 6.35610880e-04, 1.79225024e-02, 8.19014090e-05,
        8.83589672e-05, 3.05366244e-05, 2.30939754e-03, 5.32170173e-03,
        1.79517266e-05, 1.25841718e-03, 4.13000407e-02, 4.34905636e-04,
        4.23811520e-03, 5.46100794e-04, 1.19503351e-01, 1.25841718e-03,
        4.69195992e-04, 3.12849689e-03, 2.89985626e-03, 8.61049963e-04,
        6.35610880e-04, 9.76617696e-03, 2.03611574e-04, 3.46351652e-04,
        1.16644800e-03, 2.61935846e-02, 1.39317661e-04, 7.39792353e-04,
        3.29443000e-05, 1.19698211e-04, 4.03121330e-04, 3.73659924e-04,
        5.46100794e-04, 2.14061932e-03, 2.61935846e-02, 4.93277462e-03,
        2.49148316e-03, 3.28906690e-02, 9.51704912e-02, 4.93277462e-03 };

    double tolerance = 1e-5;

    for (unsigned i = 0;  i < 100;  ++i)
        BOOST_CHECK_CLOSE(probabilities[0][i], expected[i], tolerance);

    boost::multi_array<float, 2> reduction JML_UNUSED
        = tsne(probabilities, 2);
}

BOOST_AUTO_TEST_CASE( test_distance_to_probability_big )
{
    string input_file = Environment::instance()["JML_TOP"]
        + "/tsne/testing/mnist2500_X_min.txt.gz";

    filter_istream stream(input_file);
    Parse_Context context(input_file, stream);

    int nd = 784;
    int nx = 2500;

    boost::multi_array<float, 2> data(boost::extents[nx][nd]);

    cerr << "loading " << nx << " examples...";
    for (unsigned i = 0;  i < nx;  ++i) {
        for (unsigned j = 0;  j < nd;  ++j) {
            float f = context.expect_float();
            data[i][j] = f;
            context.expect_whitespace();
        }

        context.expect_eol();
    }
    cerr << "done." << endl;

    // Step 1: perform a dimensionality reduction via a SVD on the data
    //cerr << "performing SVD...";
    //boost::multi_array<float, 2> data_reduced
    //    = pca(data, 50);
    //cerr << "done." << endl;

    cerr << "converting to distances...";
    boost::multi_array<float, 2> distances
        = vectors_to_distances(data);
    cerr << "done." << endl;

    cerr << "converting to probabilities...";
    boost::multi_array<float, 2> probabilities JML_UNUSED
        = distances_to_probabilities(distances);
    cerr << "done." << endl;

    boost::multi_array<float, 2> reduction JML_UNUSED
        = tsne(probabilities, 2);
}
