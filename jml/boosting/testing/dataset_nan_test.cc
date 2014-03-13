/* dataset_nan_test.cc
   Francois Maillet, 16 October 2012
   Copyright (c) 2012 Jeremy Barnes.  All rights reserved.

   Test that the Dataset can parse a dataset with NaN and nan
*/

#include "jml/boosting/tools/datasets.h"
#include <iostream>

using namespace std;
using namespace ML;

int main(int argc, char ** argv)
{

// Create dataset file
string filename = "/tmp/jml_temp_dataset.txt";
ofstream myfile;
myfile.open(filename);
myfile << "LABEL min_price-AbsDist-AvgAgg:k=REAL/o=OPTIONAL\n";
myfile << "0 nan\n";
myfile << "0 2.0\n";
myfile << "0 1.0\n";
myfile << "0 1.0\n";
myfile << "1 3.0\n";
myfile << "1 Nan\n";
myfile << "0 NaN\n";
myfile << "0 NaN\n";
myfile << "1 3.0\n";
myfile.close();

vector<string> params(1);
params[0] = filename;

Datasets datasets;
datasets.init(params, 1, 0);

}
