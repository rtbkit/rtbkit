/* boosting_training_tool.cc                                       -*- C++ -*-
   Jeremy Barnes, 12 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Tool to use to train boosting with.  Very similar in spirit to the MLP
   training tool.
*/

#include "jml/boosting/classifier.h"
#include "jml/utils/command_line.h"
#include "jml/utils/file_functions.h"

#include <iterator>
#include <iostream>
#include <set>


using namespace std;

using namespace ML;


int main(int argc, char ** argv)
try
{
    ios::sync_with_stdio(false);

    string classifier_in;
    string classifier_out;
    {
        using namespace CmdLine;

        static const Option classifier_options[] = {
            { "classifier-in", 'i', classifier_in, classifier_in,
              false, "load classifier from FILE", "FILE" },
            { "classifier-out", 'o', classifier_out, classifier_out,
              false, "save classifier to FILE", "FILE" },
            Last_Option
        };

        static const Option options[] = {
            group("Classifier options", classifier_options),
            Help_Options,
            Last_Option };

        Command_Line_Parser parser("classifier_tool", argc, argv,
                                   options);
        
        bool res = parser.parse();
        if (res == false) exit(1);
        
        if (parser.extra_begin() != parser.extra_end()) {
            parser.extra_options_error();
            exit(1);
        }
    }

    /* Load the classifier. */
    if (classifier_in == "")
        throw Exception("Need to specify a classifier to test.");

    Classifier classifier;
    classifier.load(classifier_in);

    if (classifier_out != "")
        classifier.save(classifier_out);
}
catch (const std::exception & exc) {
    cerr << "error: " << exc.what() << endl;
    exit(1);
}
