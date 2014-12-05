/* exchange_parsing_from_file.h
   Jean-Sebastien Bejeau, 27 May 2014
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Allow to test batch of Bid Request parsing from a file.
*/

struct Exchange_parsing_from_file {

    Exchange_parsing_from_file(const std::string config);

    void run();
    int getNumError();
    void resetNumError();

    std::string header;

private :
    std::string configurationFile;
    int error;

};




