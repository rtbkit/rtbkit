/* file_writer_block.h
   Eric Robert, 5 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

namespace Datacratic
{
    struct FileWriterBlock :
        public Block
    {
        FileWriterBlock();

        void run();

        PullingPin<std::string> lines;
        std::string filename;
        std::string folder;

    private:
        ML::filter_ostream file;
        int count;
        int bytes;
    };
}

