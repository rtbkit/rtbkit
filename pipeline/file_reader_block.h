/* file_reader_block.h
   Eric Robert, 10 January 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

namespace Datacratic
{
    struct TextLine {
        TextLine() : number(0), offset(0) {
        }

        std::string text;
        unsigned number;
        unsigned offset;
    };

    CREATE_STRUCTURE_DESCRIPTION(TextLine)

    struct FileReaderBlock :
        public Block
    {
        FileReaderBlock();

        void run();

        PushingPin<TextLine> lines;
        std::string folder;
        std::string filename;
    };
}

