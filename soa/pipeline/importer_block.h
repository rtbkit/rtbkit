/* importer_block.h
   Eric Robert, 14 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

namespace Datacratic
{
    struct ImporterBlock :
        public Block
    {
        ImporterBlock();

        void run();

        virtual void onRead(TextLine const & line);
        virtual void onDone();

        PullingPin<TextLine> lines;

    private:
        Logging::Progress progress;
        int done;
    };
}

