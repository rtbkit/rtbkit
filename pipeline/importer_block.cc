/* importer_block.h
   Eric Robert, 14 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

ImporterBlock::ImporterBlock() :
    lines(this, "lines"),
    progress(trace, [&]() {
        LOG(progress) << "imported "
                      << done
                      << " lines in "
                      << progress.secondsSinceStart()
                      << "s at a rate of "
                      << done / progress.secondsSinceStart()
                      << " line/s"
                      << std::endl;
    }),
    done(0) {
}

void ImporterBlock::run() {
    done = 0;

    lines->pushHandler = [&](TextLine const & line) {
        onRead(line);
        ++done;
        progress.output();
    };

    lines->doneHandler = [&]() {
        onDone();
        progress.stop();
    };

    lines.push();
}

void ImporterBlock::onRead(TextLine const & line) {
}

void ImporterBlock::onDone() {
}

