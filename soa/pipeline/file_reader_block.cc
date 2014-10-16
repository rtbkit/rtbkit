/* file_reader_block.cc
   Eric Robert, 10 January 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

TextLineDescription::
TextLineDescription() {
}

FileReaderBlock::FileReaderBlock() :
    lines(this, "lines"),
    folder("%{input-path}") {
}

void FileReaderBlock::run() {
    auto & env = getPipeline()->environment;
    std::string path = env->expandVariables(folder + "/" + filename);
    LOG(trace) << "opening file '" << path << "'" << std::endl;

    ML::filter_istream stream(path);
    if(!stream) {
        THROW(error) << "cannot open file '" << path << "'" << std::endl;
    }

    TextLine line;
    while(std::getline(stream, line.text)) {
        lines.push(line);
        line.number += 1;
        line.offset += line.text.size();
    }

    LOG(print) << "done reading file '" << folder << "/" << filename << "'" << std::endl;
    LOG(trace) << "done reading " << line.number << " lines for a total of " << line.offset << " bytes" << std::endl;
    lines.done();
}

