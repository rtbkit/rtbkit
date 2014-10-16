/* file_writer_block.cc
   Eric Robert, 5 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

FileWriterBlock::FileWriterBlock() :
    lines(this, "lines"),
    folder("%{output-path}"),
    count(0),
    bytes(0) {
}

void FileWriterBlock::run() {
    auto & env = getPipeline()->environment;
    std::string path = env->expandVariables(folder + "/" + filename);
    LOG(trace) << "opening file '" << path << "'" << std::endl;

    file.open(path);
    if(!file) {
        LOG(error) << "cannot open file '" << filename << "'" << std::endl;
    }

    lines->pushHandler = [&](std::string const & line) {
        file << line << std::endl;
        count += 1;
        bytes += line.size();
    };

    lines->doneHandler = [&]() {
        file.close();
        LOG(trace) << "done writing file '" << folder << "/" << filename << "'" << std::endl;
        LOG(trace) << "done writing " << count << " lines for a total of " << bytes << " bytes" << std::endl;
    };

    lines.push();
}

