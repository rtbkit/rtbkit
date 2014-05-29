/* pipeline_test.cc
   Eric Robert, 25 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "soa/pipeline/headers.h"

using namespace Datacratic;

struct MyBlock :
    public Block
{
    MyBlock() :
        readingPin(this, "reading pin"), writingPin(this, "writing pin") {
        readingPin.set("empty");
    }

    void run() {
        std::string value = *readingPin + " " + text;
        writingPin.push(value);
    }

    ReadingPin<std::string> readingPin;
    WritingPin<std::string> writingPin;
    std::string text;
};

BOOST_AUTO_TEST_CASE( test_blocks_basics )
{
    DefaultPipeline pipeline;
    auto a = pipeline.create<MyBlock>("a");
    a->text = "ipsum";

    BOOST_CHECK(a->getName() == "a");
    BOOST_CHECK(a->getPath() == "/a");
    BOOST_CHECK(*(a->readingPin) == "empty");
    BOOST_CHECK(a->writingPin.get() == nullptr);

    a->readingPin.set("Lorem");
    a->run();

    BOOST_CHECK(*(a->writingPin) == "Lorem ipsum");

    auto b = pipeline.create<MyBlock>("b");
    b->text = "dolor";
    b->readingPin.connectWith(a->writingPin);

    pipeline.run();

    BOOST_CHECK(*(b->writingPin) == "Lorem ipsum dolor");
}

struct MyBlockThatMergesLines :
    public Block
{
    MyBlockThatMergesLines() :
        lines(this, "lines"), text(this, "text") {
    }

    void run() {
        text.set("");

        lines->pushHandler = [&](TextLine const & line) {
            if(text->length()) *text += " ";
            *text += line.text;
        };

        lines->doneHandler = [&]() {
            text.push();
        };

        lines.push();
    }

    PullingPin<TextLine> lines;
    WritingPin<std::string> text;
};

struct MyBlockThatSplitsString :
    public Block
{
    MyBlockThatSplitsString() :
        text(this, "text"), lines(this, "lines") {
    }

    void run() {
        std::vector<std::string> items;
        boost::split(items, *text, boost::is_any_of(" "));
        for(auto & item : items) {
            lines.push(item);
        }

        lines.done();
    }

    ReadingPin<std::string> text;
    PushingPin<std::string> lines;
};

BOOST_AUTO_TEST_CASE( test_blocks )
{
    DefaultPipeline pipeline;

    std::string path("./build/x86_64/tmp");
    boost::filesystem::create_directories(path);

    auto environment = std::make_shared<Environment>();
    environment->set("name", "lorem");
    environment->set("input-path", path);
    pipeline.environment.set(environment);

    {
        std::ofstream file(path + "/lorem-1.txt");
        file << "Lorem" << std::endl;
        file << "ipsum" << std::endl;
        file << "dolor" << std::endl;
    }

    auto r = pipeline.create<FileReaderBlock>("r");
    r->filename = "%{name}-1.txt";

    auto a = pipeline.create<MyBlockThatMergesLines>("a");
    a->lines.connectWith(r->lines);

    auto b = pipeline.create<MyBlock>("b");
    b->text = "sit";
    b->readingPin.connectWith(a->text);

    auto c = pipeline.create<MyBlock>("c");
    c->text = "amet.";
    c->readingPin.connectWith(b->writingPin);

    auto d = pipeline.create<MyBlockThatSplitsString>("d");
    d->text.connectWith(c->writingPin);

    auto w = pipeline.create<FileWriterBlock>("w");
    w->filename = "%{name}-2.txt";
    w->folder = "%{input-path}";
    w->lines.connectWith(d->lines);

    pipeline.run();

    {
        std::ifstream file(path + "/lorem-2.txt");
        std::string line;
        std::getline(file, line);
        BOOST_CHECK_EQUAL(line, "Lorem");
        std::getline(file, line);
        BOOST_CHECK_EQUAL(line, "ipsum");
        std::getline(file, line);
        BOOST_CHECK_EQUAL(line, "dolor");
        std::getline(file, line);
        BOOST_CHECK_EQUAL(line, "sit");
        std::getline(file, line);
        BOOST_CHECK_EQUAL(line, "amet.");
    }
}

