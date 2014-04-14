/* logs.cc
   Eric Robert, 9 October 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Basic logs
*/

#include <iostream>
#include "soa/service/logs.h"

using namespace Datacratic;

void Logging::ConsoleWriter::head(char const * timestamp,
                                  char const * name,
                                  char const * function,
                                  char const * file,
                                  int line) {
    stream << timestamp << " " << name << " " << file << ":" << line << " - " << function;
}

void Logging::ConsoleWriter::body(std::string const & content) {
    if(color) {
        stream << " \033[1;31m";
        stream.write(content.c_str(), content.size() - 1);
        stream << "\033[0m\n";
    }
    else {
        stream << " " << content;
    }

    std::cerr << stream.str();
    stream.str("");
}

void Logging::JsonWriter::head(char const * timestamp,
                               char const * name,
                               char const * function,
                               char const * file,
                               int line) {
    stream << "{\"time\":\"" << timestamp
           << "\",\"name\":\"" << name
           << "\",\"call\":\"" << function
           << "\",\"file\":\"" << file
           << "\",\"line\":" << line
           << ",\"text\":\"";
}

void Logging::JsonWriter::body(std::string const & content) {
    stream.write(content.c_str(), content.size() - 1);
    stream << "\"}\n";
    std::cerr << stream.str();
    stream.str("");
}

Logging::Category::Category(char const * name, Category & super) :
    initialized(true),
    enabled(true),
    name(name),
    parent(&super)
{
    if(parent != this) {
        nextChild = parent->children;
        parent->children = this;
    }

    // Make sure to propagate our state to any children who were initialized
    // before we were.

    if(parent->initialized && parent != this) {
        parent->enabled ? activate() : deactivate();
        writeTo(parent->writer);
    }
    else {
        enabled ? activate() : deactivate();
        writeTo(std::make_shared<ConsoleWriter>());
    }
}

bool Logging::Category::isEnabled() const {
    return enabled;
}

bool Logging::Category::isDisabled() const {
    return !enabled;
}

void Logging::Category::activate(bool recurse) {
    enabled = true;
    if(recurse) {
        for(auto item = children; item; item = item->nextChild) {
            item->activate(recurse);
        }
    }
}

void Logging::Category::deactivate(bool recurse) {
    enabled = false;
    if(recurse) {
        for(auto item = children; item; item = item->nextChild) {
            item->deactivate(recurse);
        }
    }
}

void Logging::Category::writeTo(std::shared_ptr<Writer> output, bool recurse) {
    writer = output;
    if(recurse) {
        for(auto item = children; item; item = item->nextChild) {
            item->writeTo(output, recurse);
        }
    }
}

auto Logging::Category::getWriter() const -> std::shared_ptr<Writer> const &
{
    return writer;
}

std::ostream & Logging::Category::beginWrite(char const * fct, char const * file, int line) {
    timeval now;
    gettimeofday(&now, 0);
    char text[64];
    auto count = strftime(text, sizeof(text), "%Y-%m-%d %H:%M:%S", localtime(&now.tv_sec));
    int ms = now.tv_usec / 1000;
    sprintf(text + count, ".%03d", ms);
    writer->head(text, name, fct, file, line);
    return stream;
}

Logging::Category Logging::Category::root("*");

void Logging::Printer::operator&(std::ostream & stream) {
    std::stringstream & text = (std::stringstream &) stream;
    category.getWriter()->body(text.str());
    text.str("");
}

void Logging::Thrower::operator&(std::ostream & stream) {
    std::stringstream & text = (std::stringstream &) stream;
    std::string message(text.str());
    text.str("");
    throw ML::Exception(message);
}

