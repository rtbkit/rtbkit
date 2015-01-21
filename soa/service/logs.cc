/* logs.cc
   Eric Robert, 9 October 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Basic logs
*/

#include "soa/service/logs.h"
#include "jml/utils/exc_check.h"

#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Datacratic {

void Logging::ConsoleWriter::head(char const * timestamp,
                                  char const * name,
                                  char const * function,
                                  char const * file,
                                  int line) {
    if(color) {
        stream << timestamp << " " << "\033[1;32m" << name << " ";
    }
    else {
        stream << timestamp << " " << name << " ";
    }
}

void Logging::ConsoleWriter::body(std::string const & content) {
    if(color) {
        stream << "\033[1;34m";
        stream.write(content.c_str(), content.size() - 1);
        stream << "\033[0m\n";
    }
    else {
        stream << content;
    }

    std::cerr << stream.str();
    stream.str("");
}

void Logging::FileWriter::head(char const * timestamp,
                               char const * name,
                               char const * function,
                               char const * file,
                               int line) {
    stream << timestamp << " " << name << " ";
}

void Logging::FileWriter::body(std::string const & content) {
    file << stream.str() << content;
    stream.str("");
}

void Logging::FileWriter::open(char const * filename, char const mode) {
    if(mode == 'a')
        file.open(filename, std::ofstream::app);
    else if (mode == 'w')
        file.open(filename);
    else
        throw ML::Exception("File mode not recognized");

    if(!file) {
        std::cerr << "unable to open log file '" << filename << "'" << std::endl;
    }
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
    if(!writer) {
        std::cerr << stream.str();
    }
    else {
        writer->body(stream.str());
    }

    stream.str("");
}

namespace {

struct Registry {
    std::mutex lock;
    std::unordered_map<std::string, std::unique_ptr<Logging::CategoryData> > categories;
};

Registry& getRegistry() {
    // Will leak but that's on program exit so who cares.
    static Registry* registry = new Registry;

    return *registry;
}

} // namespace anonymous

struct Logging::CategoryData {
    bool initialized;
    bool enabled;
    char const * name;
    std::shared_ptr<Writer> writer;
    std::stringstream stream;

    CategoryData * parent;
    std::vector<CategoryData *> children;

    static CategoryData * getRoot();
    static CategoryData * get(char const * name);

    static CategoryData * create(char const * name, char const * super, bool enabled);
    static void destroy(CategoryData * name);

    void activate(bool recurse = true);
    void deactivate(bool recurse = true);
    void writeTo(std::shared_ptr<Writer> output, bool recurse = true);

private:

    CategoryData(char const * name, bool enabled) :
        initialized(false),
        enabled(enabled),
        name(name),
        parent(nullptr) {
    }

};

Logging::CategoryData * Logging::CategoryData::get(char const * name) {
    Registry& registry = getRegistry();

    auto it = registry.categories.find(name);
    return it != registry.categories.end() ? it->second.get() : nullptr;
}

Logging::CategoryData * Logging::CategoryData::getRoot() {
    CategoryData * root = get("*");
    if (root) return root;

    getRegistry().categories["*"].reset(root = new CategoryData("*", true /* enabled */));
    root->parent = root;
    root->writer = std::make_shared<ConsoleWriter>();

    return root;
}

Logging::CategoryData * Logging::CategoryData::create(char const * name, char const * super, bool enabled) {
    Registry& registry = getRegistry();
    std::lock_guard<std::mutex> guard(registry.lock);

    CategoryData * root = getRoot();
    CategoryData * data = get(name);

    if (!data) {
        registry.categories[name].reset(data = new CategoryData(name, enabled));
    }
    else {
        ExcCheck(!data->initialized,
                "making duplicate category: " + std::string(name));
    }

    data->initialized = true;

    data->parent = get(super);
    if (!data->parent) {
        registry.categories[super].reset(data->parent = new CategoryData(super, enabled));
    }

    data->parent->children.push_back(data);

    if (data->parent->initialized) {
        data->writer = data->parent->writer;
    }
    else {
        data->writer = root->writer;
    }

    return data;
}

void Logging::CategoryData::destroy(CategoryData * data) {
    if (data->parent == data) return;

    Registry& registry = getRegistry();
    std::string name = data->name;

    std::lock_guard<std::mutex> guard(registry.lock);

    auto dataIt = registry.categories.find(name);
    ExcCheck(dataIt != registry.categories.end(),
            "double destroy of a category: " + name);

    auto& children = data->parent->children;

    auto childIt = std::find(children.begin(), children.end(), data);
    if (childIt != children.end()) {
        children.erase(childIt);
    }

    CategoryData* root = getRoot();
    for (auto& child : data->children) {
        child->parent = root;
    }

    registry.categories.erase(dataIt);
}


void Logging::CategoryData::activate(bool recurse) {
    enabled = true;
    if(recurse) {
        for(auto item : children) {
            item->activate(recurse);
        }
    }
}

void Logging::CategoryData::deactivate(bool recurse) {
    enabled = false;
    if(recurse) {
        for(auto item : children) {
            item->deactivate(recurse);
        }
    }
}

void Logging::CategoryData::writeTo(std::shared_ptr<Writer> output, bool recurse) {
    writer = output;
    if(recurse) {
        for(auto item : children) {
            item->writeTo(output, recurse);
        }
    }
}

Logging::Category& Logging::Category::root() {
    static Category root(CategoryData::getRoot());
    return root;
}

Logging::Category::Category(CategoryData * data) :
    data(data) {
}

Logging::Category::Category(char const * name, Category & super, bool enabled) :
    data(CategoryData::create(name, super.name(), enabled)) {
}

Logging::Category::Category(char const * name, char const * super, bool enabled) :
    data(CategoryData::create(name, super, enabled)) {
}

Logging::Category::Category(char const * name, bool enabled) :
    data(CategoryData::create(name, "*", enabled)) {
}

Logging::Category::~Category()
{
    CategoryData::destroy(data);
}

char const * Logging::Category::name() const {
    return data->name;
}

bool Logging::Category::isEnabled() const {
    return data->enabled;
}

bool Logging::Category::isDisabled() const {
    return !data->enabled;
}

auto Logging::Category::getWriter() const -> std::shared_ptr<Writer> const &
{
    return data->writer;
}

void Logging::Category::activate(bool recurse) {
    data->activate(recurse);
}

void Logging::Category::deactivate(bool recurse) {
    data->deactivate(recurse);
}

void Logging::Category::writeTo(std::shared_ptr<Writer> output, bool recurse) {
    data->writeTo(output, recurse);
}

// This lock is a quick-fix for the case where a category is used by multiple
// threads. Note that this lock should either eventually be removed or replaced
// by a per category lock. Unfortunately the current setup makes it very
// difficult to pass the header information to the operator& so that everything
// can be dumped in the stream in one go.
namespace { std::mutex loggingMutex; }

std::ostream & Logging::Category::beginWrite(char const * fct, char const * file, int line) {
    loggingMutex.lock();

    timeval now;
    gettimeofday(&now, 0);
    char text[64];
    auto count = strftime(text, sizeof(text), "%Y-%m-%d %H:%M:%S", localtime(&now.tv_sec));
    int ms = now.tv_usec / 1000;
    sprintf(text + count, ".%03d", ms);
    data->writer->head(text, data->name, fct, file, line);
    return data->stream;
}

void Logging::Printer::operator&(std::ostream & stream) {
    std::stringstream & text = (std::stringstream &) stream;
    category.getWriter()->body(text.str());
    text.str("");

    loggingMutex.unlock();
}

void Logging::Thrower::operator&(std::ostream & stream) {
    std::stringstream & text = (std::stringstream &) stream;
    std::string message(text.str());
    text.str("");
    loggingMutex.unlock();

    throw ML::Exception(message);
}

} // namespace Datacratic
