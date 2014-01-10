/* logs.h
   Eric Robert, 9 October 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Basic log interface
*/

#pragma once

#include <iostream>
#include <sstream>
#include "soa/types/date.h"

namespace Datacratic {

struct Logging
{
    struct Writer {
        virtual ~Writer() {
        }

        virtual void head(char const * timestamp,
                          char const * name,
                          char const * function,
                          char const * file,
                          int line) {
        }

        virtual void body(std::string const & content) {
        }
    };

    struct ConsoleWriter : public Writer {
        ConsoleWriter(bool color = true) :
            color(color) {
        }

        void head(char const * timestamp,
                  char const * name,
                  char const * function,
                  char const * file,
                  int line);

        void body(std::string const & content);

    private:
        bool color;
        std::stringstream stream;
    };

    struct JsonWriter : public Writer {
        void head(char const * timestamp,
                  char const * name,
                  char const * function,
                  char const * file,
                  int line);

        void body(std::string const & content);

    private:
        std::stringstream stream;
    };

    struct Category
    {
        Category(char const * name, Category & super = root);

        bool isEnabled() const;
        bool isDisabled() const;

        void writeTo(std::shared_ptr<Writer> output, bool recurse = true);

        void activate(bool recurse = true);
        void deactivate(bool recurse = true);

        std::ostream & beginWrite(char const * fct, char const * file, int line);

        static Category root;

    public:
        bool enabled;
        char const * name;
        std::shared_ptr<Writer> writer;
        std::stringstream stream;
        Category * parent;
        std::vector<Category *> children;
    };

    struct Printer {
        Printer(Category & category) : category(category) {
        }

        void operator&(std::ostream & stream);

    private:
        Category & category;
    };

    struct Thrower {
        Thrower(Category & category) : category(category) {
        }

        void operator&(std::ostream & stream);

    private:
        Category & category;
    };
};

} // namespace Datacratic

#define LOG(group, ...) \
    group.isDisabled() ? (void) 0 : Logging::Printer(group) & \
    group.beginWrite(__PRETTY_FUNCTION__, __FILE__, __LINE__ __VA_ARGS__)

#define THROW(group, ...) \
    Logging::Thrower(group) & \
    group.beginWrite(__PRETTY_FUNCTION__, __FILE__, __LINE__ __VA_ARGS__)

