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

    struct CategoryData;

    struct Category {
        Category(char const * name, Category & super);
        Category(char const * name, char const * super = "*");
        ~Category();

        Category(const Category&) = delete;
        Category& operator=(const Category&) = delete;

        char const * name() const;

        bool isEnabled() const;
        bool isDisabled() const;

        std::shared_ptr<Writer> const & getWriter() const;
        void writeTo(std::shared_ptr<Writer> output, bool recurse = true);

        void activate(bool recurse = true);
        void deactivate(bool recurse = true);

        std::ostream & beginWrite(char const * function, char const * file, int line);

        static Category& root();

    private:
        Category(CategoryData * data);

        CategoryData * data;
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

        void operator&(std::ostream & stream) __attribute__((noreturn));

    private:
        Category & category;
    };

    struct Progress
    {
        Progress(Category & category, std::function<void()> output, double delta = 2.0) :
            output(output),
            done(false),
            category(category),
            delta(delta) {
            start = print = Date::now();
        }

        bool isDisabled() const {
            return !done && Date::now().secondsSince(print) < delta;
        }

        std::ostream & beginWrite(char const * function, char const * file, int line) {
            print = Date::now();
            return category.beginWrite(function, file, line);
        }

        operator Category & () {
            return category;
        }

        double secondsSinceStart() const {
            return Date::now().secondsSince(start);
        }

        void stop() {
            done = true;
            output();
        }

        std::function<void()> output;

    private:
        bool done;
        Category & category;
        Date print;
        Date start;
        double delta;
    };
};

} // namespace Datacratic

#define LOG(group, ...) \
    group.isDisabled() ? (void) 0 : Logging::Printer(group) & \
    group.beginWrite(__PRETTY_FUNCTION__, __FILE__, __LINE__ __VA_ARGS__)

#define THROW(group, ...) \
    Logging::Thrower(group) & \
    group.beginWrite(__PRETTY_FUNCTION__, __FILE__, __LINE__ __VA_ARGS__)

