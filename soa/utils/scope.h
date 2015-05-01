/* scope.h                                                         -*- C++ -*-
   Mathieu Stefani, 26 November 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
  * When you exit a scope, you might want to execute a piece of code depending on the
  * context in which the scope exited. For example, when an exception occurs, it can
  * be considered as a "failure" and you might want to execute a certain piece of code
  * (typical RAII containers are using this concept to guarantee exception-safety).
  *
  * This header provides three components to deal with scope exit:
  *
  * * Exit : the Exit component will execute a function whatever happens 
  * * Success : the Success component will execute a function if and only if no exception has
  *   been thrown (no stack unwinding is currently in progress when reaching the destructor)
  * * Failure : the Failure component will execute a function if and only if an exception has
  *   been throw and stack unwinding is currently in progress or if the user explicitly 
  *   signaled the failure.
  *
  * The structs should not been used directly. Instead, you should use the ScopeExit,
  * ScopeSuccess and ScopeFailure functions like so:
  *
  *
  * Transaction transaction;
  * auto failure = ScopeFailure([&] { transaction.cancel(); });
  *
  * auto success = ScopeSuccess([&] { transaction.commit(); });
  *
  * If the code to execute is simple enough, you also can use the macros-version :
  *
  * Scope_Success(transaction.commit());
  * Scope_Failure(transaction.cancel());
  *
  * Note that this is inspired from D's scope() statement
*/

#pragma once

#include <exception>
#include <utility>

namespace Datacratic {

namespace Scope {

    template<typename Func>
    struct Base {
        Base(Func func)
            : func { func }
            , active { true }
        {
            static_assert(noexcept(func()), "The function must be declared noexcept");
        }

        Base(const Base<Func>& other) = delete;
        Base& operator=(const Base<Func>& other) = delete;

        Base(Base<Func>&& other) = default;
        Base& operator=(Base<Func>&& other) = default;

        void clear() { active = false; }

        ~Base() noexcept { }

    private:
        Func func;
        bool active;
    protected:
        void exec() noexcept {
            if (active) func();
        } 
    };

    template<typename Func>
    struct Exit : public Base<Func> {
        Exit(Func func)
            : Base<Func>(func)
        { }
	Exit(Exit<Func>&& other) = default;
        Exit& operator=(Exit<Func>&& other) = default;

        ~Exit() noexcept { Exit<Func>::exec(); }
    };

    template<typename Func>
    struct Success : public Base<Func> {
        Success(Func func)
            : Base<Func>(func)
        { }
        Success(Success<Func>&& other) = default;
        Success& operator=(Success<Func>&& other) = default;

        ~Success() noexcept {
            if (!std::uncaught_exception()) {
                Success<Func>::exec();
            }
        }
    };

    template<typename Func>
    struct Failure : public Base<Func> {
        Failure(Func func)
            : Base<Func>(func)
            , failed { false }
        { }

	Failure(Failure<Func>&& other) = default;
        Failure& operator=(Failure<Func>&& other) = default;

        ~Failure() noexcept {
            if (std::uncaught_exception() || failed) {
                Failure<Func>::exec();
            }
        }

        bool ok() const { return !failed; }

        template<typename T, typename U>
        friend void fail(Failure<T>& failure, U func);

    private:
        bool failed;
    };

    template<typename T, typename Func>
    void fail(Failure<T>& failure, Func func) {
        failure.failed = true;

        func();
    }

} // namespace Scope

template<typename Func>
Scope::Exit<Func> ScopeExit(Func && func) {
    return Scope::Exit<Func>(std::forward<Func>(func));
}

template<typename Func>
Scope::Success<Func> ScopeSuccess(Func && func) {
    return Scope::Success<Func>(std::forward<Func>(func));
}

template<typename Func>
Scope::Failure<Func> ScopeFailure(Func && func) {
    return Scope::Failure<Func>(std::forward<Func>(func));
}

using Scope::fail;


#define CAT(a, b) a##b
#define LABEL_(prefix, a) CAT(prefix, a)
#define UNIQUE_LABEL(prefix) LABEL_(CAT(__scope, prefix), __LINE__)

#define Scope_Exit(func) \
    auto UNIQUE_LABEL(exitlambda) = [&]() noexcept { func; }; \
    Scope::Exit<decltype(UNIQUE_LABEL(exitlambda))> UNIQUE_LABEL(exit)(std::move(UNIQUE_LABEL(exitlambda))); \
    (void) 0

#define Scope_Success(func) \
    auto UNIQUE_LABEL(successlambda) = [&]() noexcept { func; }; \
    Scope::Success<decltype(UNIQUE_LABEL(successlambda))> UNIQUE_LABEL(success)(std::move(UNIQUE_LABEL(successlambda))); \
    (void) 0

#define Scope_Failure(func) \
    auto UNIQUE_LABEL(failurelambda) = [&]() noexcept { func; }; \
    Scope::Failure<decltype(UNIQUE_LABEL(failurelambda))> UNIQUE_LABEL(failure)(std::move(UNIQUE_LABEL(failurelambda))); \
    (void) 0

} // namespace Datacratic
