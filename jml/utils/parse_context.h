/* parse_context.h                                                  -*- C++ -*-
   Jeremy Barnes, 27 January 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
      
   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   A class to ease the recursive-descent parsing of strings.
*/

#ifndef __utils__parse_context_h__
#define __utils__parse_context_h__

#include "jml/utils/unnamed_bool.h"
#include "jml/arch/exception.h"
#include "jml/compiler/compiler.h"
#include <cmath>
#include <string>
#include <iostream>
#include <list>
#include <limits.h>
#include <string.h>
#include <stdint.h>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>


namespace ML {


class File_Read_Buffer;


/*****************************************************************************/
/* PARSE_CONTEXT                                                             */
/*****************************************************************************/

struct Parse_Context {

    /** Create but don't initialize. */
    Parse_Context();

    /** Initialize from a filename, loading the file and uncompressing if
        necessary. */
    explicit Parse_Context(const std::string & filename);
    
    /** Initialize from a memory region. */
    Parse_Context(const std::string & filename, const char * start,
                  const char * finish, unsigned line = 1, unsigned col = 1);

    Parse_Context(const std::string & filename, const char * start,
                  size_t length, unsigned line = 1, unsigned col = 1);

    /** Initialize from a File_Read_Buffer. */
    explicit Parse_Context(const File_Read_Buffer & buf);

    /** Default chunk size. */
    enum { DEFAULT_CHUNK_SIZE = 65500 };

    /** Initialize from an istream. */
    Parse_Context(const std::string & filename, std::istream & stream,
                  unsigned line = 1, unsigned col = 1,
                  size_t chunk_size = DEFAULT_CHUNK_SIZE);

    ~Parse_Context();

    /** Initialize from a filename, loading the file and uncompressing if
        necessary. */
    void init(const std::string & filename);

    /** Initialize from a memory region. */
    void init(const std::string & filename, const char * start,
              const char * finish, unsigned line = 1, unsigned col = 1);

    /** Initialize from a File_Read_Buffer. */
    void init(const File_Read_Buffer & buf);

    /** Initialize from an istream. */
    void init(const std::string & filename, std::istream & stream,
              unsigned line = 1, unsigned col = 1);

    /** Set the chunk size for the buffers.  Mostly used for testing
        purposes.  Note that this is only useful when initialized from
        a stream. */
    void set_chunk_size(size_t size);

    /** Get the chunk size. */
    size_t get_chunk_size() const { return chunk_size_; }

    /** How many characters are available to read ahead from? */
    size_t readahead_available() const;

    /** How many characters are buffered in total, both before and after
        the current character? */
    size_t total_buffered() const;

    /** Increment.  Note that it always sets up the buffer such that more
        characters are available. */
    JML_ALWAYS_INLINE Parse_Context & operator ++ ()
    {
        if (eof()) exception("unexpected EOF");

        if (*cur_ == '\n') { ++line_;  col_ = 0; }
        ofs_ += 1;  col_ += 1;

        ++cur_;
        if (JML_UNLIKELY(cur_ == ebuf_))
            next_buffer();

        return *this;
    }

    /** Little helper class that allows character at current position to
        be returned.  This allows us to do "*context++", without making
        a (heavyweight!) copy at each stage (as context++ requires a copy
        of the object to be returned).
    */
    struct Last_Char {
        Last_Char(char c) : c(c) {}
        char operator * () const { return c; }
        char c;
    };
    
    Last_Char operator ++ (int)
    {
        char result = operator * ();
        operator ++ ();
        return Last_Char(result);
    }

    char operator * () const
    {
        if (eof()) exception("unexpected EOF");
        return *cur_;
    }
    
    /** Match a literal character.  Return true if matched or false if not.
        Never throws.
    */
    bool match_literal(char c)
    {
        if (eof()) return false;
        if (*cur_ == c) { operator ++();  return true; }
        return false;
    }
    
    /** Expect a literal character.  Throws if the character is not matched. */
    void expect_literal(char c, const char * error = "expected '%c', got '%c'")
    {
        if (!match_literal(c)) exception_fmt(error, c, (eof() ? '\0' : *cur_));
    }
    
    /** Match a literal string.  Returns true if it was matched and false if
        not.  Never throws.
    */
    bool match_literal(const std::string & str)
    {
        return match_literal_str(str.data(), str.length());
    }
    
    /** Expect a literal string.  Throws an exception if the given string
        was not at the current position.
    */
    void expect_literal(const std::string & str,
                        const char * error = "expected '%s'")
    {
        if (!match_literal(str)) exception_fmt(error, str.c_str());
    }

    /** Match a literal string.  Returns true if it was matched and false if
        not.  Never throws.
    */
    bool match_literal(const char * str)
    {
        return match_literal_str(str, strlen(str));
    }
    
    /** Expect a literal string.  Throws an exception if the given string
        was not at the current position.
    */
    void expect_literal(const char * str,
                        const char * error = "expected '%s'")
    {
        if (!match_literal(str)) exception_fmt(error, str);
    }

    template<class FoundEnd>
    bool match_text(std::string & text, const FoundEnd & found)
    {
#if 0 // buggy        
        /* We do each buffer separately, to avoid overhead. */
        while (!eof()) {
            const char * text_start = cur_;

            /* Go to an EOF or the end of the buffer, whatever first. */
            while (cur_ < ebuf_ && !found(*cur_)) ++cur_;
            
            /* Copy the text. */
            text.append(text_start, cur_);
            
            /* Did we find the end of line? */
            if (cur_ < ebuf_) break;
            
            /* We need a new buffer. */
            --cur_;  // make sure the operator ++ will return a new buffer
            operator ++ ();  // get the new buffer
        }
#else
        char internalBuffer[4096];
        
        char * buffer = internalBuffer;
        size_t bufferSize = 4096;
        size_t pos = 0;
        char c;
        
        while (!eof() && !found(c = operator *())) {
            if (pos == bufferSize) {
                size_t newBufferSize = bufferSize * 8;
                char * newBuffer = new char[newBufferSize];
                std::copy(buffer, buffer + bufferSize, newBuffer);
                if (buffer != internalBuffer)
                    delete[] buffer;
                buffer = newBuffer;
                bufferSize = newBufferSize;
            }
            buffer[pos++] = c;
            operator ++();
        }

        text = std::string(buffer, buffer + pos);
        if (buffer != internalBuffer)
            delete[] buffer;
#endif

        return true;
    }

    struct Matches_Char {
        Matches_Char(char c) : c(c) {}
        char c;
        bool operator () (char c2) const { return c == c2; }
    };

    /** Match a string of any length delimited by the given character.  EOF is
        implicitly considered a delimiter.  The text may be of zero length if
        the delimiter is encountered straight away.  The text up to but not
        including the delimiter is returned in text, and the position will be
        at the delimiter.  Always returns true, as the empty string counts as
        being matched.
    */
    bool match_text(std::string & text, char delimiter)
    {
        return match_text(text, Matches_Char(delimiter));
    }

    bool match_text(std::string & text, const char * delimiters);

    std::string expect_text(char delimiter,
                            bool allow_empty = true,
                            const char * error = "expected text");
    
    std::string expect_text(const char * delimiters,
                            bool allow_empty = true,
                            const char * error = "expected text");
    
    bool match_int(int & val, int min = -INT_MAX, int max = INT_MAX);
    
    int expect_int(int min = -INT_MAX, int max = INT_MAX,
                   const char * error = "expected integer");

    bool match_hex4(int & val, int min = -INT_MAX, int max = INT_MAX);
    
    int expect_hex4(int min = -INT_MAX, int max = INT_MAX,
                   const char * error = "invalid hexadecimal in code");


    bool match_unsigned(unsigned & val, unsigned min = 0,
                        unsigned max = INT_MAX);
    
    unsigned expect_unsigned(unsigned min = 0, unsigned max = INT_MAX,
                        const char * error = "expected unsigned");
 
    bool match_long(long & val,
                    long min = LONG_MIN,
                    long max = LONG_MAX);
    
    long expect_long(long min = -LONG_MAX,
                     long max = LONG_MAX,
                     const char * error = "expected long integer");

    bool match_unsigned_long(unsigned long & val,
                             unsigned long min = 0,
                             unsigned long max = ULONG_MAX);
    
    unsigned long
    expect_unsigned_long(unsigned long min = 0,
                         unsigned long max = ULONG_MAX,
                         const char * error = "expected long integer");
    
    bool match_long_long(long long & val,
                         long long min = LONG_LONG_MIN,
                         long long max = LONG_LONG_MAX);
    
    long long
    expect_long_long(long long min = -LONG_LONG_MAX,
                     long long max = LONG_LONG_MAX,
                     const char * error = "expected long long integer");

    bool match_unsigned_long_long(unsigned long long & val,
                                  unsigned long long min = 0,
                                  unsigned long long max = ULONG_LONG_MAX);
    
    unsigned long long
    expect_unsigned_long_long(unsigned long long min = 0,
                              unsigned long long max = ULONG_LONG_MAX,
                              const char * error = "expected long long integer");
    
    /** Matches a floating point value in the given range. */
    bool match_float(float & val, float min = -INFINITY, float max = INFINITY);
    
    float expect_float(float min = -INFINITY, float max = INFINITY,
                       const char * error = "expected float");
    
    /** Matches a floating point value in the given range. */
    bool match_double(double & val,
                      double min = -INFINITY, double max = INFINITY);
    
    double expect_double(double min = -INFINITY, double max = INFINITY,
                         const char * error = "expected double");
    
    bool match_whitespace()
    {
        bool result = false;
        while (!eof() && isblank(*cur_)) {
        // while (!eof() && isspace(*cur_) && *cur_ != '\n') {
            result = true;
            operator ++ ();
        }
        return result;
    }

    void skip_whitespace()
    {
        match_whitespace();
    }
    
    void expect_whitespace()
    {
        if (!match_whitespace()) exception("expected whitespace");
    }

    bool match_numeric(signed int & i)
    {
        return match_int(i);
    }

    bool match_numeric(unsigned int & i)
    {
        return match_unsigned(i);
    }

    template<typename MatchAs, typename T>
    bool match_numeric_as(T & i)
    {
        Revert_Token token(*this);
        MatchAs r;
        if (!match_numeric(r)) return false;
        i = r;
        if (i != r)
            exception("type did not fit in range");
        token.ignore();
        return true;
    }

    bool match_numeric(short signed int & i)
    {
        return match_numeric_as<int>(i);
    }

    bool match_numeric(short unsigned int & i)
    {
        return match_numeric_as<unsigned>(i);
    }

    bool match_numeric(signed char & i)
    {
        return match_numeric_as<int>(i);
    }

    bool match_numeric(unsigned char & i)
    {
        return match_numeric_as<unsigned int>(i);
    }

    bool match_numeric(signed long & i)
    {
        return match_numeric_as<signed long long>(i);
    }

    bool match_numeric(unsigned long & i)
    {
        return match_numeric_as<unsigned long long>(i);
    }

    bool match_numeric(signed long long & i)
    {
        return match_long_long(i);
    }

    bool match_numeric(unsigned long long & i)
    {
        return match_unsigned_long_long(i);
    }

    bool match_numeric(float & f)
    {
        return match_float(f);
    }

    bool match_numeric(double & f)
    {
        return match_double(f);
    }

    template<typename T>
    T expect_numeric(const char * error = "expected numeric value of type %s")
    {
        T result;
        if (!match_numeric(result))
            throw ML::Exception(error, typeid(T).name());
        return result;
    }

    /** Return a message giving filename:line:col */
    std::string where() const;
    
    void exception(const std::string & message) const JML_NORETURN;

    void exception(const char * message) const JML_NORETURN;

    void exception_fmt(const char * message, ...) const JML_NORETURN;
    
    size_t get_offset() const { return ofs_; }
    size_t get_line() const { return line_; }
    size_t get_col() const { return col_; }

    /** Query if we are at the end of file.  This occurs when we can't find
        any more characters. */
    JML_ALWAYS_INLINE bool eof() const
    { 
        //using namespace std;
        //cerr << "eof: cur_ = " << (void *)cur_ << "ebuf_ = " << (void *)ebuf_
        //     << endl;
        return cur_ == ebuf_;
    }

    /** Query if we are at the end of file. */
    operator unnamed_bool () const
    {
        return make_unnamed_bool(!eof());
    }

    bool match_eol(bool eof_is_eol = true)
    {
        if (eof_is_eol && eof()) return true;  // EOF is considered EOL
        if (*cur_ == '\n') {
            operator ++ ();
            if (eof_is_eol && eof()) return true;  // EOF is considered EOL
            if (*cur_ == '\r')
                operator ++ ();
            return true;
        }
        if (*cur_ != '\r') return false;

        // deal with DOS line endings
        return match_literal("\r\n");
    }

    void expect_eol(const char * error = "expected eol")
    {
        if (!match_eol()) exception(error);
    }

    void expect_eof(const char * error = "expected eof")
    {
        if (!eof()) exception(error);
    }
    
    bool match_line(std::string & line)
    {
        if (eof()) return false;
        match_text(line, '\n');
        match_eol();
        return true;
    }

    std::string expect_line(const char * error = "expected line of text")
    {
        std::string result;
        if (!match_line(result)) exception(error);
        return result;
    }

    void skip_line()
    {
        expect_line();
    }

    bool match_literal_str(const char * start, size_t len);

protected: 
    /** This token class allows speculative parsing.  It saves the position
        of the parse context, and will on destruction revert back to that
        position, unless it was ignored.

        Note that we require these to be on the stack (it is checked that
        the address of multiple tokens is always descending).  Tokens may
        not be used from more than one thread.

        They are stored as a doubly linked list.  The Parse_Context
        structure maintains a pointer to the earliest one.
    */
    struct Token {
    protected:
        Token(Parse_Context & context)
            : context(&context),
              ofs(context.ofs_), line(context.line_), col(context.col_),
              prev(0), next(0)
        {
            //std::cerr << "creating token " << this
            //          << " at ofs " << ofs << " line " << line
            //          << " col " << col << std::endl;
            //using namespace std;
            //cerr << "next = " << next << " prev = " << prev << endl;
            //cerr << "first = " << context.first_token_ << " last = "
            //     << context.last_token_ << endl;

            prev = context.last_token_;
            if (prev) prev->next = this;
            else context.first_token_ = this;
            context.last_token_ = this;

            //cerr << "next = " << next << " prev = " << prev << endl;
            //cerr << "first = " << context.first_token_ << " last = "
            //     << context.last_token_ << endl;
        }
        
        ~Token()
        {
            //std::cerr << "deleting token " << this << std::endl;
            if (context)
                throw Exception("Parse_Context::Token::~Token(): "
                                "active token was destroyed");
        }
        
        void apply()
        {
            //std::cerr << "applying token " << this << 
            //    " context = " << context << std::endl;
            /* Apply the token.  This reverts us back to the current
               position. */
            if (!context) return;  // nothing to do
            
            //std::cerr << "  check..." << std::endl;

            /* We should be the last token. */
            if (next != 0) {
                //using namespace std;
                //cerr << "next = " << next << " prev = " << prev << endl;
                //cerr << "first = " << context->first_token_ << " last = "
                //     << context->last_token_ << endl;
                context = 0;
                throw Exception("Parse_Context::Token::apply(): logic error: "
                                "applied token was not the latest one");
            }

            //std::cerr << "going to ofs " << ofs << " line " << line
            //          << " col " << col << std::endl;

            //std::cerr << "  goto..." << std::endl;
            context->goto_ofs(ofs, line, col);

            //std::cerr << "  remove..." << std::endl;

            /* Finish off by removing it. */
            remove();
        }
        
        void remove()
        {
            //std::cerr << "removing token " << this << std::endl;
            if (!context) return;  // already ignored

            /* We need to remove this token from the token list. */
            if (prev) prev->next = next;
            else {
                if (context->first_token_ != this)
                    throw Exception("Parse_Context::Token::ignore(): "
                                    "logic error: no prev but not first");
                context->first_token_ = next;
            }
            
            if (next) next->prev = prev;
            else {
                if (context->last_token_ != this)
                    throw Exception("Parse_Context::Token::ignore(): "
                                    "logic error: no next but not last");
                context->last_token_ = prev;
            }

            /* Maybe we can free some buffers since this token no longer
               exists. */
            context->free_buffers();
            context = 0;
        }

        Parse_Context * context;   ///< The Parse_Context object that owns us

        uint64_t ofs;              ///< Offset for this token
        unsigned line;             ///< Line number for this token
        unsigned col;              ///< Column number for this token

        /* Token linked list */
        Token * prev;              ///< The previous token in the series
        Token * next;              ///< The next token in the series

        friend class Parse_Context;
    };
    
public:
    /** A token that, unless ignore() is called, will cause the parse context
        to revert back to its position once it goes out of scope.  Used for
        speculative parsing. */
    struct Revert_Token : public Token {
        Revert_Token(Parse_Context & context)
            : Token(context)
        {
        }

        ~Revert_Token()
        {
            try {
                if (context) apply();
            }
            catch (...) {
                remove();
                throw;
            }
        }

        void ignore()
        {
            remove();
        }

        using Token::apply;
    };

    /** A token that, unless stop() is called, will cause the parse context
        to remember the text from there onwards.  Used to force the
        Parse_Context to buffer text from a certain point onwards. */
    struct Hold_Token : public Token {
        Hold_Token(Parse_Context & context)
            : Token(context)
        {
        }

        ~Hold_Token()
        {
            if (context) remove();
        }

        void stop()
        {
            remove();
        }

        std::string captured() const
        {
            if (!context)
                throw ML::Exception("hold token hasn't captured any text");

            return context->text_between(ofs, context->get_offset());
        }
    };

private:
    /** Go to the next buffer, creating and populating a new one if
        necessary. */
    void next_buffer();

    /** Go to a given offset.  It must be within the current set of buffers. */
    void goto_ofs(uint64_t ofs, size_t line, size_t col);

    /** Return the text between the given offsets. */
    std::string text_between(uint64_t ofs1, uint64_t ofs2) const;

    /** Check if there are and buffers that can be freed, and do so if
        possible. */
    void free_buffers();

    /** This contains a single contiguous block of text. */
    struct Buffer {
        Buffer(uint64_t ofs = 0, const char * pos = 0, size_t size = 0,
               bool del = false)
            : ofs(ofs), pos(pos), size(size), del(del)
        {
        }

        uint64_t ofs;         ///< Offset of first character
        const char * pos;     ///< First character
        size_t size;          ///< Length
        bool del;             ///< Do we delete it once finished with?
    };

    /** Read a new buffer if possible, and update everything.  Doesn't
        do anything if it fails. */
    std::list<Buffer>::iterator read_new_buffer();

    std::istream * stream_;   ///< Stream we read from; zero if none
    size_t chunk_size_;       ///< Size of chunks we read in

    Token * first_token_;     ///< The earliest token
    Token * last_token_;      ///< The latest token

    std::list<Buffer> buffers_;
    std::list<Buffer>::iterator current_;

    std::string filename_;    ///< For reporting errors only

    const char * cur_;        ///< Current position (points inside buffer)
    const char * ebuf_;       ///< Position for the end of the buffer

    size_t line_;             ///< Line number at current position
    size_t col_;              ///< Column number at current position
    uint64_t ofs_;            ///< Offset of current position (chars since 0)

    std::shared_ptr<const File_Read_Buffer> buf;
};

} // namespace ML


#endif /* __utils__parse_context_h__ */
