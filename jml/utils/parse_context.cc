/* parse_context.cc                                                 -*- C++ -*-
   Jeremy Barnes, 13 March 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source:$
   $Id:$

   Routines for parsing of text files.
*/

#include "parse_context.h"
#include "stdarg.h"
#include "file_functions.h"
#include "string_functions.h"
#include "jml/arch/exception.h"
#include "fast_int_parsing.h"
#include "fast_float_parsing.h"
#include "jml/utils/file_functions.h"
#include <cassert>
#include <boost/scoped_array.hpp>


using namespace std;


namespace ML {


/*****************************************************************************/
/* PARSE_CONTEXT                                                             */
/*****************************************************************************/

Parse_Context::
Parse_Context()
    : stream_(0), chunk_size_(0), first_token_(0), last_token_(0),
      cur_(0), ebuf_(0),
      line_(0), col_(0), ofs_(0)
{
}

Parse_Context::
Parse_Context(const std::string & filename, const char * start,
              const char * end, unsigned line, unsigned col)
    : stream_(0), chunk_size_(0), first_token_(0), last_token_(0),
      filename_(filename), cur_(start), ebuf_(end),
      line_(line), col_(col), ofs_(0)
{
    current_ = buffers_.insert(buffers_.end(),
                               Buffer(0, start, end - start, false));

    //cerr << "current buffer has " << current_->size << " chars" << endl;
}

Parse_Context::
Parse_Context(const std::string & filename, const char * start,
              size_t length, unsigned line, unsigned col)
    : stream_(0), chunk_size_(0), first_token_(0), last_token_(0),
      filename_(filename), cur_(start), ebuf_(start + length),
      line_(line), col_(col), ofs_(0)
{
    current_ = buffers_.insert(buffers_.end(),
                               Buffer(0, start, length, false));

    //cerr << "current buffer has " << current_->size << " chars" << endl;
}

Parse_Context::
Parse_Context(const std::string & filename)
    : stream_(0), chunk_size_(0), first_token_(0), last_token_(0),
      filename_(filename),
      line_(1), col_(1), ofs_(0)
{
    buf.reset(new File_Read_Buffer(filename));
    cur_ = buf->start();
    ebuf_ = buf->end();
    current_ = buffers_.insert(buffers_.end(),
                               Buffer(0, cur_, ebuf_ - cur_, false));
}

Parse_Context::
Parse_Context(const File_Read_Buffer & buf)
    : stream_(0), chunk_size_(0), first_token_(0), last_token_(0),
      filename_(buf.filename()), cur_(buf.start()), ebuf_(buf.end()),
      line_(1), col_(1), ofs_(0)
{
    current_ = buffers_.insert(buffers_.end(),
                               Buffer(0, cur_, ebuf_ - cur_, false));
}

Parse_Context::
Parse_Context(const std::string & filename, std::istream & stream,
              unsigned line, unsigned col, size_t chunk_size)
    : stream_(&stream), chunk_size_(chunk_size),
      first_token_(0), last_token_(0), filename_(filename), cur_(0), ebuf_(0),
      line_(line), col_(col), ofs_(0)
{
    current_ = read_new_buffer();

    if (current_ != buffers_.end()) {
        cur_ = current_->pos;
        ebuf_ = cur_ + current_->size;
    }
}

Parse_Context::
~Parse_Context()
{
}

void
Parse_Context::
init(const std::string & filename)
{
    stream_ = 0;
    chunk_size_ = 0;
    first_token_ = 0;
    last_token_ = 0;
    filename_ = filename;
    line_ = 1;
    col_ = 1;
    ofs_ = 0;

    buf.reset(new File_Read_Buffer(filename));
    cur_ = buf->start();
    ebuf_ = buf->end();
    current_ = buffers_.insert(buffers_.end(),
                               Buffer(0, cur_, ebuf_ - cur_, false));
}

namespace {

struct MatchAnyChar {
    MatchAnyChar(const char * delimiters, int nd)
    {
        assert(nd > 0 && nd <= 4);
        for (unsigned i = 0;  i < nd;  ++i)
            chars[i] = delimiters[i];
        for (unsigned i = nd;  i < 4;  ++i)
            chars[i] = delimiters[0];
    }
    
    char chars[4];

    bool operator () (char c) const
    {
        return (c == chars[0] || c == chars[1]
                || c == chars[2] || c == chars[3]);
    }
};

struct MatchAnyCharLots {
    MatchAnyCharLots(const char * delimiters, int nd)
    {
        const unsigned char * del2
            = reinterpret_cast<const unsigned char *>(delimiters);

        for (unsigned i = 0;  i < 8;  ++i)
            bits[i] = 0;
        for (unsigned i = 0;  i < nd;  ++i) {
            int x = del2[i];
            bits[x >> 5] |= (1 << (x & 31));
        }
    }
    
    uint32_t bits[8];

    bool operator () (unsigned char c) const
    {
        return bits[c >> 5] & (1 << (c & 31));
    }
};

} // file scope

bool
Parse_Context::
match_text(std::string & text, const char * delimiters)
{
    int nd = strlen(delimiters);

    if (nd == 0)
        throw Exception("Parse_Context::match_text(): no characters");

    if (nd <= 4) return match_text(text, MatchAnyChar(delimiters, nd));
    else return match_text(text, MatchAnyCharLots(delimiters, nd));
}

std::string
Parse_Context::
expect_text(char delimiter, bool allow_empty, const char * error)
{
    string result;
    if (!match_text(result, delimiter)
        || (result.empty() && !allow_empty)) exception(error);
    return result;
}
    
std::string
Parse_Context::
expect_text(const char * delimiters, bool allow_empty, const char * error)
{
    string result;
    if (!match_text(result, delimiters)
        || (result.empty() && !allow_empty)) exception(error);
    return result;
}

bool
Parse_Context::
match_int(int & val_, int min, int max)
{
    Revert_Token tok(*this);
    long val = 0;
    if (!ML::match_int(val, *this)) return false;
    if (val < min || val > max) return false;
    val_ = val;
    tok.ignore();
    return true;
}
    
int
Parse_Context::
expect_int(int min, int max, const char * error)
{
    int result;
    if (!match_int(result, min, max)) exception(error);
    return result;
}

bool
Parse_Context::
match_hex4(int & val_, int min, int max)
{
    Revert_Token tok(*this);
    long val = 0;
    if (!ML::match_hex4(val, *this)) return false;
    if (val < min || val > max) return false;
    val_ = val;
    tok.ignore();
    return true;
}

int
Parse_Context::
expect_hex4(int min, int max, const char * error)
{
    int result;
    if (!match_hex4(result, min, max)) exception(error);
    return result;
}

bool
Parse_Context::
match_unsigned(unsigned & val_, unsigned min, unsigned max)
{
    Revert_Token tok(*this);
    unsigned long val;
    if (!ML::match_unsigned(val, *this)) return false;
    if (val < min || val > max) return false;
    val_ = val;
    tok.ignore();
    return true;
}

unsigned
Parse_Context::
expect_unsigned(unsigned min, unsigned max, const char * error)
{
    unsigned result = 1;
    if (!match_unsigned(result, min, max)) exception(error);
    return result;
}

bool
Parse_Context::
match_long(long & val_, long min, long max)
{
    Revert_Token tok(*this);
    long val = 0;
    if (!ML::match_long(val, *this)) return false;
    if (val < min || val > max) return false;
    val_ = val;
    tok.ignore();
    return true;
}
    
long
Parse_Context::
expect_long(long min, long max, const char * error)
{
    long result;
    if (!match_long(result, min, max)) exception(error);
    return result;
}

bool
Parse_Context::
match_unsigned_long(unsigned long & val_, unsigned long min,
                         unsigned long max)
{
    Revert_Token tok(*this);
    unsigned long val = 0;
    if (!ML::match_unsigned_long(val, *this)) return false;
    if (val < min || val > max) return false;
    val_ = val;
    tok.ignore();
    return true;
}

unsigned long
Parse_Context::
expect_unsigned_long(unsigned long min, unsigned long max,
                          const char * error)
{
    unsigned long result;
    if (!match_unsigned_long(result, min, max)) exception(error);
    return result;
}

bool
Parse_Context::
match_long_long(long long & val_, long long min, long long max)
{
    Revert_Token tok(*this);
    long long val = 0;
    if (!ML::match_long_long(val, *this)) return false;
    if (val < min || val > max) return false;
    val_ = val;
    tok.ignore();
    return true;
}
    
long long
Parse_Context::
expect_long_long(long long min, long long max, const char * error)
{
    long long result;
    if (!match_long_long(result, min, max)) exception(error);
    return result;
}

bool
Parse_Context::
match_unsigned_long_long(unsigned long long & val_, unsigned long long min,
                         unsigned long long max)
{
    Revert_Token tok(*this);
    unsigned long long val = 0;
    if (!ML::match_unsigned_long_long(val, *this)) return false;
    if (val < min || val > max) return false;
    val_ = val;
    tok.ignore();
    return true;
}

unsigned long long
Parse_Context::
expect_unsigned_long_long(unsigned long long min, unsigned long long max,
                          const char * error)
{
    unsigned long long result;
    if (!match_unsigned_long_long(result, min, max)) exception(error);
    return result;
}

bool
Parse_Context::
match_float(float & val, float min, float max)
{
    Revert_Token t(*this);
    if (!ML::match_float(val, *this)) return false;
    if (val < min || val > max) return false;
    t.ignore();
    return true;
}

float
Parse_Context::
expect_float(float min, float max, const char * error)
{
    float val;
    if (!match_float(val, min, max))
        exception(error);
    return val;
}

bool
Parse_Context::
match_double(double & val, double min, double max)
{
    Revert_Token t(*this);
    if (!ML::match_float(val, *this)) return false;
    if (val < min || val > max) return false;
    t.ignore();
    return true;
}

double
Parse_Context::
expect_double(double min, double max, const char * error)
{
    double val;
    if (!match_double(val, min, max))
        exception(error);
    return val;
}

std::string
Parse_Context::
where() const
{
    return filename_ + format(":%zd:%zd", line_, col_);
}

void
Parse_Context::
exception(const std::string & message) const
{
    throw Exception(where() + ": " + message);
}

void
Parse_Context::
exception(const char * message) const
{
    throw Exception(where() + ": " + string(message));
}

void
Parse_Context::
exception_fmt(const char * fmt, ...) const
{
    va_list ap;
    va_start(ap, fmt);
    string str = vformat(fmt, ap);
    va_end(ap);
    exception(str);
}

bool
Parse_Context::
match_literal_str(const char * start, size_t len)
{
    Revert_Token token(*this);

    //cerr << "got revert token" << endl;
    //cerr << "len = " << len << " eof() = " << eof() << " char = " << *cur_
    //     << " match = " << *cur_ << endl;

    while (len && !eof() && *start++ == *cur_) {
        //cerr << "len = " << len << endl;
        operator ++ ();  --len;
    }

    if (len == 0) token.ignore();
    return (len == 0);
}

void
Parse_Context::
next_buffer()
{
    //cerr << "next_buffer: ofs_ = " << ofs_ << " line_ = " << line_
    //     << " col_ = " << col_ << endl;
    //cerr << buffers_.size() << " buffers, eof = "
    //     << (current_ == buffers_.end()) << endl;

    if (current_ == buffers_.end()) {
        return; // eof
        throw Exception("Parse_Context: asked for new buffer when already "
                        " at end");
    }
    else {
        ++current_;
        
        if (current_ == buffers_.end())
            current_ = read_new_buffer();
        
        if (current_ != buffers_.end()) {
            cur_ = current_->pos;
            ebuf_ = cur_ + current_->size;
            //cerr << "got buffer with " << current_->size << " chars"
            //     << endl;
        }

        /* Free any buffers if we can. */
        free_buffers();
    }

    //cerr << "after next_buffer: ofs_ = " << ofs_ << " line_ = " << line_
    //     << " col_ = " << col_ << endl;
    //cerr << buffers_.size() << " buffers, eof = "
    //     << (current_ == buffers_.end()) << endl;
    //int i = 0;
    //for (std::list<Buffer>::const_iterator it = buffers_.begin();
    //     it != buffers_.end();  ++it, ++i) {
    //    cerr << "buffer " << i << " of " << buffers_.size() << ": ofs "
    //         << it->ofs << " size " << it->size << endl;
    //}
}

void
Parse_Context::
goto_ofs(uint64_t ofs, size_t line, size_t col)
{
    //cerr << "goto_ofs: ofs = " << ofs << " line = " << line << " col = "
    //     << col << endl;
    //cerr << "current: ofs_ = " << ofs_ << " line_ = " << line_
    //     << " col_ = " << col_ << endl;
    //cerr << buffers_.size() << " buffers" << endl;

    ofs_ = ofs;
    line_ = line;
    col_ = col;

    int i = 0, s = buffers_.size();
    /* TODO: be more efficient... */
    for (std::list<Buffer>::iterator it = buffers_.begin();
         it != buffers_.end();  ++it, ++i) {
        //cerr << "buffer " << i << " of " << buffers_.size() << ": ofs "
        //     << it->ofs << " size " << it->size << endl;
        if (ofs < it->ofs + it->size
            || (ofs == 0 && it->ofs + it->size == 0)
            || (i == s - 1 && ofs == it->ofs + it->size)) {
            /* In here. */
            cur_ = it->pos + (ofs - it->ofs);
            ebuf_ = it->pos + it->size;
            current_ = it;
            return;
        }
    }

    exception_fmt("Parse_Context::goto_ofs(): couldn't find position %zd (l%zdc%zd)", ofs, line, col);
}

std::string
Parse_Context::
text_between(uint64_t ofs1, uint64_t ofs2) const
{
    std::string result;

    for (auto it = buffers_.begin();
         it != buffers_.end() && ofs1 < ofs2;  ++it) {

        if (ofs1 < it->ofs + it->size
            || (ofs1 == 0 && it->ofs + it->size == 0)) {
            /* In here. */
            const char * cur = it->pos + (ofs1 - it->ofs);
            const char * ebuf = it->pos + it->size;

            int64_t bufToDo = std::min<int64_t>(ofs2 - ofs1, ebuf - cur);
            result.append(cur, cur + bufToDo);
            ofs1 += bufToDo;
        }
    }

    return result;
}

void
Parse_Context::
free_buffers()
{
    /* Free buffers so long as a) it's not the current buffer, and b)
       the first token isn't inside it. */
    for (std::list<Buffer>::iterator it = buffers_.begin();
         it != current_;  /* no inc */) {
        if (first_token_ && (first_token_->ofs < it->ofs + it->size))
            break;  // first token is in this buffer
        std::list<Buffer>::iterator to_erase = it;
        ++it;
        if (to_erase->del) delete[] (const_cast<char *>(to_erase->pos));
        buffers_.erase(to_erase);
    }
}

std::list<Parse_Context::Buffer>::iterator
Parse_Context::
read_new_buffer()
{
    if (!stream_) return buffers_.end();

    if (stream_->eof()) return buffers_.end();

    //cerr << "stream is OK" << endl;

    if (stream_->bad() || stream_->fail())
        exception("stream is bad/has failed 1");
    
    static const size_t MAX_STACK_CHUNK_SIZE = 65536;

    //char tmpbuf_stack[chunk_size_];
    char tmpbuf_stack[std::min(chunk_size_, MAX_STACK_CHUNK_SIZE)];
    char * tmpbuf = tmpbuf_stack;
    boost::scoped_array<char> tmpbuf_dynamic;

    if (chunk_size_ > MAX_STACK_CHUNK_SIZE) {
        tmpbuf_dynamic.reset(new char[chunk_size_]);
        tmpbuf = tmpbuf_dynamic.get();
    }
    
    stream_->read(tmpbuf, chunk_size_);
    size_t read = stream_->gcount();

    //cerr << "read " << read << " bytes" << endl;

    if (stream_->bad())
        exception("stream is bad/has failed 2");
    
    if (read == 0) return buffers_.end();

    uint64_t last_ofs = (buffers_.empty() ? ofs_
                         : buffers_.back().ofs + buffers_.back().size);
    
    //cerr << "last_ofs = " << last_ofs << endl;

    list<Buffer>::iterator result
        = buffers_.insert(buffers_.end(),
                          Buffer(last_ofs, new char[read], read, true));
    
    //cerr << "  now " << buffers_.size() << " buffers active" << endl;

    memcpy(const_cast<char *>(buffers_.back().pos), tmpbuf, read);

    return result;
}

void
Parse_Context::
set_chunk_size(size_t size)
{
    if (size == 0)
        throw Exception("Parse_Context::chunk_size(): invalid chunk size");
    chunk_size_ = size;
}

size_t
Parse_Context::
readahead_available() const
{
    if (eof()) return 0;
    size_t in_current_buffer = ebuf_ - cur_;

    size_t in_future_buffers = 0;
    for (std::list<Buffer>::const_iterator it = boost::next(current_),
             end = buffers_.end();
         it != end;  ++it) {
        in_future_buffers += it->size;
    }

    return in_current_buffer + in_future_buffers;
}

size_t
Parse_Context::
total_buffered() const
{
    if (eof()) return 0;

    size_t result = 0;
    for (std::list<Buffer>::const_iterator it = buffers_.begin(),
             end = buffers_.end();
         it != end;  ++it) {
        result += it->size;
    }

    return result;
}

} // namespace ML
