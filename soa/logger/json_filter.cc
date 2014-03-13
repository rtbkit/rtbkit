/* json_filter.cc
   Jeremy Barnes, 5 June 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Filters to pre-compress JSON quickly.
*/

#include "json_filter.h"
#include "jml/db/portable_oarchive.h"
#include <unordered_map>
#include <map>
#include "jml/utils/hash_specializations.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/format.h"

using namespace std;
using namespace ML;


namespace Datacratic {

/*****************************************************************************/
/* JSON COMPRESSOR                                                           */
/*****************************************************************************/

struct JsonContext {
    JsonContext()
        : stringNum(0)
    {
        size_t buf_size = 65536;
        start = new char[buf_size];
        current = start;
        end = start + buf_size;
    }

    ~JsonContext()
    {
        //cerr << "stringNum = " << stringNum << endl;
    }

    void writeByte(uint8_t byte)
    {
        if (JML_UNLIKELY(current >= end)) {
            //cerr << "overflow " << (int)byte << endl;
            overflow += byte;
            return;
        }
        *current++ = byte;
    }

    void writeSize(size_t size)
    {
        int len = DB::compact_encode_length(size);

        if (JML_UNLIKELY(current + len > end)) {
            char buf[len];
            char * b = buf;
            DB::encode_compact(b, buf + len, size);
            overflow.append(buf, 0, len);
            return;
        }

        DB::encode_compact(current, end, size);
    }

    void writeString(const std::string & s)
    {
        writeSize(s.length());
        writeBinary(s.c_str(), s.length());
    }
    
    void writeString(const char * first, const char * last)
    {
        size_t sz = last - first;
        writeSize(sz);
        writeBinary(first, last);
    }

    void writeBinary(const char * first, const char * last)
    {
        writeBinary(first, last - first);
    }

    void writeBinary(const char * first, size_t sz)
    {
        if (JML_UNLIKELY(current + sz > end)) {
            overflow.append(first, first + sz);
            return;
        }

        std::copy(first, first + sz, current);
        current += sz;
    }
    
    void reset()
    {
    }

    void writeInternedString(const std::string & s)
    {
        auto itC = commonStrings.find(s);

        if (itC != commonStrings.end()) {
            int idx = itC->second;
            if (idx < 128) {
                writeByte(idx + 128);
                return;
            }
            else {
                writeByte('C');
                writeByte(idx - 128);
                return;
            }
        }

        auto it = lastStrings.find(s);

        if (it != lastStrings.end()) {
            // It's already there; write a reference to it
            size_t n = it->second.index;
            ++it->second.count;
            writeByte('R');  // interned string reference
            writeSize(n);
        }
        else {
            // not there
            lastStrings.insert(make_pair(s, StringEntry(stringNum, 0))).second;
            //cerr << "string " << s << " is at ID " << stringNum << endl;
            ++stringNum;
            writeByte('S');  // interned string definition
            writeString(s);
        }
        
        if (stringNum == 5000)
            sortStrings();
    }

    void sortStrings()
    {
        writeByte('<');
        vector<pair<size_t, std::string> > counts;
        counts.reserve(lastStrings.size());
        
        for (auto it = lastStrings.begin(), end = lastStrings.end();  it != end;  ++it) {
            size_t count = it->second.count;
            if (count < 2) continue;
            counts.push_back(make_pair(count, it->first));
        }

        std::sort(counts.begin(), counts.end(), std::greater<std::pair<size_t, string> >());

        commonStrings.clear();

        for (unsigned i = 0;  i < 256 + 128 && i < counts.size();  ++i)
            commonStrings[counts[i].second] = i;
    }

    struct StringEntry {
        StringEntry(int index = 0, int count = 0)
            : index(index), count(count)
        {
        }

        size_t index;
        size_t count;
    };

#if 0
    struct StringHash {
        size_t operator () (const std::string & str) const
        {
            size_t l = str.length();
            switch (l) {
            case 0: return 0;
            case 1: return ML::chain_hash(str[0], 0);
            case 2: return ML::chain_hash(str[0], str[1]);
            case 3: return ML::chain_hash((str[0] << 8) + str[1], str[2]);
            default:
                return ML::chain_hash((str[0] << 24)
                                      + (str[1] << 16)
                                      + (str[l - 1] << 8)
                                      + l,
                                      (str[l << 1] << 16) + (str[(l << 1) + 1] << 8));
            }
        }
    };
#endif

    std::unordered_map<std::string, int> commonStrings;

    std::unordered_map<std::string, StringEntry> lastStrings;
    size_t stringNum;

    void writeOutput(Filter::OnOutput onOutput,
                     FlushLevel level,
                     boost::function<void ()> onMessageDone)
    {
        if (overflow.empty()) {
            onOutput(start, current - start, level, onMessageDone);
        }
        else {
            onOutput(start, current - start, FLUSH_NONE, []{});
            onOutput(overflow.c_str(), overflow.length(), level, onMessageDone);
            overflow.clear();
        }
        current = start;
    }
    
    char * start;
    char * current;
    char * end;
    
    std::string overflow;
};

struct ParseState {
    ParseState(JsonContext & context)
        : context(context)
    {
    }

    virtual ~ParseState()
    {
    }

    enum ProcessResult {
        CONTINUE,   // State should continue
        FINISHED,   // State has finished without an error
        ERROR       // State had an error; parent should back out to before
    };

    JsonContext & context;

    virtual ProcessResult
    process(const char * & first, const char * last) = 0;

    virtual ProcessResult flush() = 0;
};

struct StringState : public ParseState {

    enum State {
        IN_STRING,
        AFTER_BACKSLASH,
    } state;

    StringState(JsonContext & context)
        : ParseState(context), state(IN_STRING)
    {
        current.reserve(64);
        current = "\"";
    }

    virtual ProcessResult
    process(const char * & first, const char * last)
    {
        while (first < last) {
            switch (state) {
            case IN_STRING: {
                const char * start = first;
                while (first < last && *first != '"' && *first != '\\')
                    ++first;

                if (first == last) {
                    current.append(start, first);
                    return CONTINUE;
                }
                
                char c = *first++;

                if (c == '"') {
                    writeOutput(start, first);
                    return FINISHED;
                }
                else if (c == '\\') {
                    state = AFTER_BACKSLASH;
                }
                break;
            }

            case AFTER_BACKSLASH:
                current += *first++;
                state = IN_STRING;
                break;

            default:
                throw Exception("invalid state");
            };
        }
        return CONTINUE;
    }

    virtual ProcessResult flush()
    {
        writeOutput();
        return CONTINUE;
    }
    
    ProcessResult do_c(char c)
    {
        return CONTINUE;
    }

    void writeOutput(const char * first = 0, const char * last = 0)
    {
        size_t slen = current.size() + (last - first);
        if (slen > 1 && slen < 32) {
            if (current.empty())
                context.writeInternedString(string(first, last));
            else {
                current.append(first, last);
                context.writeInternedString(current);
                current.clear();
            }
        }
        else {
            context.writeByte('s');
            context.writeSize(slen);
            if (!current.empty()) {
                context.writeBinary(current.c_str(), current.length());
                current.clear();
            }
            context.writeBinary(first, last);
        }
    }

    std::string current;
};

struct RootState : public ParseState {

    RootState(JsonContext & context)
        : ParseState(context)
    {
    }
    
    virtual ProcessResult process(const char * & first, const char * last)
    {
        ProcessResult result = CONTINUE;

        for (; first < last;  /* no inc */) {
            if (next) {
                ProcessResult nextResult = next->process(first, last);
                if (nextResult == CONTINUE && first != last)
                    throw Exception("asked to continue but didn't consume all input");
                if (nextResult == FINISHED)
                    next.reset();
            }
            else {
                const char * start = first;
                while (first < last && *first != '"')
                    ++first;


                if (first == last) {
                    current.append(start, first);
                    return CONTINUE;
                }

                if (*first == '"') {
                    writeOutput(start, first);
                    ++first;
                    next.reset(new StringState(context));
                }
                else throw Exception("out of sync");
            }
        }

        if (first > last)
            throw Exception("first > last");
        
        return result;
    }

    virtual ProcessResult flush()
    {
        if (next)
            return next->flush();
        else {
            writeOutput();
            return CONTINUE;
        }
    }

    void writeOutput(const char * first = 0, const char * last = 0)
    {
        size_t slen = current.size() + (last - first);
        if (slen > 1 && slen < 32) {
            if (current.empty())
                context.writeInternedString(string(first, last));
            else {
                current.append(first, last);
                context.writeInternedString(current);
                current.clear();
            }
        }
        else {
            context.writeByte('r');
            context.writeSize(slen);
            if (!current.empty()) {
                context.writeBinary(current.c_str(), current.length());
                current.clear();
            }
            context.writeBinary(first, last);
        }
    }

    void reset()
    {
    }
    
    std::string current;
    std::shared_ptr<ParseState> next;
};



struct JsonCompressor::Itl {
    Itl()
        : state(context)
    {
    }

    JsonContext context;
    RootState state;

    void reset()
    {
        context.reset();
        state.reset();
    }
};

JsonCompressor::
JsonCompressor()
    : itl(new Itl())
{
}

JsonCompressor::
~JsonCompressor()
{
}

void
JsonCompressor::
process(const char * src_begin, const char * src_end,
        FlushLevel level,
        boost::function<void ()> onMessageDone)
{
    itl->state.process(src_begin, src_end);
    if (level != FLUSH_NONE)
        itl->state.flush();
    if (level == FLUSH_FULL)
        itl->reset();
    itl->context.writeOutput(onOutput, level, onMessageDone);
}



/*****************************************************************************/
/* JSON DECOMPRESSOR                                                         */
/*****************************************************************************/

struct JsonDecompressor::Itl {

    struct State;

    typedef void (Itl::* OnDataFn) (const char * &, const char *, State &);
    typedef void (Itl::* OnFinishedFn) (State & current, State & parent);

    struct State {
        State()
            : onData(0), onFinished(0), param(0), len(0), done(0), phase(0)
        {
        }

        // Thing to call once we get data
        OnDataFn onData;
        OnFinishedFn onFinished;
        void * param;

        size_t len, done;
        int phase;
        std::string str;
    };

    std::vector<State> states;

    Itl()
    {
        states.reserve(20);
        pushState(&Itl::processBase);
    }

    State & pushState(OnDataFn onData, OnFinishedFn onFinished = 0,
                      void * param = 0)
    {
        states.push_back(State());
        states.back().onData = onData;
        states.back().onFinished = onFinished;
        states.back().param = param;

        return states.back();
    }

    void popState()
    {
        if (states.size() < 2)
            throw Exception("pop of base state");
        State & current = states.back();

        if (!current.onFinished) {
            states.pop_back();
            return;
        }

        State & parent = states[states.size() - 2];
        
        (this ->* current.onFinished) (current, parent);

        states.pop_back();
    }

    void process(const char * first, const char * last)
    {
        while (first != last) {
            if (states.empty())
                throw Exception("empty state");
            (this ->* states.back().onData)(first, last, states.back());
        }
    }

    void processBase(const char * & first, const char * last, State & state)
    {
        //cerr << "processBase" << endl;

        if (first >= last)
            throw Exception("processing with no characters");
        uint8_t c = *first++;

        switch (c) {

        case 'r':
            pushState(&Itl::processRaw);
            return;

        case 's':
            pushState(&Itl::processString);
            return;

        case 'S':
            pushState(&Itl::processInternedStringDefinition);
            return;

        case 'R':
            pushState(&Itl::processInternedStringReference);
            return;

        case '<':
            sortStrings();
            return;

        case 'C':
            pushState(&Itl::processCommonString);
            return;

        default:
            if (c >= 128)
                current << commonStrings.at(c - 128);
            else
                throw Exception("unknown state character %d %c", c, c);
        }
    }

    void processString(const char * & first, const char * last, State & state)
    {
        //cerr << "processing string input phase " << state.phase
        //     << " with " << last - first << " characters depth "
        //     << states.size() << endl;

        if (first >= last)
            throw Exception("processing with no characters");

        switch (state.phase) {

        case 0: // reading length
            // TODO: if length is inline then do that to avoid indirection

            pushState(&Itl::processLength, &Itl::doneStringLength);
            break;

        case 1: { // reading data
            size_t avail = last - first;
            size_t toRead = std::min(avail, state.len - state.done);
            
            //cerr << "string is " << std::string(first, first + toRead)
            //     << endl;

            current.write(first, toRead);
            first += toRead;
            state.done += toRead;

            //cerr << "string: done " << state.done << " len "
            //     << state.len << endl;

            if (state.done == state.len)
                popState();
            break;
        }
        default:
            throw Exception("processString: invalid phase");
        }
    }

    void doneStringLength(State & current, State & parent)
    {
        //cerr << "string is of length " << current.len << endl;
        parent.done = 0;
        parent.len = current.len;
        parent.phase = 1;
    }

    void processRaw(const char * & first, const char * last, State & state)
    {
        //cerr << "processing raw input" << endl;
        return processString(first, last, state);
    }

    void doneInternedStringLength(State & current, State & parent)
    {
        //cerr << "string is of length " << current.len << endl;
        parent.done = 0;
        parent.len = current.len;
        parent.phase = 1;
        parent.str.reserve(current.len);
    }

    void processInternedStringDefinition(const char * & first,
                                         const char * last,
                                         State & state)
    {
        if (first >= last)
            throw Exception("processing with no characters");

        switch (state.phase) {

        case 0: // reading length
            // TODO: if length is inline then do that to avoid indirection

            pushState(&Itl::processLength, &Itl::doneInternedStringLength);
            break;

        case 1: { // reading data
            size_t avail = last - first;
            size_t toRead = std::min(avail, state.len - state.done);
            
            state.str.append(first, first + toRead);
            first += toRead;
            state.done += toRead;

            if (state.done == state.len) {
                //cerr << "defined interned string " << state.str << " at "
                //     << currentStrings.size() << endl;
                current << state.str;
                currentStrings.push_back(make_pair(1, state.str));
                popState();
            }
            break;
        }
        default:
            throw Exception("processString: invalid phase");
        }
    }

    void doneInternedStringReference(State & current, State & parent)
    {
        parent.phase = 1;
        parent.str = currentStrings.at(current.len).second;
        currentStrings[current.len].first += 1;
        //cerr << "reference to interned string " << current.len
        //     << " -> " << parent.str << endl;
    }

    void processInternedStringReference(const char * & first,
                                        const char * last,
                                        State & state)
    {
        //cerr << "doing interned string reference phase "
        //     << state.phase << endl;

        if (first >= last)
            throw Exception("processing with no characters");

        switch (state.phase) {

        case 0: // reading length
            pushState(&Itl::processLength, &Itl::doneInternedStringReference);
            break;

        case 1: // finished
            current << state.str;
            popState();
            break;
            
        default:
            throw Exception("processString: invalid phase");
        }
    }
    
    void processCommonString(const char * & first, const char * last, State & state)
    {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(first);
        int idx = *p;
        current << commonStrings.at(idx + 128);
        ++first;
        popState();
    }

    /** Extract a length.  Param points to a size_t * which will be filled
        in with the length once it's available.
    */
    void processLength(const char * & first, const char * last, State & state)
    {
        //cerr << "processing length" << endl;
        // Length of the length in bytes
        size_t len = ML::DB::compact_decode_length(*first);
        size_t avail = last - first;

        // If we have enough data, get the length
        if (avail >= len) {
            state.len = ML::DB::decode_compact(first, last);
            //cerr << "length is " << state.len << endl;
            popState();
            return;
        }
        
        // Otherwise, require more data
        requireData(len, first, last);
    }

    void requireData(size_t amount, const char * & first, const char * last)
    {
        //cerr << "requiring " << amount << " bytes" << endl;

        size_t avail = last - first;
        if (avail >= amount)
            throw Exception("required amount already available");
        State & state = pushState(&Itl::processRequireData);
        state.len = amount;
        state.str.reserve(amount);
        state.str.append(first, last);
        first = last;
    }

    void processRequireData(const char * & first, const char * last,
                            State & state)
    {
        // Buffer until we get enough
        size_t avail = last - first;
        size_t toRead = std::min(avail, state.len - state.str.size());
        state.str.append(first, first + toRead);
        first += toRead;

        if (state.len == state.str.size()) {
            // Enough buffered; pass it on to the previous entry in the
            // stack.
            string s = state.str;  // copy so it doesn't go out of scope
            states.pop_back();
            process(s.c_str(), s.c_str() + s.length());
        }
    }
    
    void writeOutput(Filter::OnOutput onOutput,
                     FlushLevel level,
                     boost::function<void ()> onMessageDone)
    {
        string s = current.str();
        if (!s.empty())
            onOutput(s.c_str(), s.length(), level, onMessageDone);
        else onMessageDone();
        current.str("");
    }

    void sortStrings()
    {
        vector<pair<size_t, std::string> > counts = currentStrings;
        std::sort(counts.begin(), counts.end(), std::greater<std::pair<size_t, string> >());

        commonStrings.clear();

        for (unsigned i = 0;  i < 256 + 128 && i < counts.size();  ++i)
            commonStrings.push_back(counts[i].second);
        
        //cerr << "commonStrings = " << commonStrings << endl;
    }

    ostringstream current;
    std::vector<std::string> commonStrings;
    std::vector<std::pair<size_t, std::string> > currentStrings;
};

JsonDecompressor::
JsonDecompressor()
    : itl(new Itl())
{
}

JsonDecompressor::
~JsonDecompressor()
{
}

void
JsonDecompressor::
process(const char * src_begin, const char * src_end,
        FlushLevel level,
        boost::function<void ()> onMessageDone)
{
    itl->process(src_begin, src_end);
    itl->writeOutput(onOutput, level, onMessageDone);
}

} // namespace Datacratic

