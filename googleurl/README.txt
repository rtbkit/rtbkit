                       ==============================
                       The Google URL Parsing Library
                       ==============================

This is the Google URL Parsing Library which parses and canonicalizes URLs.
Please see the LICENSE.txt file for licensing information.

Features
========

   * Easily embeddable: This library was written for a variety of client and
     server programs in mind, so unlike most implementations of URL parsing
     and canonicalization, it can be easily emdedded.

   * Fast: hundreds of thousands of typical URLs can be parsed and
     canonicalized per second on a modern CPU. It is much faster than, for
     example, calling WinInet's corresponding functions.

   * Compatible: When possible, this library has strived for IE7 compatability
     for both general web compatability, and so IE addons or other applications
     that communicate with or embed IE will work properly.

     It supports Unix-style file URLs, as well as the more complex rules for
     Window file URLs. Note that total compatability is not possible (for
     example, IE6 and IE7 disagree about how to parse certain IP addresses),
     and that this is more strict about certain illegal, rarely used, and
     potentially dangerous constructs such as escaped control characters in
     host names that IE will allow. It is typically a little less strict than
     Firefox.


Example
=======

An example implementation of a URL object that uses this library is provided
in src/gurl.*. This implementation uses the "application integration" layer
discussed below to interface with the low-level parsing and canonicalization
functions.


Building
========

The canonicalization files require ICU for some UTF-8 and UTF-16 conversion
macros. If your project does not use ICU, it should be straightforward to
factor out the macros and functions used in ICU, there are only a few well-
isolated things that are used.

TODO(brettw) ADD INSTRUCTIONS FOR GETTING ICU HERE!

logging.h and logging.cc are Windows-only because the corresponding Unix
logging system has many dependencies. This library uses few of the logging
macros, and a dummy header can easily be written that defines the
appropriate things for Unix.


Definitions
===========

"Standard URL": A URL with an "authority", which is a hostname and optionally
   a port, username, and password. Most URLs are standard such as HTTP and FTP.

"File URL": A URL that references a file on disk. There are special rules for
   this type of URL. Note that it may have a hostname! "localhost" is allowed,
   for example "file://localhost/foo" is the same as "file:///foo".

"FileSystem URL": A URL referring to a file reached via the FileSystem API
   described at http://www.w3.org/TR/file-system-api/.  These are nested URLs,
   with compound schemes of e.g. "filesystem:file:" or "filesystem:https:".
   Parsed FileSystem URLs will have a nested inner_parsed() object containing
   information about the inner URL.

"Path URL": This is everything else. There is no standard on how to treat these
   URLs, or even what they are called. This library decomposes them into a
   scheme and a path. The path is everything following the scheme. This type of
   URL includes "javascript", "data", and even "mailto" (although "mailto"
   might look like a standard scheme in some respects, it is not).

Design
======

The library is divided into four layers. They are listed here from the lowest
to the highest; you can use any portion of the library as long as you embed the
layers below it.

1. Parsing
----------
At the lowest level is the parsing code. The files encompassing this are
url_parse.* and the main include file is src/url_parse.h. This code will, given
an input string, parse it into the most likely form of a URL.

Parsing cannot fail and does no validation. The exception is the port number,
which it currently validates, but this is a bug. Given crazy input, the parser
will do its best to find the various URL components according to its rules (see
url_parse_unittest.cc for some examples).

To use this, an application will typically use ExtractScheme to determine the
type of a given input URL, and then call one of the initialization functions:
"ParseStandardURL", "ParsePathURL", or "ParseFileURL". This will result in
a "Parsed" structure which identifies the substrings of each identified
component.

2. Canonicalization
-------------------
At the next highest level is canonicalization. The files encompasing this are
url_canon.* and the main include file is src/url_canon.h. This code will
validate an already-parsed URL, and will convert it to a canonical form. For
example, this will convert host names to lowercase, convert IP addresses
into dotted-decimal notation, handle encoding issues, etc.

This layer will always do its best to produce a reasonable output string, but
it may return that the string is invalid. For example, if there are invalid
characters in the host name, it will escape them or replace them with the
Unicode "invalid character" character, but will fail. This way, the program can
display error messages to the user with the output, log it, etc.  and the
string will have some meaning.

Canonicalized output is written to a CanonOutput object which is a simple
wrapper around an expanding buffer. An implementation called RawCanonOutput is
proivided that writes to a raw buffer with a fixed amount statically allocated
(for performance). Applications using STL can use StdStringCanonOutput defined
in url_canon_stdstring.h which writes into a std::string.

A normal application would call one of the four high-level functions
"CanonicalizeStandardURL", "CanonicalizeFileURL", "CanonicalizeFileSystemURL",
and CanonicalizePathURL" depending on the type of URL in question. Lower-level
functions are also provided which will canonicalize individual parts of a URL
(for example, "CanonicalizeHost").

Part of this layer is the integration with the host system for IDN and encoding
conversion. An implementation that provides integration with the ICU
(http://www-306.ibm.com/software/globalization/icu/index.jsp) is provided in
src/url_canon_icu.cc. The embedder may wish to replace this file with
implementations of the functions for their own IDN library if they do not use
ICU.

3. Application integration
--------------------------
The canonicalization and parsing layers do not know anything about the URI
schemes supported by your application. The parsing and canonicalization
functions are very low-level, and you must call the correct function to do the
work (for example, "CanonicalizeFileURL").

The application integration in url_util.* provides wrappers around the
low-level parsing and canonicalization to call the correct versions for
different identified schemes.  Embedders will want to modify this file if
necessary to suit the needs of their application.

4. URL object
-------------
The highest level is the "URL" object that a C++ application would use to
to encapsulate a URL. Embedders will typically want to provide their own URL
object that meets the requirements of their system. A reasonably complete
example implemnetation is provided in src/gurl.*. You may wish to use this
object, extend or modify it, or write your own.

Whitespace
----------
Sometimes, you may want to remove linefeeds and tabs from the content of a URL.
Some web pages, for example, expect that a URL spanning two lines should be
treated as one with the newline removed. Depending on the source of the URLs
you are canonicalizing, these newlines may or may not be trimmed off.

If you want this behavior, call RemoveURLWhitespace before parsing. This will
remove CR, LF and TAB from the input. Note that it preserves spaces. On typical
URLs, this function produces a 10-15% speed reduction, so it is optional and
not done automatically. The example GURL object and the url_util wrapper does
this for you.

Tests
=====

There are a number of *_unittest.cc and *_perftest.cc files. These files are
not currently compilable as they rely on a not-included unit testing framework
Tests are declared like this:
  TEST(TestCaseName, TestName) {
    ASSERT_TRUE(a);
    EXPECT_EQ(a, b);
  }
If you would like to compile them, it should be straightforward to define
the TEST macro (which would declare a function by combining the two arguments)
and the other macros whose behavior should be self-explanatory (EXPECT is like
an ASSERT, but does not stop the test, if you are doing this, you probably
don't care about this difference). Then you would define a .cc file that
calls all of these functions.
