// Copyright 2012, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Functions for canonicalizing "filesystem:file:" URLs.

#include "googleurl/src/url_canon.h"
#include "googleurl/src/url_canon_internal.h"
#include "googleurl/src/url_file.h"
#include "googleurl/src/url_parse_internal.h"
#include "googleurl/src/url_util.h"
#include "googleurl/src/url_util_internal.h"

namespace url_canon {

namespace {

// We use the URLComponentSource for the outer URL, as it can have replacements,
// whereas the inner_url can't, so it uses spec.
template<typename CHAR, typename UCHAR>
bool DoCanonicalizeFileSystemURL(const CHAR* spec,
                                 const URLComponentSource<CHAR>& source,
                                 const url_parse::Parsed& parsed,
                                 CharsetConverter* charset_converter,
                                 CanonOutput* output,
                                 url_parse::Parsed* new_parsed) {
  // filesystem only uses {scheme, path, query, ref} -- clear the rest.
  new_parsed->username = url_parse::Component();
  new_parsed->password = url_parse::Component();
  new_parsed->host = url_parse::Component();
  new_parsed->port = url_parse::Component();

  const url_parse::Parsed* inner_parsed = parsed.inner_parsed();
  url_parse::Parsed new_inner_parsed;

  // Scheme (known, so we don't bother running it through the more
  // complicated scheme canonicalizer).
  new_parsed->scheme.begin = output->length();
  output->Append("filesystem:", 11);
  new_parsed->scheme.len = 10;

  if (!parsed.inner_parsed() || !parsed.inner_parsed()->scheme.is_valid())
    return false;

  bool success = true;
  if (url_util::CompareSchemeComponent(spec, inner_parsed->scheme,
      url_util::kFileScheme)) {
    new_inner_parsed.scheme.begin = output->length();
    output->Append("file://", 7);
    new_inner_parsed.scheme.len = 4;
    success &= CanonicalizePath(spec, inner_parsed->path, output,
                                &new_inner_parsed.path);
  } else if (url_util::IsStandard(spec, inner_parsed->scheme)) {
    success =
        url_canon::CanonicalizeStandardURL(spec,
                                           parsed.inner_parsed()->Length(),
                                           *parsed.inner_parsed(),
                                           charset_converter, output,
                                           &new_inner_parsed);
  } else {
    // TODO(ericu): The URL is wrong, but should we try to output more of what
    // we were given?  Echoing back filesystem:mailto etc. doesn't seem all that
    // useful.
    return false;
  }
  // The filesystem type must be more than just a leading slash for validity.
  success &= parsed.inner_parsed()->path.len > 1;

  success &= CanonicalizePath(source.path, parsed.path, output,
                              &new_parsed->path);

  // Ignore failures for query/ref since the URL can probably still be loaded.
  CanonicalizeQuery(source.query, parsed.query, charset_converter,
                    output, &new_parsed->query);
  CanonicalizeRef(source.ref, parsed.ref, output, &new_parsed->ref);
  if (success)
    new_parsed->set_inner_parsed(new_inner_parsed);

  return success;
}

}  // namespace

bool CanonicalizeFileSystemURL(const char* spec,
                               int spec_len,
                               const url_parse::Parsed& parsed,
                               CharsetConverter* charset_converter,
                               CanonOutput* output,
                               url_parse::Parsed* new_parsed) {
  return DoCanonicalizeFileSystemURL<char, unsigned char>(
      spec, URLComponentSource<char>(spec), parsed, charset_converter, output,
      new_parsed);
}

bool CanonicalizeFileSystemURL(const char16* spec,
                               int spec_len,
                               const url_parse::Parsed& parsed,
                               CharsetConverter* charset_converter,
                               CanonOutput* output,
                               url_parse::Parsed* new_parsed) {
  return DoCanonicalizeFileSystemURL<char16, char16>(
      spec, URLComponentSource<char16>(spec), parsed, charset_converter, output,
      new_parsed);
}

bool ReplaceFileSystemURL(const char* base,
                          const url_parse::Parsed& base_parsed,
                          const Replacements<char>& replacements,
                          CharsetConverter* charset_converter,
                          CanonOutput* output,
                          url_parse::Parsed* new_parsed) {
  URLComponentSource<char> source(base);
  url_parse::Parsed parsed(base_parsed);
  SetupOverrideComponents(base, replacements, &source, &parsed);
  return DoCanonicalizeFileSystemURL<char, unsigned char>(
      base, source, parsed, charset_converter, output, new_parsed);
}

bool ReplaceFileSystemURL(const char* base,
                          const url_parse::Parsed& base_parsed,
                          const Replacements<char16>& replacements,
                          CharsetConverter* charset_converter,
                          CanonOutput* output,
                          url_parse::Parsed* new_parsed) {
  RawCanonOutput<1024> utf8;
  URLComponentSource<char> source(base);
  url_parse::Parsed parsed(base_parsed);
  SetupUTF16OverrideComponents(base, replacements, &utf8, &source, &parsed);
  return DoCanonicalizeFileSystemURL<char, unsigned char>(
      base, source, parsed, charset_converter, output, new_parsed);
}

}  // namespace url_canon
