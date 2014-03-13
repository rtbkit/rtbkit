// Copyright 2011, Google Inc.
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

// ICU integration functions.

#include <stdlib.h>
#include <string.h>
#include <unicode/ucnv.h>
#include <unicode/ucnv_cb.h>
#include <unicode/uidna.h>

#include "googleurl/src/url_canon_icu.h"
#include "googleurl/src/url_canon_internal.h"  // for _itoa_s

#include "googleurl/base/logging.h"

namespace url_canon {

namespace {

// Called when converting a character that can not be represented, this will
// append an escaped version of the numerical character reference for that code
// point. It is of the form "&#1234;" and we will escape the non-digits to
// "%26%231234%3B". Why? This is what Netscape did back in the olden days.
void appendURLEscapedChar(const void* context,
                          UConverterFromUnicodeArgs* from_args,
                          const UChar* code_units,
                          int32_t length,
                          UChar32 code_point,
                          UConverterCallbackReason reason,
                          UErrorCode* err) {
  if (reason == UCNV_UNASSIGNED) {
    *err = U_ZERO_ERROR;

    const static int prefix_len = 6;
    const static char prefix[prefix_len + 1] = "%26%23";  // "&#" percent-escaped
    ucnv_cbFromUWriteBytes(from_args, prefix, prefix_len, 0, err);

    DCHECK(code_point < 0x110000);
    char number[8];  // Max Unicode code point is 7 digits.
    _itoa_s(code_point, number, 10);
    int number_len = static_cast<int>(strlen(number));
    ucnv_cbFromUWriteBytes(from_args, number, number_len, 0, err);

    const static int postfix_len = 3;
    const static char postfix[postfix_len + 1] = "%3B";   // ";" percent-escaped
    ucnv_cbFromUWriteBytes(from_args, postfix, postfix_len, 0, err);
  }
}

// A class for scoping the installation of the invalid character callback.
class AppendHandlerInstaller {
 public:
  // The owner of this object must ensure that the converter is alive for the
  // duration of this object's lifetime.
  AppendHandlerInstaller(UConverter* converter) : converter_(converter) {
    UErrorCode err = U_ZERO_ERROR;
    ucnv_setFromUCallBack(converter_, appendURLEscapedChar, 0,
                          &old_callback_, &old_context_, &err);
  }

  ~AppendHandlerInstaller() {
    UErrorCode err = U_ZERO_ERROR;
    ucnv_setFromUCallBack(converter_, old_callback_, old_context_, 0, 0, &err);
  }

 private:
  UConverter* converter_;

  UConverterFromUCallback old_callback_;
  const void* old_context_;
};

}  // namespace

ICUCharsetConverter::ICUCharsetConverter(UConverter* converter)
    : converter_(converter) {
}

ICUCharsetConverter::~ICUCharsetConverter() {
}

void ICUCharsetConverter::ConvertFromUTF16(const char16* input,
                                           int input_len,
                                           CanonOutput* output) {
  // Install our error handler. It will be called for character that can not
  // be represented in the destination character set.
  AppendHandlerInstaller handler(converter_);

  int begin_offset = output->length();
  int dest_capacity = output->capacity() - begin_offset;
  output->set_length(output->length());

  do {
    UErrorCode err = U_ZERO_ERROR;
    char* dest = &output->data()[begin_offset];
    int required_capacity = ucnv_fromUChars(converter_, dest, dest_capacity,
                                            input, input_len, &err);
    if (err != U_BUFFER_OVERFLOW_ERROR) {
      output->set_length(begin_offset + required_capacity);
      return;
    }

    // Output didn't fit, expand
    dest_capacity = required_capacity;
    output->Resize(begin_offset + dest_capacity);
  } while (true);
}

// Converts the Unicode input representing a hostname to ASCII using IDN rules.
// The output must be ASCII, but is represented as wide characters.
//
// On success, the output will be filled with the ASCII host name and it will
// return true. Unlike most other canonicalization functions, this assumes that
// the output is empty. The beginning of the host will be at offset 0, and
// the length of the output will be set to the length of the new host name.
//
// On error, this will return false. The output in this case is undefined.
bool IDNToASCII(const char16* src, int src_len, CanonOutputW* output) {
  DCHECK(output->length() == 0);  // Output buffer is assumed empty.
  while (true) {
    // Use ALLOW_UNASSIGNED to be more tolerant of hostnames that violate
    // the spec (which do exist). This does not present any risk and is a
    // little more future proof.
    UErrorCode err = U_ZERO_ERROR;
    int num_converted = uidna_IDNToASCII(src, src_len, output->data(),
                                         output->capacity(),
                                         UIDNA_ALLOW_UNASSIGNED, NULL, &err);
    if (err == U_ZERO_ERROR) {
      output->set_length(num_converted);
      return true;
    }
    if (err != U_BUFFER_OVERFLOW_ERROR)
      return false;  // Unknown error, give up.

    // Not enough room in our buffer, expand.
    output->Resize(output->capacity() * 2);
  }
}

bool ReadUTFChar(const char* str, int* begin, int length,
                 unsigned* code_point_out) {
  int code_point;  // Avoids warning when U8_NEXT writes -1 to it.
  U8_NEXT(str, *begin, length, code_point);
  *code_point_out = static_cast<unsigned>(code_point);

  // The ICU macro above moves to the next char, we want to point to the last
  // char consumed.
  (*begin)--;

  // Validate the decoded value.
  if (U_IS_UNICODE_CHAR(code_point))
    return true;
  *code_point_out = kUnicodeReplacementCharacter;
  return false;
}

bool ReadUTFChar(const char16* str, int* begin, int length,
                 unsigned* code_point) {
  if (U16_IS_SURROGATE(str[*begin])) {
    if (!U16_IS_SURROGATE_LEAD(str[*begin]) || *begin + 1 >= length ||
        !U16_IS_TRAIL(str[*begin + 1])) {
      // Invalid surrogate pair.
      *code_point = kUnicodeReplacementCharacter;
      return false;
    } else {
      // Valid surrogate pair.
      *code_point = U16_GET_SUPPLEMENTARY(str[*begin], str[*begin + 1]);
      (*begin)++;
    }
  } else {
    // Not a surrogate, just one 16-bit word.
    *code_point = str[*begin];
  }

  if (U_IS_UNICODE_CHAR(*code_point))
    return true;

  // Invalid code point.
  *code_point = kUnicodeReplacementCharacter;
  return false;
}

}  // namespace url_canon
