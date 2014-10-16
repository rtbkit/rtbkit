// Copyright 2008, Google Inc.
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

#include "googleurl/src/url_canon.h"
#include "googleurl/src/url_canon_stdstring.h"
#include "googleurl/src/url_parse.h"
#include "googleurl/src/url_test_utils.h"
#include "googleurl/src/url_util.h"
#include "testing/gtest/include/gtest/gtest.h"

TEST(URLUtilTest, FindAndCompareScheme) {
  url_parse::Component found_scheme;

  // Simple case where the scheme is found and matches.
  const char kStr1[] = "http://www.com/";
  EXPECT_TRUE(url_util::FindAndCompareScheme(
      kStr1, static_cast<int>(strlen(kStr1)), "http", NULL));
  EXPECT_TRUE(url_util::FindAndCompareScheme(
      kStr1, static_cast<int>(strlen(kStr1)), "http", &found_scheme));
  EXPECT_TRUE(found_scheme == url_parse::Component(0, 4));

  // A case where the scheme is found and doesn't match.
  EXPECT_FALSE(url_util::FindAndCompareScheme(
      kStr1, static_cast<int>(strlen(kStr1)), "https", &found_scheme));
  EXPECT_TRUE(found_scheme == url_parse::Component(0, 4));

  // A case where there is no scheme.
  const char kStr2[] = "httpfoobar";
  EXPECT_FALSE(url_util::FindAndCompareScheme(
      kStr2, static_cast<int>(strlen(kStr2)), "http", &found_scheme));
  EXPECT_TRUE(found_scheme == url_parse::Component());

  // When there is an empty scheme, it should match the empty scheme.
  const char kStr3[] = ":foo.com/";
  EXPECT_TRUE(url_util::FindAndCompareScheme(
      kStr3, static_cast<int>(strlen(kStr3)), "", &found_scheme));
  EXPECT_TRUE(found_scheme == url_parse::Component(0, 0));

  // But when there is no scheme, it should fail.
  EXPECT_FALSE(url_util::FindAndCompareScheme("", 0, "", &found_scheme));
  EXPECT_TRUE(found_scheme == url_parse::Component());

  // When there is a whitespace char in scheme, it should canonicalize the url
  // before comparison.
  const char whtspc_str[] = " \r\n\tjav\ra\nscri\tpt:alert(1)";
  EXPECT_TRUE(url_util::FindAndCompareScheme(
      whtspc_str, static_cast<int>(strlen(whtspc_str)), "javascript",
      &found_scheme));
  EXPECT_TRUE(found_scheme == url_parse::Component(1, 10));

  // Control characters should be stripped out on the ends, and kept in the
  // middle.
  const char ctrl_str[] = "\02jav\02scr\03ipt:alert(1)";
  EXPECT_FALSE(url_util::FindAndCompareScheme(
      ctrl_str, static_cast<int>(strlen(ctrl_str)), "javascript",
      &found_scheme));
  EXPECT_TRUE(found_scheme == url_parse::Component(1, 11));
}

TEST(URLUtilTest, ReplaceComponents) {
  url_parse::Parsed parsed;
  url_canon::RawCanonOutputT<char> output;
  url_parse::Parsed new_parsed;

  // Check that the following calls do not cause crash
  url_canon::Replacements<char> replacements;
  replacements.SetRef("test", url_parse::Component(0, 4));
  url_util::ReplaceComponents(NULL, 0, parsed, replacements, NULL, &output,
                              &new_parsed);
  url_util::ReplaceComponents("", 0, parsed, replacements, NULL, &output,
                              &new_parsed);
  replacements.ClearRef();
  replacements.SetHost("test", url_parse::Component(0, 4));
  url_util::ReplaceComponents(NULL, 0, parsed, replacements, NULL, &output,
                              &new_parsed);
  url_util::ReplaceComponents("", 0, parsed, replacements, NULL, &output,
                              &new_parsed);

  replacements.ClearHost();
  url_util::ReplaceComponents(NULL, 0, parsed, replacements, NULL, &output,
                              &new_parsed);
  url_util::ReplaceComponents("", 0, parsed, replacements, NULL, &output,
                              &new_parsed);
  url_util::ReplaceComponents(NULL, 0, parsed, replacements, NULL, &output,
                              &new_parsed);
  url_util::ReplaceComponents("", 0, parsed, replacements, NULL, &output,
                              &new_parsed);
}

static std::string CheckReplaceScheme(const char* base_url,
                                      const char* scheme) {
  // Make sure the input is canonicalized.
  url_canon::RawCanonOutput<32> original;
  url_parse::Parsed original_parsed;
  url_util::Canonicalize(base_url, strlen(base_url), NULL,
                         &original, &original_parsed);

  url_canon::Replacements<char> replacements;
  replacements.SetScheme(scheme, url_parse::Component(0, strlen(scheme)));

  std::string output_string;
  url_canon::StdStringCanonOutput output(&output_string);
  url_parse::Parsed output_parsed;
  url_util::ReplaceComponents(original.data(), original.length(),
                              original_parsed, replacements, NULL,
                              &output, &output_parsed);

  output.Complete();
  return output_string;
}

TEST(URLUtilTest, ReplaceScheme) {
  EXPECT_EQ("https://google.com/",
            CheckReplaceScheme("http://google.com/", "https"));
  EXPECT_EQ("file://google.com/",
            CheckReplaceScheme("http://google.com/", "file"));
  EXPECT_EQ("http://home/Build",
            CheckReplaceScheme("file:///Home/Build", "http"));
  EXPECT_EQ("javascript:foo",
            CheckReplaceScheme("about:foo", "javascript"));
  EXPECT_EQ("://google.com/",
            CheckReplaceScheme("http://google.com/", ""));
  EXPECT_EQ("http://google.com/",
            CheckReplaceScheme("about:google.com", "http"));
  EXPECT_EQ("http:", CheckReplaceScheme("", "http"));

#ifdef WIN32
  // Magic Windows drive letter behavior when converting to a file URL.
  EXPECT_EQ("file:///E:/foo/",
            CheckReplaceScheme("http://localhost/e:foo/", "file"));
#endif

  // This will probably change to "about://google.com/" when we fix
  // http://crbug.com/160 which should also be an acceptable result.
  EXPECT_EQ("about://google.com/",
            CheckReplaceScheme("http://google.com/", "about"));
}

TEST(URLUtilTest, DecodeURLEscapeSequences) {
  struct DecodeCase {
    const char* input;
    const char* output;
  } decode_cases[] = {
    {"hello, world", "hello, world"},
    {"%01%02%03%04%05%06%07%08%09%0a%0B%0C%0D%0e%0f/",
     "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0B\x0C\x0D\x0e\x0f/"},
    {"%10%11%12%13%14%15%16%17%18%19%1a%1B%1C%1D%1e%1f/",
     "\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1B\x1C\x1D\x1e\x1f/"},
    {"%20%21%22%23%24%25%26%27%28%29%2a%2B%2C%2D%2e%2f/",
     " !\"#$%&'()*+,-.//"},
    {"%30%31%32%33%34%35%36%37%38%39%3a%3B%3C%3D%3e%3f/",
     "0123456789:;<=>?/"},
    {"%40%41%42%43%44%45%46%47%48%49%4a%4B%4C%4D%4e%4f/",
     "@ABCDEFGHIJKLMNO/"},
    {"%50%51%52%53%54%55%56%57%58%59%5a%5B%5C%5D%5e%5f/",
     "PQRSTUVWXYZ[\\]^_/"},
    {"%60%61%62%63%64%65%66%67%68%69%6a%6B%6C%6D%6e%6f/",
     "`abcdefghijklmno/"},
    {"%70%71%72%73%74%75%76%77%78%79%7a%7B%7C%7D%7e%7f/",
     "pqrstuvwxyz{|}~\x7f/"},
    // Test un-UTF-8-ization.
    {"%e4%bd%a0%e5%a5%bd", "\xe4\xbd\xa0\xe5\xa5\xbd"},
  };

  for (size_t i = 0; i < ARRAYSIZE_UNSAFE(decode_cases); i++) {
    const char* input = decode_cases[i].input;
    url_canon::RawCanonOutputT<char16> output;
    url_util::DecodeURLEscapeSequences(input, strlen(input), &output);
    EXPECT_EQ(decode_cases[i].output,
              url_test_utils::ConvertUTF16ToUTF8(
                string16(output.data(), output.length())));
  }

  // Our decode should decode %00
  const char zero_input[] = "%00";
  url_canon::RawCanonOutputT<char16> zero_output;
  url_util::DecodeURLEscapeSequences(zero_input, strlen(zero_input),
                                     &zero_output);
  EXPECT_NE("%00",
            url_test_utils::ConvertUTF16ToUTF8(
              string16(zero_output.data(), zero_output.length())));

  // Test the error behavior for invalid UTF-8.
  const char invalid_input[] = "%e4%a0%e5%a5%bd";
  const char16 invalid_expected[4] = {0x00e4, 0x00a0, 0x597d, 0};
  url_canon::RawCanonOutputT<char16> invalid_output;
  url_util::DecodeURLEscapeSequences(invalid_input, strlen(invalid_input),
                                     &invalid_output);
  EXPECT_EQ(string16(invalid_expected),
            string16(invalid_output.data(), invalid_output.length()));
}

TEST(URLUtilTest, TestEncodeURIComponent) {
  struct EncodeCase {
    const char* input;
    const char* output;
  } encode_cases[] = {
    {"hello, world", "hello%2C%20world"},
    {"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F",
     "%01%02%03%04%05%06%07%08%09%0A%0B%0C%0D%0E%0F"},
    {"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1A\x1B\x1C\x1D\x1E\x1F",
     "%10%11%12%13%14%15%16%17%18%19%1A%1B%1C%1D%1E%1F"},
    {" !\"#$%&'()*+,-./",
     "%20!%22%23%24%25%26'()*%2B%2C-.%2F"},
    {"0123456789:;<=>?",
     "0123456789%3A%3B%3C%3D%3E%3F"},
    {"@ABCDEFGHIJKLMNO",
     "%40ABCDEFGHIJKLMNO"},
    {"PQRSTUVWXYZ[\\]^_",
     "PQRSTUVWXYZ%5B%5C%5D%5E_"},
    {"`abcdefghijklmno",
     "%60abcdefghijklmno"},
    {"pqrstuvwxyz{|}~\x7f",
     "pqrstuvwxyz%7B%7C%7D~%7F"},
  };

  for (size_t i = 0; i < ARRAYSIZE_UNSAFE(encode_cases); i++) {
    const char* input = encode_cases[i].input;
    url_canon::RawCanonOutputT<char> buffer;
    url_util::EncodeURIComponent(input, strlen(input), &buffer);
    std::string output(buffer.data(), buffer.length());
    EXPECT_EQ(encode_cases[i].output, output);
  }
}

