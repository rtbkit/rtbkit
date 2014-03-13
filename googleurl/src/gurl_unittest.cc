// Copyright 2007 Google Inc. All Rights Reserved.
// Author: brettw@google.com (Brett Wilson)

#include "googleurl/src/gurl.h"
#include "googleurl/src/url_canon.h"
#include "googleurl/src/url_test_utils.h"
#include "testing/gtest/include/gtest/gtest.h"

// Some implementations of base/basictypes.h may define ARRAYSIZE.
// If it's not defined, we define it to the ARRAYSIZE_UNSAFE macro
// which is in our version of basictypes.h.
#ifndef ARRAYSIZE
#define ARRAYSIZE ARRAYSIZE_UNSAFE
#endif

using url_test_utils::WStringToUTF16;
using url_test_utils::ConvertUTF8ToUTF16;

namespace {

template<typename CHAR>
void SetupReplacement(void (url_canon::Replacements<CHAR>::*func)(const CHAR*,
                          const url_parse::Component&),
                      url_canon::Replacements<CHAR>* replacements,
                      const CHAR* str) {
  if (str) {
    url_parse::Component comp;
    if (str[0])
      comp.len = static_cast<int>(strlen(str));
    (replacements->*func)(str, comp);
  }
}

// Returns the canonicalized string for the given URL string for the
// GURLTest.Types test.
std::string TypesTestCase(const char* src) {
  GURL gurl(src);
  return gurl.possibly_invalid_spec();
}

}  // namespace

// Different types of URLs should be handled differently by url_util, and
// handed off to different canonicalizers.
TEST(GURLTest, Types) {
  // URLs with unknown schemes should be treated as path URLs, even when they
  // have things like "://".
  EXPECT_EQ("something:///HOSTNAME.com/",
            TypesTestCase("something:///HOSTNAME.com/"));

  // In the reverse, known schemes should always trigger standard URL handling.
  EXPECT_EQ("http://hostname.com/", TypesTestCase("http:HOSTNAME.com"));
  EXPECT_EQ("http://hostname.com/", TypesTestCase("http:/HOSTNAME.com"));
  EXPECT_EQ("http://hostname.com/", TypesTestCase("http://HOSTNAME.com"));
  EXPECT_EQ("http://hostname.com/", TypesTestCase("http:///HOSTNAME.com"));

#ifdef WIN32
  // URLs that look like absolute Windows drive specs.
  EXPECT_EQ("file:///C:/foo.txt", TypesTestCase("c:\\foo.txt"));
  EXPECT_EQ("file:///Z:/foo.txt", TypesTestCase("Z|foo.txt"));
  EXPECT_EQ("file://server/foo.txt", TypesTestCase("\\\\server\\foo.txt"));
  EXPECT_EQ("file://server/foo.txt", TypesTestCase("//server/foo.txt"));
#endif
}

// Test the basic creation and querying of components in a GURL. We assume
// the parser is already tested and works, so we are mostly interested if the
// object does the right thing with the results.
TEST(GURLTest, Components) {
  GURL url(WStringToUTF16(L"http://user:pass@google.com:99/foo;bar?q=a#ref"));
  EXPECT_TRUE(url.is_valid());
  EXPECT_TRUE(url.SchemeIs("http"));
  EXPECT_FALSE(url.SchemeIsFile());

  // This is the narrow version of the URL, which should match the wide input.
  EXPECT_EQ("http://user:pass@google.com:99/foo;bar?q=a#ref", url.spec());

  EXPECT_EQ("http", url.scheme());
  EXPECT_EQ("user", url.username());
  EXPECT_EQ("pass", url.password());
  EXPECT_EQ("google.com", url.host());
  EXPECT_EQ("99", url.port());
  EXPECT_EQ(99, url.IntPort());
  EXPECT_EQ("/foo;bar", url.path());
  EXPECT_EQ("q=a", url.query());
  EXPECT_EQ("ref", url.ref());
}

TEST(GURLTest, Empty) {
  GURL url;
  EXPECT_FALSE(url.is_valid());
  EXPECT_EQ("", url.spec());

  EXPECT_EQ("", url.scheme());
  EXPECT_EQ("", url.username());
  EXPECT_EQ("", url.password());
  EXPECT_EQ("", url.host());
  EXPECT_EQ("", url.port());
  EXPECT_EQ(url_parse::PORT_UNSPECIFIED, url.IntPort());
  EXPECT_EQ("", url.path());
  EXPECT_EQ("", url.query());
  EXPECT_EQ("", url.ref());
}

TEST(GURLTest, Copy) {
  GURL url(WStringToUTF16(L"http://user:pass@google.com:99/foo;bar?q=a#ref"));

  GURL url2(url);
  EXPECT_TRUE(url2.is_valid());

  EXPECT_EQ("http://user:pass@google.com:99/foo;bar?q=a#ref", url2.spec());
  EXPECT_EQ("http", url2.scheme());
  EXPECT_EQ("user", url2.username());
  EXPECT_EQ("pass", url2.password());
  EXPECT_EQ("google.com", url2.host());
  EXPECT_EQ("99", url2.port());
  EXPECT_EQ(99, url2.IntPort());
  EXPECT_EQ("/foo;bar", url2.path());
  EXPECT_EQ("q=a", url2.query());
  EXPECT_EQ("ref", url2.ref());

  // Copying of invalid URL should be invalid
  GURL invalid;
  GURL invalid2(invalid);
  EXPECT_FALSE(invalid2.is_valid());
  EXPECT_EQ("", invalid2.spec());
  EXPECT_EQ("", invalid2.scheme());
  EXPECT_EQ("", invalid2.username());
  EXPECT_EQ("", invalid2.password());
  EXPECT_EQ("", invalid2.host());
  EXPECT_EQ("", invalid2.port());
  EXPECT_EQ(url_parse::PORT_UNSPECIFIED, invalid2.IntPort());
  EXPECT_EQ("", invalid2.path());
  EXPECT_EQ("", invalid2.query());
  EXPECT_EQ("", invalid2.ref());
}

#ifdef FULL_FILESYSTEM_URL_SUPPORT
TEST(GURLTest, CopyFileSystem) {
  GURL url(WStringToUTF16(L"filesystem:https://user:pass@google.com:99/t/foo;bar?q=a#ref"));

  GURL url2(url);
  EXPECT_TRUE(url2.is_valid());

  EXPECT_EQ("filesystem:https://user:pass@google.com:99/t/foo;bar?q=a#ref", url2.spec());
  EXPECT_EQ("filesystem", url2.scheme());
  EXPECT_EQ("", url2.username());
  EXPECT_EQ("", url2.password());
  EXPECT_EQ("", url2.host());
  EXPECT_EQ("", url2.port());
  EXPECT_EQ(url_parse::PORT_UNSPECIFIED, url2.IntPort());
  EXPECT_EQ("/foo;bar", url2.path());
  EXPECT_EQ("q=a", url2.query());
  EXPECT_EQ("ref", url2.ref());

  const GURL* inner = url2.inner_url();
  ASSERT_TRUE(inner);
  EXPECT_EQ("https", inner->scheme());
  EXPECT_EQ("user", inner->username());
  EXPECT_EQ("pass", inner->password());
  EXPECT_EQ("google.com", inner->host());
  EXPECT_EQ("99", inner->port());
  EXPECT_EQ(99, inner->IntPort());
  EXPECT_EQ("/t", inner->path());
  EXPECT_EQ("", inner->query());
  EXPECT_EQ("", inner->ref());
}
#endif

// Given an invalid URL, we should still get most of the components.
TEST(GURLTest, Invalid) {
  GURL url("http:google.com:foo");
  EXPECT_FALSE(url.is_valid());
  EXPECT_EQ("http://google.com:foo/", url.possibly_invalid_spec());

  EXPECT_EQ("http", url.scheme());
  EXPECT_EQ("", url.username());
  EXPECT_EQ("", url.password());
  EXPECT_EQ("google.com", url.host());
  EXPECT_EQ("foo", url.port());
  EXPECT_EQ(url_parse::PORT_INVALID, url.IntPort());
  EXPECT_EQ("/", url.path());
  EXPECT_EQ("", url.query());
  EXPECT_EQ("", url.ref());
}

TEST(GURLTest, Resolve) {
  // The tricky cases for relative URL resolving are tested in the
  // canonicalizer unit test. Here, we just test that the GURL integration
  // works properly.
  struct ResolveCase {
    const char* base;
    const char* relative;
    bool expected_valid;
    const char* expected;
  } resolve_cases[] = {
    {"http://www.google.com/", "foo.html", true, "http://www.google.com/foo.html"},
    {"http://www.google.com/", "http://images.google.com/foo.html", true, "http://images.google.com/foo.html"},
    {"http://www.google.com/blah/bloo?c#d", "../../../hello/./world.html?a#b", true, "http://www.google.com/hello/world.html?a#b"},
    {"http://www.google.com/foo#bar", "#com", true, "http://www.google.com/foo#com"},
    {"http://www.google.com/", "Https:images.google.com", true, "https://images.google.com/"},
      // Unknown schemes are not standard.
    {"data:blahblah", "http://google.com/", true, "http://google.com/"},
    {"data:blahblah", "http:google.com", true, "http://google.com/"},
    {"data:/blahblah", "file.html", false, ""},
#ifdef FULL_FILESYSTEM_URL_SUPPORT
      // Filesystem URLs have different paths to test.
    {"filesystem:http://www.google.com/type/", "foo.html", true, "filesystem:http://www.google.com/type/foo.html"},
    {"filesystem:http://www.google.com/type/", "../foo.html", true, "filesystem:http://www.google.com/type/foo.html"},
#endif
  };

  for (size_t i = 0; i < ARRAYSIZE(resolve_cases); i++) {
    // 8-bit code path.
    GURL input(resolve_cases[i].base);
    GURL output = input.Resolve(resolve_cases[i].relative);
    EXPECT_EQ(resolve_cases[i].expected_valid, output.is_valid()) << i;
    EXPECT_EQ(resolve_cases[i].expected, output.spec()) << i;

    // Wide code path.
    GURL inputw(ConvertUTF8ToUTF16(resolve_cases[i].base));
    GURL outputw =
        input.Resolve(ConvertUTF8ToUTF16(resolve_cases[i].relative));
    EXPECT_EQ(resolve_cases[i].expected_valid, outputw.is_valid()) << i;
    EXPECT_EQ(resolve_cases[i].expected, outputw.spec()) << i;
  }
}

TEST(GURLTest, GetOrigin) {
  struct TestCase {
    const char* input;
    const char* expected;
  } cases[] = {
    {"http://www.google.com", "http://www.google.com/"},
    {"javascript:window.alert(\"hello,world\");", ""},
    {"http://user:pass@www.google.com:21/blah#baz", "http://www.google.com:21/"},
    {"http://user@www.google.com", "http://www.google.com/"},
    {"http://:pass@www.google.com", "http://www.google.com/"},
    {"http://:@www.google.com", "http://www.google.com/"},
#ifdef FULL_FILESYSTEM_URL_SUPPORT
    {"filesystem:http://www.google.com/temp/foo?q#b", "http://www.google.com/"},
    {"filesystem:http://user:pass@google.com:21/blah#baz", "http://google.com:21/"},
#endif
  };
  for (size_t i = 0; i < ARRAYSIZE(cases); i++) {
    GURL url(cases[i].input);
    GURL origin = url.GetOrigin();
    EXPECT_EQ(cases[i].expected, origin.spec());
  }
}

TEST(GURLTest, GetWithEmptyPath) {
  struct TestCase {
    const char* input;
    const char* expected;
  } cases[] = {
    {"http://www.google.com", "http://www.google.com/"},
    {"javascript:window.alert(\"hello, world\");", ""},
    {"http://www.google.com/foo/bar.html?baz=22", "http://www.google.com/"},
#ifdef FULL_FILESYSTEM_URL_SUPPORT
    {"filesystem:http://www.google.com/temporary/bar.html?baz=22", "filesystem:http://www.google.com/temporary/"},
    {"filesystem:file:///temporary/bar.html?baz=22", "filesystem:file:///temporary/"},
#endif
  };

  for (size_t i = 0; i < ARRAYSIZE(cases); i++) {
    GURL url(cases[i].input);
    GURL empty_path = url.GetWithEmptyPath();
    EXPECT_EQ(cases[i].expected, empty_path.spec());
  }
}

TEST(GURLTest, Replacements) {
  // The url canonicalizer replacement test will handle most of these case.
  // The most important thing to do here is to check that the proper
  // canonicalizer gets called based on the scheme of the input.
  struct ReplaceCase {
    const char* base;
    const char* scheme;
    const char* username;
    const char* password;
    const char* host;
    const char* port;
    const char* path;
    const char* query;
    const char* ref;
    const char* expected;
  } replace_cases[] = {
    {"http://www.google.com/foo/bar.html?foo#bar", NULL, NULL, NULL, NULL, NULL, "/", "", "", "http://www.google.com/"},
    {"http://www.google.com/foo/bar.html?foo#bar", "javascript", "", "", "", "", "window.open('foo');", "", "", "javascript:window.open('foo');"},
    {"file:///C:/foo/bar.txt", "http", NULL, NULL, "www.google.com", "99", "/foo","search", "ref", "http://www.google.com:99/foo?search#ref"},
#ifdef WIN32
    {"http://www.google.com/foo/bar.html?foo#bar", "file", "", "", "", "", "c:\\", "", "", "file:///C:/"},
#endif
#ifdef FULL_FILESYSTEM_URL_SUPPORT
    {"filesystem:http://www.google.com/foo/bar.html?foo#bar", NULL, NULL, NULL, NULL, NULL, "/", "", "", "filesystem:http://www.google.com/foo/"},
#endif
  };

  for (size_t i = 0; i < ARRAYSIZE(replace_cases); i++) {
    const ReplaceCase& cur = replace_cases[i];
    GURL url(cur.base);
    GURL::Replacements repl;
    SetupReplacement(&GURL::Replacements::SetScheme, &repl, cur.scheme);
    SetupReplacement(&GURL::Replacements::SetUsername, &repl, cur.username);
    SetupReplacement(&GURL::Replacements::SetPassword, &repl, cur.password);
    SetupReplacement(&GURL::Replacements::SetHost, &repl, cur.host);
    SetupReplacement(&GURL::Replacements::SetPort, &repl, cur.port);
    SetupReplacement(&GURL::Replacements::SetPath, &repl, cur.path);
    SetupReplacement(&GURL::Replacements::SetQuery, &repl, cur.query);
    SetupReplacement(&GURL::Replacements::SetRef, &repl, cur.ref);
    GURL output = url.ReplaceComponents(repl);

    EXPECT_EQ(replace_cases[i].expected, output.spec());
  }
}

TEST(GURLTest, PathForRequest) {
  struct TestCase {
    const char* input;
    const char* expected;
  } cases[] = {
    {"http://www.google.com", "/"},
    {"http://www.google.com/", "/"},
    {"http://www.google.com/foo/bar.html?baz=22", "/foo/bar.html?baz=22"},
    {"http://www.google.com/foo/bar.html#ref", "/foo/bar.html"},
    {"http://www.google.com/foo/bar.html?query#ref", "/foo/bar.html?query"},
#ifdef FULL_FILESYSTEM_URL_SUPPORT
    {"filesystem:http://www.google.com/temporary/foo/bar.html?query#ref", "/foo/bar.html?query"},
#endif
  };

  for (size_t i = 0; i < ARRAYSIZE(cases); i++) {
    GURL url(cases[i].input);
    std::string path_request = url.PathForRequest();
    EXPECT_EQ(cases[i].expected, path_request);
  }
}

TEST(GURLTest, EffectiveIntPort) {
  struct PortTest {
    const char* spec;
    int expected_int_port;
  } port_tests[] = {
    // http
    {"http://www.google.com/", 80},
    {"http://www.google.com:80/", 80},
    {"http://www.google.com:443/", 443},

    // https
    {"https://www.google.com/", 443},
    {"https://www.google.com:443/", 443},
    {"https://www.google.com:80/", 80},

    // ftp
    {"ftp://www.google.com/", 21},
    {"ftp://www.google.com:21/", 21},
    {"ftp://www.google.com:80/", 80},

    // gopher
    {"gopher://www.google.com/", 70},
    {"gopher://www.google.com:70/", 70},
    {"gopher://www.google.com:80/", 80},

    // file - no port
    {"file://www.google.com/", url_parse::PORT_UNSPECIFIED},
    {"file://www.google.com:443/", url_parse::PORT_UNSPECIFIED},

    // data - no port
    {"data:www.google.com:90", url_parse::PORT_UNSPECIFIED},
    {"data:www.google.com", url_parse::PORT_UNSPECIFIED},

#ifdef FULL_FILESYSTEM_URL_SUPPORT
    // filesystem - no port
    {"filesystem:http://www.google.com:90/t/foo", url_parse::PORT_UNSPECIFIED},
    {"filesystem:file:///t/foo", url_parse::PORT_UNSPECIFIED},
#endif
  };

  for (size_t i = 0; i < ARRAYSIZE(port_tests); i++) {
    GURL url(port_tests[i].spec);
    EXPECT_EQ(port_tests[i].expected_int_port, url.EffectiveIntPort());
  }
}

TEST(GURLTest, IPAddress) {
  struct IPTest {
    const char* spec;
    bool expected_ip;
  } ip_tests[] = {
    {"http://www.google.com/", false},
    {"http://192.168.9.1/", true},
    {"http://192.168.9.1.2/", false},
    {"http://192.168.m.1/", false},
    {"http://2001:db8::1/", false},
    {"http://[2001:db8::1]/", true},
    {"", false},
    {"some random input!", false},
  };

  for (size_t i = 0; i < ARRAYSIZE(ip_tests); i++) {
    GURL url(ip_tests[i].spec);
    EXPECT_EQ(ip_tests[i].expected_ip, url.HostIsIPAddress());
  }
}

TEST(GURLTest, HostNoBrackets) {
  struct TestCase {
    const char* input;
    const char* expected_host;
    const char* expected_plainhost;
  } cases[] = {
    {"http://www.google.com", "www.google.com", "www.google.com"},
    {"http://[2001:db8::1]/", "[2001:db8::1]", "2001:db8::1"},
    {"http://[::]/", "[::]", "::"},

    // Don't require a valid URL, but don't crash either.
    {"http://[]/", "[]", ""},
    {"http://[x]/", "[x]", "x"},
    {"http://[x/", "[x", "[x"},
    {"http://x]/", "x]", "x]"},
    {"http://[/", "[", "["},
    {"http://]/", "]", "]"},
    {"", "", ""},
  };
  for (size_t i = 0; i < ARRAYSIZE(cases); i++) {
    GURL url(cases[i].input);
    EXPECT_EQ(cases[i].expected_host, url.host());
    EXPECT_EQ(cases[i].expected_plainhost, url.HostNoBrackets());
  }
}

TEST(GURLTest, DomainIs) {
  const char google_domain[] = "google.com";

  GURL url_1("http://www.google.com:99/foo");
  EXPECT_TRUE(url_1.DomainIs(google_domain));

  GURL url_2("http://google.com:99/foo");
  EXPECT_TRUE(url_2.DomainIs(google_domain));

  GURL url_3("http://google.com./foo");
  EXPECT_TRUE(url_3.DomainIs(google_domain));

  GURL url_4("http://google.com/foo");
  EXPECT_FALSE(url_4.DomainIs("google.com."));

  GURL url_5("http://google.com./foo");
  EXPECT_TRUE(url_5.DomainIs("google.com."));

  GURL url_6("http://www.google.com./foo");
  EXPECT_TRUE(url_6.DomainIs(".com."));

  GURL url_7("http://www.balabala.com/foo");
  EXPECT_FALSE(url_7.DomainIs(google_domain));

  GURL url_8("http://www.google.com.cn/foo");
  EXPECT_FALSE(url_8.DomainIs(google_domain));

  GURL url_9("http://www.iamnotgoogle.com/foo");
  EXPECT_FALSE(url_9.DomainIs(google_domain));

  GURL url_10("http://www.iamnotgoogle.com../foo");
  EXPECT_FALSE(url_10.DomainIs(".com"));

#ifdef FULL_FILESYSTEM_URL_SUPPORT
  GURL url_11("filesystem:http://www.google.com:99/foo/");
  EXPECT_TRUE(url_11.DomainIs(google_domain));

  GURL url_12("filesystem:http://www.iamnotgoogle.com/foo/");
  EXPECT_FALSE(url_12.DomainIs(google_domain));
#endif
}

// Newlines should be stripped from inputs.
TEST(GURLTest, Newlines) {
  // Constructor.
  GURL url_1(" \t ht\ntp://\twww.goo\rgle.com/as\ndf \n ");
  EXPECT_EQ("http://www.google.com/asdf", url_1.spec());

  // Relative path resolver.
  GURL url_2 = url_1.Resolve(" \n /fo\to\r ");
  EXPECT_EQ("http://www.google.com/foo", url_2.spec());

  // Note that newlines are NOT stripped from ReplaceComponents.
}

TEST(GURLTest, IsStandard) {
  GURL a("http:foo/bar");
  EXPECT_TRUE(a.IsStandard());

  GURL b("foo:bar/baz");
  EXPECT_FALSE(b.IsStandard());

  GURL c("foo://bar/baz");
  EXPECT_FALSE(c.IsStandard());
}
