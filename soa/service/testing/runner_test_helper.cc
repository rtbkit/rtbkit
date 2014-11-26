#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <string>

#include "jml/arch/timers.h"
#include "jml/utils/exc_check.h"

using namespace std;


void
DoOutput(FILE * in, FILE * out)
{
    int len(0);
    unique_ptr<char> buffer;

    size_t n = ::fread(&len, sizeof(len), 1, in);
    ExcCheckNotEqual(n, 0, "sizeof(len) must be 4");

    buffer.reset(new char[len + 1]);
    n = ::fread(buffer.get(), sizeof(char), len, in);
    ExcCheckNotEqual(n, 0, "received 0 bytes");

    buffer.get()[len] = 0;

    // ::fprintf(stderr, "helper output: %s\n", buffer);

    ::fprintf(out, "%s\n", buffer.get());
}

void
DoSleep(FILE * in)
{
    char delay_buf[5]; /* in .1 secs units */
    delay_buf[4] = 0;

    size_t r = ::fread(&delay_buf, 1, 4, in);
    if (r != 4) {
        throw ML::Exception("wrong delay: " + to_string(r));
    }

    long delay = ::strtol(delay_buf, NULL, 16);
    ML::sleep(delay * 0.1);
}

void
DoExit(FILE * in, bool quiet)
{
    int code;

    size_t n = ::fread(&code, sizeof(code), 1, in);
    ExcCheckNotEqual(n, 0, "no exit code received");

    if (!quiet) {
        printf("helper: exit with code %d\n", code);
    }

    exit(code);
}

int main(int argc, char *argv[])
{
    /** commands:
        err/out|bytes8|nstring
        xit|code(int)
        abt
    */

    bool quiet(false);

    if (argc > 1) {
        if (strcmp(argv[1], "--quiet") == 0) {
            quiet = true;
        }
    }

    if (!quiet) {
        printf("helper: ready\n");
    }

    while (1) {
        char command[3];
        size_t n = ::fread(command, 1, sizeof(command), stdin);
        if (n < 3) {
            if (::feof(stdin)) {
                break;
            }
        }
        if (n == 0) {
            continue;
        }
        
        if (::strncmp(command, "err", 3) == 0) {
            DoOutput(stdin, stderr);
        }
        else if (::strncmp(command, "out", 3) == 0) {
            DoOutput(stdin, stdout);
        }
        else if (::strncmp(command, "slp", 3) == 0) {
            DoSleep(stdin);
        }
        else if (::strncmp(command, "xit", 3) == 0) {
            DoExit(stdin, quiet);
        }
        else if (::strncmp(command, "abt", 3) == 0) {
            ::abort();
        }
    }

    return 0;
}
