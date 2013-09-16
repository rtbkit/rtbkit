#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jml/utils/exc_check.h"

void
DoOutput(FILE * in, FILE * out)
{
    int len(0);
    char *buffer;

    size_t n = ::fread(&len, sizeof(len), 1, in);
    ExcCheckNotEqual(n, 0, "sizeof(len) must be 4");

    buffer = (char *) ::malloc(len + 1);
    n = ::fread(buffer, sizeof(char), len, in);
    ExcCheckNotEqual(n, 0, "received 0 bytes");

    buffer[len] = 0;

    ::fprintf(out, "%s\n", buffer);
    ::free(buffer);
}

void
DoExit(FILE * in)
{
    int code;

    size_t n = ::fread(&code, sizeof(code), 1, in);
    ExcCheckNotEqual(n, 0, "no exit code received");

    printf("helper: exit with code %d\n", code);

    exit(code);
}

int main(int argc, char *argv[])
{
    /** commands:
        err/out|bytes8|nstring
        xit|code(int)
        abt
    */

    printf("helper: ready\n");
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
        else if (::strncmp(command, "xit", 3) == 0) {
            DoExit(stdin);
        }
        else if (::strncmp(command, "abt", 3) == 0) {
            ::abort();
        }
    }

    return 0;
}
