/** dtoa.h

    Print a floating point number at the precision necessary to preserve its
    binary representation, no more.
    
    Derived from dtoa.c.  See copyright and authorship information in that file.
*/

#pragma once

#include <string>

extern "C" {


    /*  Arguments ndigits, decpt, sign are similar to those
        of ecvt and fcvt; trailing zeros are suppressed from
        the returned string.  If not null, *rve is set to point
        to the end of the return value.  If d is +-Infinity or NaN,
        then *decpt is set to 9999.

        mode:
        0 ==> shortest string that yields d when read in
        and rounded to nearest.
        1 ==> like 0, but with Steele & White stopping rule;
        e.g. with IEEE P754 arithmetic , mode 0 gives
        1e23 whereas mode 1 gives 9.999999999999999e22.
        2 ==> max(1,ndigits) significant digits.  This gives a
        return value similar to that of ecvt, except
        that trailing zeros are suppressed.
        3 ==> through ndigits past the decimal point.  This
        gives a return value similar to that from fcvt,
        except that trailing zeros are suppressed, and
        ndigits can be negative.
        4,5 ==> similar to 2 and 3, respectively, but (in
        round-nearest mode) with the tests of mode 0 to
        possibly return a shorter string that rounds to d.
        With IEEE arithmetic and compilation with
        -DHonor_FLT_ROUNDS, modes 4 and 5 behave the same
        as modes 2 and 3 when FLT_ROUNDS != 1.
        6-9 ==> Debugging modes similar to mode - 4:  don't try
        fast floating-point estimate (if applicable).

        Values of mode other than 0-9 are treated as mode 0.

        Sufficient space is allocated to the return value
        to hold the suppressed trailing zeros.
    */

char *
soa_dtoa(double dd, int mode, int ndigits,
          int *decpt, int *sign, char **rve);

void
soa_freedtoa(char *s);

double
soa_strtod(const char *s00, char **se);

} // extern "C"

namespace Datacratic {

inline std::string dtoa(double floatVal)
{
    // Use dtoa to make sure we print a value that will be converted
    // back to the same on input, without printing more digits than
    // necessary.
    int decpt;
    int sign;

    char * result = soa_dtoa(floatVal, 1, -1 /* ndigits */,
                             &decpt, &sign, nullptr);
    std::string toReturn(result);
    soa_freedtoa(result);

    //cerr << "decpt = " << decpt << " sign = " << sign
    //     << " result = " << toReturn << " val = " << floatVal
    //     << endl;
    if (decpt > 0 && decpt <= toReturn.size())
        toReturn.insert(decpt, ".");
    else if (decpt == 9999)
        ;
    else if (decpt <= 0 && decpt > -6) {
        toReturn.insert(0, "0.");
        for (unsigned i = 0;  i < -decpt;  ++i)
            toReturn.insert(2, "0");
    }
    else {
        if (toReturn.size() > 1)
            toReturn.insert(1, ".");
        toReturn += "e" + std::to_string(decpt - 1);
    }
        
    if (sign)
        toReturn.insert(0, "-");

    if (toReturn.back() == '.') {
        toReturn.erase(toReturn.size()-1, 1);
    }

    return toReturn;
}


} // namespace Datacratic
