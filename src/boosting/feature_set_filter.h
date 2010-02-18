/* feature_set_filter.h                                            -*- C++ -*-
   Jeremy Barnes, 27 August 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

   A filter for feature sets.
*/

#ifndef __element_tools__feature_set_filter_h__
#define __element_tools__feature_set_filter_h__

#include "dense_features.h"
#include "feature_set.h"
#include "jml/utils/parse_context.h"

namespace ML {

struct Feature_Set_Filter {

    enum Op {
        OP_EQ,
        OP_NE,
        OP_GT,
        OP_GE,
        OP_LT,
        OP_LE
    };

    bool operator () (const Feature_Set & fs) const
    {
        return empty || apply(fs[feat], op, val);
    }

    static bool apply(float val1, Op op, float val2)
    {
        switch (op) {
        case OP_EQ:  return (val1 == val2);
        case OP_NE: return (val1 != val2);
        case OP_GT:  return (val1 >  val2);
        case OP_GE:  return (val1 >= val2);
        case OP_LT:  return (val1 <  val2);
        case OP_LE:  return (val1 <= val2);
        default:
            throw Exception("Feature_Set_Filter::apply(): unknown op");
        }
    }

    void parse(const std::string & value,
               const Dense_Feature_Space & fs)
    {
        Parse_Context context(value, value.c_str(),
                              value.c_str() + value.size());

        context.skip_whitespace();

        if (context.eof()) {
            empty = true;
            return;
        }
        empty = false;

        fs.expect(context, feat);
        context.skip_whitespace();
        if (context.match_literal('=')) {
            context.expect_literal('=');
            op = OP_EQ;
        }
        else if (context.match_literal('!')) {
            context.expect_literal('=');
            op = OP_NE;
        }
        else if (context.match_literal('>')) {
            if (context.match_literal('=')) op = OP_GE;
            else op = OP_GT;
        }
        else if (context.match_literal('<')) {
            if (context.match_literal('=')) op = OP_LE;
            else op = OP_LT;
        }
        else context.exception("expected binary operator");

        context.skip_whitespace();

        val = context.expect_float();

        context.skip_whitespace();

        context.expect_eof();

        //cerr << "got filter: " << fs.print(feat) <<  " " << op << " "
        //     << val << endl;
    }

    bool empty;
    Feature feat;
    Op op;
    float val;
};

} // namespace ML


#endif /* __element_tools__feature_set_filter_h__ */

