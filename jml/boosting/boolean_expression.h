/* boolean_expression.h                                            -*- C++ -*-
   Jeremy Barnes, 4 July 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.
   
   Class to represent and manipulate a boolean expression from a classifier.
*/

#ifndef __jml__boosting__boolean_expression_h__
#define __jml__boosting__boolean_expression_h__

#include <vector>

namespace ML {

class Feature_Space;

struct Predicate : public Split {
    Predicate(const Split & split, int polarity)
    {
        if (polarity == MISSING) {
            Split::operator = (split);
            this->polarity = polarity;
            // TODO: something different?
        }
        else {
            Split::operator = (split);
            this->polarity = polarity;
        }
    }

    int polarity;  // TRUE/FALSE/MISSING

    std::string print(const Feature_Space & fs) const
    {
        return Split::print(fs, polarity);
    }
};

inline std::string print_outcome(bool outcome)
{
    if (outcome) return "true";
    else return "false";
}

// Conjunction, AKA "and"
template<typename Outcome>
struct Conjunction {
    std::vector<std::shared_ptr<Predicate> > predicates;
    Outcome outcome;

    std::string print(const Feature_Space & fs) const
    {
        std::string result;
        if (predicates.size() > 0) {
            result = "(";
            for (unsigned i = 0;  i < predicates.size();  ++i) {
                if (i != 0) result += " AND ";
                result += predicates[i]->print(fs);
            }
            result += "): ";
        }

        result += print_outcome(outcome);
        return result;
    }
};

// Disjunction, AKA "or"
template<typename Outcome>
struct Disjunction {
    std::vector<std::shared_ptr<Conjunction<Outcome> > > predicates;
    std::shared_ptr<const Feature_Space> feature_space;

    std::string
    print() const
    {
        std::string result = "(   ";
        for (unsigned i = 0;  i < predicates.size();  ++i) {
            if (i != 0)
                result += "\n OR ";
            result += predicates[i]->print(*feature_space);
        }
        return result;
    }

    /** Transform into a new kind of outcome, possibly pruning expressions
        as we go. */
    template<typename NewOutcome>
    Disjunction<NewOutcome>
    transform(const boost::function<bool (Outcome outcome,
                                          NewOutcome & noutcome)> & fn)
        const
    {
        Disjunction<NewOutcome> result;
        result.feature_space = feature_space;

        for (unsigned i = 0;  i < predicates.size();  ++i) {
            NewOutcome newOutcome;
            if (fn(predicates[i]->outcome, newOutcome)) {
                std::shared_ptr<Conjunction<NewOutcome> >
                    newConj(new Conjunction<NewOutcome>());
                newConj->outcome = newOutcome;
                newConj->predicates = predicates[i]->predicates;
                result.predicates.push_back(newConj);
            }
        }

        return result;
    }
};


} // namespace ML


#endif /* jml__boosting__boolean_expression_h__ */

