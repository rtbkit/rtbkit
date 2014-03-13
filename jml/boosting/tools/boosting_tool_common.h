/* boosting_tool_common.h                                          -*- C++ -*-
   Jeremy Barnes, 6 June 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Common routines for the boosting tools.
*/

#ifndef __boosting_tool_common_h__
#define __boosting_tool_common_h__

#include "jml/stats/sparse_distribution.h"
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>



namespace ML {


class Classifier_Impl;
class Decoder_Impl;
class Training_Data;
class Feature;
class Feature_Space;
class Mutable_Feature_Space;
class Weight_Spec;
class Optimization_Info;


/** Calculate and print statistics of how a classifier went over a dataset. */
void calc_stats(const Classifier_Impl & current,
                const Optimization_Info & opt_info,
                const Decoder_Impl & prob,
                const Training_Data & data,
                int draw_graphs,
                bool dump_testing, int dump_confusion,
                bool by_group, const Feature & group_feature);

/** Calculate and print statistics of how a classifier went over a dataset, but
    done group by group. */
void calc_stats_by_group(const Classifier_Impl & current,
                         const Optimization_Info & opt_info,
                         const Decoder_Impl & prob,
                         const Training_Data & data, int draw_graphs,
                         bool dump_testing, int dump_confusion);

/** Calculate and print statistics of how good a regression was. */
void calc_stats_regression(const Classifier_Impl & current,
                           const Optimization_Info & opt_info,
                           const Decoder_Impl & prob,
                           const Training_Data & data, int draw_graphs,
                           bool dump_testing, int dump_confusion);

/** Clean the dataset by removing those examples which are aliased (where
    the features are the same). */
void remove_aliased_examples(Training_Data & data, const Feature & predicted,
                             int verbosity, bool profile);

/** Get a list of features and an index, from the full specifications. */
void do_features(const Training_Data & data,
                 std::shared_ptr<Mutable_Feature_Space> feature_space,
                 const std::string & predicted_name,
                 std::vector<std::string> ignore_features,
                 std::vector<std::string> optional_features,
                 int min_feature_count, int verbosity,
                 std::vector<Feature> & features,
                 Feature & predicted,
                 std::map<std::string, Feature> & feature_index,
                 std::vector<std::string> type_overrides
                     = std::vector<std::string>());

/** Output a null classifier for the given label count. */
void write_null_classifier(const std::string & filename,
                           const Feature & predicted,
                           std::shared_ptr<const Feature_Space> feature_space,
                           int verbosity);

/** Print stats about a dense dataset. */
void print_data_stats(const Training_Data & data);

void print_weight_spec(const std::vector<Weight_Spec> & weight_spec,
                       std::shared_ptr<Feature_Space> feature_space);

} // namespace ML


#endif /* __boosting_tool_common_h__ */
