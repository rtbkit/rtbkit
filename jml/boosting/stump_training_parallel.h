/* stump_training_parallel.h                                       -*- C++ -*-
   Jeremy Barnes, 18 March 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.
   $Source$

   Parallelized training for stumps.
*/

#ifndef __boosting__stump_training_parallel_h__
#define __boosting__stump_training_parallel_h__


#include "jml/utils/worker_task.h"
#include "jml/utils/guard.h"
#include <boost/bind.hpp>

namespace ML {

using std::cerr;
using std::endl;

template<class W, class Z, class Tracer=No_Trace>
struct Stump_Trainer_Parallel
  : public Stump_Trainer<W, Z, Tracer>, boost::noncopyable {
    explicit Stump_Trainer_Parallel(Worker_Task & worker)
        : worker(worker)
    {
    }

    Stump_Trainer_Parallel(const Tracer & tracer, Worker_Task & worker)
        : Stump_Trainer<W, Z, Tracer>(tracer), worker(worker)
    {
    }

    Worker_Task & worker;
    using Stump_Trainer<W, Z, Tracer>::tracer;

    /** A job that trains a single feature. */
    template<class Results, class Weights, class Examples>
    struct Test_Feature_Job {
        Test_Feature_Job(Feature feature, const Training_Data & data,
                         Feature predicted, const Weights & weights,
                         const Examples & examples, const W & default_w,
                         Results & results, const Stump_Trainer_Parallel & trainer,
                         float & score)
            : feature(feature), data(data), predicted(predicted),
              weights(weights), examples(examples), default_w(default_w),
              results(results), trainer(trainer), score(score)
        {
        }

        Feature feature;
        const Training_Data & data;
        Feature predicted;
        const Weights & weights;
        const Examples & examples;
        const W & default_w;
        Results & results;
        const Stump_Trainer_Parallel & trainer;
        float & score;

        void operator () ()
        {
            score = trainer.test(feature, data, predicted, weights, examples,
                                 default_w, results);
        }
    };

    /** A job that trains all features. */
    template<class Results, class Weights, class Examples>
    struct Test_All_Job {
        ~Test_All_Job()
        {
            //cerr << "destroying Test_All_Job at " << this << " with "
            //     << feature_scores_ptr.use_count() << " references "
            //     << feature_scores_ptr << endl;
        }

        std::vector<Feature> & features;
        const Training_Data & data;
        const Feature & predicted;
        const Weights & weights;
        const Examples & examples;
        Results & results;
        const Stump_Trainer_Parallel & trainer;
        boost::function<void ()> next;
        W default_w;
        std::shared_ptr<std::vector<std::pair<int, float> > > feature_scores_ptr;
        int group;

        Test_All_Job(const Test_All_Job & other)
            : features(other.features), data(other.data),
              predicted(other.predicted), weights(other.weights),
              examples(other.examples), results(other.results),
              trainer(other.trainer), next(other.next), default_w(other.default_w),
              feature_scores_ptr(other.feature_scores_ptr)
        {
            //cerr << "weights[0][0] at " << &weights[0][0] << endl;
            //cerr << "copying Test_All_Job from " << &other << " to "
            //     << this << endl;
        }

        /* Set up the job. */
        Test_All_Job(std::vector<Feature> & features,
                     const Training_Data & data,
                     const Feature & predicted,
                     const Weights & weights,
                     const Examples & examples,
                     Results & results,
                     const Stump_Trainer_Parallel & trainer,
                     boost::function<void ()> next,
                     int parent = -1)
            : features(features), data(data), predicted(predicted),
              weights(weights), examples(examples), results(results),
              trainer(trainer), next(next),
              default_w(trainer.calc_default_w(data, predicted, examples,
                                               weights))
        {
            //cerr << "weights[0][0] at " << &weights[0][0] << endl;
            //cerr << "creating Test_All_Job at " << this << endl;
            
            using namespace std;


            if (trainer.tracer) {
                trainer.tracer("stump training", 1)
                    << "test all: " << features.size() << " features" << std::endl;
                trainer.tracer("stump training", 2)
                    << "default w: " << std::endl
                    << default_w.print() << std::endl;
            }

            feature_scores_ptr.reset(new std::vector<std::pair<int, float> >());
            std::vector<std::pair<int, float> > & feature_scores
                = *feature_scores_ptr;
            feature_scores.resize(features.size());

            /* Get our task group. */
            group = trainer.worker.get_group
                (*this, format("Stump Test_All_Job under %d", parent), parent);
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(trainer.worker),
                                         group));
            
            for (unsigned i = 0;  i < features.size();  ++i) {
                //cerr << "adding feature " << i << std::endl;
                feature_scores[i].first = i;
                trainer.worker.add
                    (Test_Feature_Job<Results, Weights, Examples>
                     (features[i], data, predicted, weights, examples,
                      default_w, results, trainer, feature_scores[i].second),
                     format("stump Test_Feature_Job for %s under %d",
                            data.feature_space()->print(features[i]).c_str(), group),
                     group);
            }

            //cerr << "feature_scores_ptr = " << feature_scores_ptr << endl;
        }
        
        /* Function that gets called at the end of the job.  It records the
           results and calls the next job.*/
        void operator () ()
        {
            std::vector<std::pair<int, float> > & feature_scores
                = *feature_scores_ptr;

            //cerr << "finished job " << this << endl;
            //cerr << "features = " << &features << endl;
            //cerr << "features.size() = " << features.size() << endl;
            //cerr << "feature_scores_ptr = " << feature_scores_ptr << endl;
            //cerr << "feature_scores = " << &feature_scores << endl;
            //cerr << "feature_scores.size() = " << feature_scores.size() << endl;

            sort_on_second_ascending(feature_scores);
            std::vector<Feature> new_features;
            new_features.reserve(features.size());
            
            // TODO: z_none?

            for (unsigned i = 0;  i < features.size();  ++i)
                new_features.push_back(features[feature_scores[i].first]);
            
            features.swap(new_features);

            //cerr << "calling next" << endl;

            if (next) next();
        }
    };
};


} // namespace ML


#endif /* __boosting__stump_training_parallel_h__ */
