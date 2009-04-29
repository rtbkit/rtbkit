/*                                                                 -*- C++ -*- */
#ifndef FDATASET_H
#define FDATASET_H

#include "fLayersGeneral.h"
#include "stats/distribution.h"

///
///  \class DataSet DataSet.h "Core/DataSet.h" 
///  \brief DataSet class is the training,valid and test sample.
///
/// Abstract Class DataSet
/// you need to call init() in all constructors;(it should set input_ and targ_)
///
class DataSet{
public:
    DataSet(real* data,real* targets,uint size,uint iter_size,uint input_size,uint max_input_size,bool subsec=false)
        : probs(size, 1.0f)
    {
        set(data,targets,size,iter_size,input_size,max_input_size,subsec);
        i=0;
        sampling_prob=false;
    }

    DataSet(real* data,real* targets,uint size,uint input_size,bool subsec=false)
    {
        set(data,targets,size,size,input_size,input_size,subsec);
    }

    void set(real* data,real* targets,uint size,uint iter_size,uint input_size,uint max_input_size,bool subsec=false)
    {
        data_=data;
        targs_=targets;
        size_=size;
        iter_size_=iter_size;
        input_size_=input_size;
        max_input_size_=max_input_size;
        subsec_=subsec;
        if (iter_size>size)
            FERR("DataSet(...): iter_size < size");
        if (max_input_size<input_size)
            FERR("DataSet(...): max_input_size < input_size");
    }

    /// pointer into data representing next input
    ML::distribution<float> input_;

    /// pointer into targets representing next target
    ML::distribution<float> targ_;

    /// probability;init to 1
    ML::distribution<float> probs;

    /// # of current example
    uint i;

    /// real size, # of examples
    uint size_; 

    ///iteration size, usefull when dataset is big and you want that an iteration is << size_ ; default=size_
    uint iter_size_; 

    /// nb of el. in inputs
    uint input_size_;

    /// max nb of el. in inputs; usefull if you want to add input (ex:cascade-correlation)
    uint max_input_size_;

    /// pointer to sequetial inputs
    real* data_;

    /// pointer to sequential targets
    real* targs_;

    /// indicator if it is a subsection of another DataSet
    bool subsec_;

    ///active sampling probability;see Next()
    bool sampling_prob;

    ///current jump counter
    uint current_jump_counter;

    virtual DataSet* getSubSet(uint start,uint size,uint iter_size)=0;

    ///init input_ and targ_ pointer to first position in dataset
    virtual void init()=0;

    ///reduce sampling probability of current example
    void reduceSamplingProb_i(){probs[i]=0;};

    ///call next to consider sampling prob mode
    ML::distribution<real> Next()
    {
        if(sampling_prob)
            return probSamplingNext();
        else return next();
    }

    ///set input_ and targ_ pointer to the next value
    virtual ML::distribution<float> next() = 0;

    ///probability sampling next
    ML::distribution<float> probSamplingNext()
    {
        current_jump_counter=0;

        next();

        while(probs[i]==0.0) {
            next();
            current_jump_counter++;
        }

        return input_;
    }
}


///
///  \class SeqDataSet SeqDataSet.h "Core/SeqDataSet.h" 
///  \brief SeqDataSet class represent a sequential DataSet.
///
class SeqDataSet:public DataSet {
protected:
    ///reference to the last input
    real* last_input_;
    bool nextInput()
    {
        if (input_ == last_input_) {
            i=0;
            input_=data_;
            return true;
        }
        else {
            input_+=max_input_size_;
            i++;
        };
        return false;
    }

public:
    SeqDataSet(real* data,real* targets,uint size,uint iter_size,uint input_size,uint max_input_size,bool subsec=false)
        : DataSet(data,targets,size,iter_size,input_size,max_input_size,subsec)
    {
        last_input_=NULL;
        if(!subsec_)init();
    };

    SeqDataSet(real* data,real* targets,uint size,uint input_size,bool subsec=false)
        :DataSet(data,targets,size,size,input_size,input_size,subsec)
    {last_input_=NULL;if(!subsec_)init();};
    virtual DataSet* getSubSet(uint start,uint size,uint iter_size);
    virtual void init(){last_input_=data_+((size_-1)*max_input_size_);targ_=targs_;input_=data_;};
    virtual real* next(){if(nextInput())targ_=targs_;else targ_++;return input_;};
    virtual ~SeqDataSet(){};
};

///
///  \class RandomDataSet RandomDataSet.h "Core/RandomDataSet.h" 
///  \brief RandomDataSet class represent an Random DataSet.
///
class RandomDataSet:public DataSet {
public:
    typedef DataSet inherited;
    //random index table
    uint *table_;
    RandomDataSet(real* data,real* targets,uint size,uint iter_size,uint input_size,uint max_input_size,bool subsec=false)
        :DataSet(data,targets,size,iter_size,input_size,max_input_size,subsec)
    {if (!subsec_)initTable();if(!subsec_)init();};
    RandomDataSet(real* data,real* targets,uint size,uint input_size,bool subsec=false)
        :DataSet(data,targets,size,size,input_size,input_size,subsec)
    {if (!subsec_)initTable();if(!subsec_)init();};
    virtual DataSet* getSubSet(uint start,uint size,uint iter_size);
    virtual void init(){input_=data_+table_[0]*max_input_size_;targ_=targs_+table_[0];i=0;};
    virtual real* next(){if((i+1)<size_){i++;input_=data_+(table_[i]*max_input_size_);targ_=targs_+table_[i];}else init();return input_;};
    virtual ~RandomDataSet(){if(subsec_)free(table_);table_=NULL;};
    void printTable();
    void initTable();
};

std::ostream& operator<<(std::ostream& out,DataSet& ds);

#endif
