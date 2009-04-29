# Makefile for flayers functions
# Jeremy Barnes, 1 April 2006
# Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

LIBFLAYERS_SOURCES := \
        ArgvParser.cc \
        ArgvParserContainer.cc \
        ClassTopology.cc \
        fDataSet.cc \
        fLayersGeneral.cc \
        fTimeMeasurer.cc \
        IncBoostTrainer.cc \
        Layer.cc \
        LinearRegressor.cc \
        NNLayer.cc \
        OptTopology.cc \
        PreDefArgvParser.cc \
        Topology.cc \
        Trainer.cc \
        TrainerUtil.cc \
        TwoDSurfaceData.cc \
        Weights.cc \
        WeightsList.cc

LIBFLAYERS_LINK :=	utils db algebra arch

$(eval $(call library,flayers,$(LIBFLAYERS_SOURCES),$(LIBFLAYERS_LINK)))

#$(eval $(call include_sub_make,flayers_testing,testing))

$(eval $(call program,fexp,flayers,,tools))
