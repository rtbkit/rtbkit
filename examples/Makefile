##
## This simple makefiles allows to build RTBkit samples
## 
CXX       = g++ -pthread
CPPFLAGS  = -I$(RTBKIT_HOME)/include 
CPPFLAGS += -I$(RTBKIT_HOME)/include/rtbkit/
CPPFLAGS += -I$(RTBKIT_HOME)/include/rtbkit/leveldb/include
CXXFLAGS += -std=c++0x -ggdb -Wno-deprecated-declarations

LDFLAGS   = -Wl,--rpath-link,$(RTBKIT_HOME)/lib -L$(RTBKIT_HOME)/lib 
LDFLAGS  += -Wl,--copy-dt-needed-entries
LDFLAGS  += -Wl,--no-as-needed
LIBS      = -lexception_hook -ltcmalloc -ldl 


EXECS     = multi_agent  data_logger_ex \
            bidding_agent_console bidding_agent_ex \
            bid_request_endpoint \
            adserver_endpoint \
            integration_endpoints

all: $(EXECS)

multi_agent: multi_agent.o
	$(CXX) -o multi_agent multi_agent.o $(LDFLAGS) $(LIBS) -lrtb_router -lbidding_agent -lboost_program_options -lservices

data_logger_ex: data_logger_ex.o
	$(CXX) -o data_logger_ex data_logger_ex.o $(LDFLAGS) $(LIBS) -ldata_logger -lboost_program_options -lservices

bidding_agent_console: bidding_agent_console.o
	$(CXX) -o bidding_agent_console bidding_agent_console.o $(LDFLAGS) $(LIBS) -lbidding_agent -lrtb_router -lboost_program_options -lservices

bidding_agent_ex: bidding_agent_ex.o
	$(CXX) -o bidding_agent_ex bidding_agent_ex.o $(LDFLAGS) $(LIBS) -lbidding_agent -lrtb_router -lboost_program_options -lservices

bid_request_endpoint: bid_request_endpoint.o
	$(CXX) -o bid_request_endpoint bid_request_endpoint.o $(LDFLAGS) $(LIBS) -lbidding_agent -lexchange -lrtb_router -lbid_request -lboost_program_options -lservices

adserver_endpoint: adserver_endpoint.o
	$(CXX) -o adserver_endpoint adserver_endpoint.o $(LDFLAGS) $(LIBS) -lstandard_adserver -ldata_logger -lrtb_router -lbidding_agent -lboost_program_options -lservices

integration_endpoints: integration_endpoints.o
	$(CXX) -o integration_endpoints integration_endpoints.o $(LDFLAGS) $(LIBS) -lexchange -lstandard_adserver -ldata_logger -lrtb_router -lbidding_agent -lboost_program_options -lservices


clean:
	$(RM) $(EXECS) *.o
