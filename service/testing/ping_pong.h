struct PingConnectionHandler : public ConnectionHandler {
    PingConnectionHandler(std::string & errorWhere,
                          ACE_Semaphore & finished)
        : errorWhere(errorWhere), finished(finished), messages(0)
    {
        constructed = Date::now();
        //cerr << "creating ping handler" << endl;
    }

    ~PingConnectionHandler()
    {
        //cerr << "destructing ping handler" << endl;
    }

    std::string & errorWhere;
    ACE_Semaphore & finished;
    int messages;
    Date constructed;

    void doError(const std::string & error)
    {
        errorWhere = error;
    }

    void onGotTransport()
    {
        startReading();
        startWriting();
    }

    void handleInput()
    {
        //cerr << "ping got input" << endl;
        //cerr << Date::now().print(5)
        //     << " ping handle_input on " << fd << " messages = "
        //     << messages << endl;

        char buf[100] = "error";
        int res = recv(buf, 100, MSG_DONTWAIT);
        if (res != -1)
            buf[res] = 0;
        BOOST_CHECK_EQUAL(res, 4);
        if (res == -1)
            BOOST_CHECK_EQUAL(strerror(errno), "success");
        BOOST_CHECK_EQUAL(string(buf), string("Hi!!"));

        if (messages == 1000) {
            cerr << "did 1000 messages in "
                 << Date::now().secondsSince(constructed)
                 << " seconds" << endl;

            stopReading();
            stopWriting();
            finished.release();
            closeWhenHandlerFinished();
            return;
        }

        ++messages;
        if (messages % 100 == 0)
            cerr << messages << endl;

        startWriting();
    }

    void handleOutput()
    {
        //cerr << "ping got output" << endl;
        //cerr << Date::now().print(5)
        //     << " ping handle_output on " << fd << endl;

        int res = send("hello", 5, MSG_DONTWAIT | MSG_NOSIGNAL);
        BOOST_CHECK_EQUAL(res, 5);
        if (res == -1)
            BOOST_CHECK_EQUAL(strerror(errno), "success");
        stopWriting();
    }
};

struct PongConnectionHandler : public ConnectionHandler {
    PongConnectionHandler(std::string & errorWhere)
        : errorWhere(errorWhere)
    {
        //cerr << "creating pong handler" << endl;
    }

    ~PongConnectionHandler()
    {
        //cerr << "destructing pong handler" << endl;
    }

    std::string & errorWhere;

    void doError(const std::string & error)
    {
        errorWhere = error;
    }

    void onGotTransport()
    {
        //cerr << "pong handler on GotTransport: handle " << getHandle()
        //     << endl;
        startReading();
    }

    void handleInput()
    {
        //cerr << "pong handler on handleInput: handle " << getHandle()
        //     << endl;
        //cerr << Date::now().print(5)
        //     << " pong handle_input on " << fd << endl;

        char buf[] = "error";
        int res = recv(buf, sizeof(buf), MSG_DONTWAIT);

        if (res == 0) {
            closeWhenHandlerFinished();
            return;
        }
        if (res != -1)
            buf[res] = 0;
        BOOST_CHECK_EQUAL(res, 5);
        if (res == -1)
            BOOST_CHECK_EQUAL(strerror(errno), "success");
        BOOST_CHECK_EQUAL(buf, string("hello"));
        startWriting();
    }

    void handleOutput()
    {
        //cerr << "pong handler on handleOutput: handle " << getHandle()
        //     << endl;
        //cerr << Date::now().print(5)
        //     << " pong handle_output on " << fd << endl;
        
        int res = send("Hi!!", 4, MSG_DONTWAIT | MSG_NOSIGNAL);
        BOOST_CHECK_EQUAL(res, 4);
        if (res == -1)
            BOOST_CHECK_EQUAL(strerror(errno), "success");
        stopWriting();
    }
};

