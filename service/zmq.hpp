/*
    Copyright (c) 2007-2011 iMatix Corporation
    Copyright (c) 2007-2011 Other contributors as noted in the AUTHORS file

    This file is part of 0MQ.

    0MQ is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    0MQ is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_HPP_INCLUDED__
#define __ZMQ_HPP_INCLUDED__

#include "zmq.h"

#include <cassert>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <iostream>

#include <thread>
#include <set>
#include "jml/utils/exc_assert.h"

namespace zmq
{

    typedef zmq_free_fn free_fn;
    typedef zmq_pollitem_t pollitem_t;

    class socket_t;

    class error_t : public std::exception
    {
    public:

        error_t () : errnum (zmq_errno ()) {}

        virtual const char *what () const throw ()
        {
            return zmq_strerror (errnum);
        }

        int num () const
        {
            return errnum;
        }

    private:

        int errnum;
    };

    inline int poll (zmq_pollitem_t *items_, int nitems_, long timeout_ = -1)
    {
        int rc = zmq_poll (items_, nitems_, timeout_);
        if (rc < 0)
            throw error_t ();
        return rc;
    }

    inline void device (int device_, void * insocket_, void* outsocket_)
    {
        int rc = zmq_device (device_, insocket_, outsocket_);
        if (rc != 0)
            throw error_t ();
    }

    class message_t : public zmq_msg_t
    {
        friend class socket_t;

    public:

        inline message_t ()
        {
            int rc = zmq_msg_init (this);
            if (rc != 0)
                throw error_t ();
        }

        inline message_t (size_t size_)
        {
            int rc = zmq_msg_init_size (this, size_);
            if (rc != 0)
                throw error_t ();
        }

        inline message_t (void *data_, size_t size_, free_fn *ffn_,
            void *hint_ = NULL)
        {
            int rc = zmq_msg_init_data (this, data_, size_, ffn_, hint_);
            if (rc != 0)
                throw error_t ();
        }

        inline message_t (const std::string & s)
        {
            int rc = zmq_msg_init_size (this, s.size());
            if (rc != 0)
                throw error_t ();
            std::copy(s.begin(), s.end(), (char *)data());
        }
        
        inline message_t (const message_t & other) noexcept
        {
            int rc = zmq_msg_init (this);
            if (rc == 0)
                rc = zmq_msg_copy(this, const_cast<message_t *>(&other));
            if (rc != 0)
                throw error_t ();
        }

        inline message_t (message_t && other) noexcept
        {
            int rc = zmq_msg_init (this);
            if (rc == 0)
                rc = zmq_msg_move(this, &other);
            if (rc != 0)
                throw error_t ();
        }

        inline message_t & operator = (const message_t & other) noexcept
        {
            int rc = zmq_msg_copy(this, const_cast<message_t *>(&other));
            if (rc != 0)
                throw error_t ();
            return *this;
        }

        inline message_t & operator = (message_t && other) noexcept
        {
            int rc = zmq_msg_move(this, &other);
            if (rc != 0)
                throw error_t ();
            return *this;
        }

        inline ~message_t ()
        {
            int rc = zmq_msg_close (this);
            assert (rc == 0);
            (void)rc;
        }

        inline void rebuild ()
        {
            int rc = zmq_msg_close (this);
            if (rc != 0)
                throw error_t ();
            rc = zmq_msg_init (this);
            if (rc != 0)
                throw error_t ();
        }

        inline void rebuild (size_t size_)
        {
            int rc = zmq_msg_close (this);
            if (rc != 0)
                throw error_t ();
            rc = zmq_msg_init_size (this, size_);
            if (rc != 0)
                throw error_t ();
        }

        inline void rebuild (void *data_, size_t size_, free_fn *ffn_,
            void *hint_ = NULL)
        {
            int rc = zmq_msg_close (this);
            if (rc != 0)
                throw error_t ();
            rc = zmq_msg_init_data (this, data_, size_, ffn_, hint_);
            if (rc != 0)
                throw error_t ();
        }

        inline void move (message_t *msg_)
        {
            int rc = zmq_msg_move (this, (zmq_msg_t*) msg_);
            if (rc != 0)
                throw error_t ();
        }

        inline void copy (message_t *msg_)
        {
            int rc = zmq_msg_copy (this, (zmq_msg_t*) msg_);
            if (rc != 0)
                throw error_t ();
        }

        inline char *data ()
        {
            return (char *)zmq_msg_data (this);
        }

        inline const char *data () const
        {
            return (const char *)zmq_msg_data (const_cast<message_t *>(this));
        }

        inline size_t size () const
        {
            return zmq_msg_size (const_cast<message_t *>(this));
        }

        std::string toString() const
        {
            return std::string(data(), data() + size());
        }

    private:

        //  Disable implicit message copying, so that users won't use shared
        //  messages (less efficient) without being aware of the fact.
        //message_t (const message_t&);
        //void operator = (const message_t&);
    };

    class context_t
    {
        friend class socket_t;

    public:

        inline context_t (int io_threads_)
        {
            ptr = zmq_init (io_threads_);
            if (ptr == NULL)
                throw error_t ();
        }

        inline ~context_t ()
        {
            // If this throws, it means that you forgot to destroy a zmq_socket_t object
            // created with this context before you closed the context.
            ExcAssertEqual(sockets.size(), 0);

            int rc = zmq_term (ptr);
            assert (rc == 0);
            (void)rc;
        }

        //  Be careful with this, it's probably only useful for
        //  using the C api together with an existing C++ api.
        //  Normally you should never need to use this.
        inline operator void* ()
        {
            return ptr;
        }
        
        void registerSocket(socket_t * sock)
        {
            std::unique_lock<std::mutex> guard(lock);
            ExcAssertEqual(sockets.count(sock), 0);
            sockets.insert(sock);
        }

        void unregisterSocket(socket_t * sock)
        {
            std::unique_lock<std::mutex> guard(lock);
            ExcAssertEqual(sockets.count(sock), 1);
            sockets.erase(sock);
        }

    private:

        void *ptr;

        context_t (const context_t&);
        void operator = (const context_t&);

        std::mutex lock;
        std::set<socket_t *> sockets;
    };

    class socket_t
    {
    public:
        context_t * context_;

        inline socket_t (context_t &context_, int type_)
            : context_(&context_)
        {
            ptr = zmq_socket (context_.ptr, type_);
            if (ptr == NULL)
                throw error_t ();

            context_.registerSocket(this);
        }

        inline ~socket_t ()
        {
            int linger = 0;
            setsockopt (ZMQ_LINGER, &linger, sizeof(linger));
            int rc = zmq_close (ptr);
            assert (rc == 0);
            (void)rc;

            context_->unregisterSocket(this);
        }

        inline operator void* ()
        {
            return ptr;
        }

        inline void setsockopt (int option_, const void *optval_,
            size_t optvallen_)
        {
            int rc = zmq_setsockopt (ptr, option_, optval_, optvallen_);
            if (rc != 0)
                throw error_t ();
        }

        inline void getsockopt (int option_, void *optval_,
            size_t *optvallen_)
        {
            int rc = zmq_getsockopt (ptr, option_, optval_, optvallen_);
            if (rc != 0)
                throw error_t ();
        }

        inline void bind (const std::string & addr_)
        {
            int rc = zmq_bind (ptr, addr_.c_str());
            if (rc != 0)
                throw error_t ();
        }

        inline void unbind (const std::string & addr_)
        {
            int rc = zmq_unbind (ptr, addr_.c_str());
            if (rc != 0)
                throw error_t ();
        }

        inline int tryUnbind (const std::string & addr_)
        {
            int rc = zmq_unbind (ptr, addr_.c_str());
            return rc;
        }

        inline void connect (const std::string & addr_)
        {
            int rc = zmq_connect (ptr, addr_.c_str());
            if (rc != 0)
                throw error_t ();
        }

        inline void disconnect (const std::string & addr_)
        {
            int rc = zmq_disconnect (ptr, addr_.c_str());
            if (rc != 0)
                throw error_t ();
        }

        inline int tryDisconnect (const std::string & addr_)
        {
            return zmq_disconnect (ptr, addr_.c_str());
        }

        inline bool send (message_t &msg_, int flags_ = 0)
        {
            int rc = zmq_sendmsg (ptr, &msg_, flags_);
            if (rc >= 0)
                return true;
            if (rc == -1 && zmq_errno () == EAGAIN)
                return false;
            throw error_t ();
        }

        inline bool send (message_t && msg_, int flags_ = 0)
        {
            int rc = zmq_sendmsg (ptr, &msg_, flags_);
            if (rc >= 0)
                return true;
            if (rc == -1 && zmq_errno () == EAGAIN)
                return false;
            throw error_t ();
        }

        inline bool recv (message_t *msg_, int flags_ = 0)
        {
            int rc = zmq_recvmsg (ptr, msg_, flags_);
            if (rc >= 0)
                return true;
            if (rc == -1 && zmq_errno () == EAGAIN)
                return false;
            throw error_t ();
        }

    private:

        void *ptr;

        socket_t (const socket_t&);
        socket_t (socket_t&&);
        void operator = (const socket_t&);
        void operator = (socket_t&&);
    };

}

#endif
