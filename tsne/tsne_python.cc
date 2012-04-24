/* tsne_python.cc
   Jeremy Barnes, 20 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Python wrapper for TSNE.
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "tsne.h"
#include <iostream>
#include "jml/arch/exception.h"
#include "jml/utils/unnamed_bool.h"
#include "jml/arch/demangle.h"
#include <cxxabi.h>
#include <typeinfo>
#include <boost/bind.hpp>


using namespace std;
using namespace ML;

// For the following: /usr/include/python2.6/numpy/__multiarray_api.h:958: error: ‘int _import_array()’ defined but not used
int (*fn) () = &_import_array;

template<class Object>
struct PyRef {
    PyRef(Object * obj)
        : obj(obj)
    {
    }

    template<class Other>
    PyRef(Other * obj)
        : obj(reinterpret_cast<Object *>(obj))
    {
    }

    Object * obj;

    JML_IMPLEMENT_OPERATOR_BOOL(obj);

    template<class ObjectOut>
    operator ObjectOut * () const { return reinterpret_cast<ObjectOut *>(obj); }

    Object * release()
    {
        Object * result = obj;
        obj = 0;
        return result;
    }

    template<class ObjectOut>
    ObjectOut * release()
    {
        ObjectOut * result = *this;
        obj = 0;
        return result;
    }

    ~PyRef()
    {
        if (obj) Py_DECREF(obj);
    }
};

struct Interrupt_Exception : public Exception {
    Interrupt_Exception()
        : Exception("Keyboard Interrupt")
    {
    }
};

typedef PyRef<PyObject> PyObjectRef;
typedef PyRef<PyArrayObject> PyArrayRef;

PyObject * to_python_exception(const std::type_info * exc_type,
                               const char * what = 0)
{
    if (*exc_type == typeid(Interrupt_Exception)) {
        PyErr_SetString(PyExc_KeyboardInterrupt, "JML processing interrupted");
        return NULL;
    }

    std::string message;
    if (what)
        message = format("JML Exception of type %s caught: %s",
                         demangle(exc_type->name()).c_str(),
                         what);
    else
        message = format("JML Exception of type %s caught",
                         demangle(exc_type->name()).c_str());

    PyErr_SetString(PyExc_RuntimeError, message.c_str());
    
    return NULL;
}

PyObject * to_python_exception(const std::exception & exc)
{
    return to_python_exception(&typeid(exc), exc.what());
}

PyObject * to_python_exception()
{
    const std::type_info * exc_type = abi::__cxa_current_exception_type();
    if (!exc_type) {
        cerr << "exception type was null" << endl;
        abort();
    }

    return to_python_exception(exc_type);
}

enum BlockOp {
    UNBLOCK,
    BLOCK
};

volatile int num_signals_received = 0;

void handle_sigint(int action)
{
    PyErr_SetInterrupt();
    ++num_signals_received;
}

struct PyThreads {
    PyThreads(BlockOp op = UNBLOCK)
        : _save(0), old_sigint(0)
    {
        if (op == UNBLOCK)
            unblock();
    }

    ~PyThreads()
    {
        if (_save)
            block();
    }

    PyThreadState *_save;
    PyOS_sighandler_t old_sigint;
    int signals_before;

    void unblock()
    {
        if (_save) {
            cerr << "threads were already unblocked; pairing error"
                 << endl;
            abort();
        }
        
        old_sigint = PyOS_setsig(SIGINT, handle_sigint);
        signals_before = num_signals_received;
        Py_UNBLOCK_THREADS;
    }

    void block()
    {
        if (!_save) {
            cerr << "threads were already unblocked; pairing error"
                 << endl;
            abort();
        }
            
        Py_BLOCK_THREADS;
        PyOS_setsig(SIGINT, old_sigint);
    }

    bool interrupted()
    {
        return num_signals_received > signals_before;
    }
};

static PyObject *
tsne_vectors_to_distances(PyObject *self, PyObject *args)
{
    PyObject * in_array;

    int args_ok = PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array);
    if (!args_ok)
        return NULL;
    
    /* Convert the argument to a float array. */
    PyArrayRef input_as_float32
        = PyArray_FromAny(in_array,
                          PyArray_DescrFromType(NPY_FLOAT32),
                          2, 2,
                          NPY_C_CONTIGUOUS | NPY_FORCECAST | NPY_ALIGNED,
                          0);
    if (!input_as_float32)
        return NULL;
    
    // TODO: inc reference?

    int n = PyArray_DIM(input_as_float32, 0);
    int d = PyArray_DIM(input_as_float32, 1);

    npy_intp npy_shape[2] = { n, n };

    /* Allocate an object (in memory) for the result */
    PyArrayRef result_array
        = PyArray_SimpleNew(2 /* num dims */,
                            npy_shape,
                            NPY_FLOAT);
    if (!result_array)
        return NULL;

    // TODO: inc reference?

    try {
        PyThreads threads(UNBLOCK);

        /* Copy into a boost multi array (TODO: avoid this copy) */
        boost::multi_array<float, 2> array(boost::extents[n][d]);

        const float * data_in = (const float *)PyArray_DATA(input_as_float32);

        std::copy(data_in, data_in + (n * d), array.data());

        input_as_float32.release();

        boost::multi_array<float, 2> result
            = ML::vectors_to_distances(array);

        if (result.shape()[0] != n || result.shape()[1] != n)
            throw Exception("wrong shapes");

        float * data_out = (float *)PyArray_DATA(result_array);

        std::copy(result.data(), result.data() + n * n, data_out);
    } catch (const std::exception & exc) {
        return to_python_exception(exc);
    } catch (...) {
        return to_python_exception();
    }

    return result_array.release<PyObject>();
}

static PyObject *
tsne_distances_to_probabilities(PyObject *self, PyObject *args)
{
    PyObject * in_array;
    double tolerance;
    double perplexity;

    int args_ok = PyArg_ParseTuple(args, "O!dd", &PyArray_Type, &in_array,
                                   &tolerance, &perplexity);
    if (!args_ok)
        return NULL;

    /* Convert the argument to a float array. */
    PyArrayRef input_as_float32
        = PyArray_FromAny(in_array,
                          PyArray_DescrFromType(NPY_FLOAT32),
                          2, 2,
                          NPY_C_CONTIGUOUS | NPY_FORCECAST | NPY_ALIGNED,
                          0);
    if (!input_as_float32)
        return NULL;
    
    int n = PyArray_DIM(input_as_float32, 0);
    int n2 = PyArray_DIM(input_as_float32, 1);

    if (n != n2) {
        PyErr_SetString(PyExc_RuntimeError, "array must be square");
        return NULL;
    }

    /* Allocate an object (in memory) for the result */
    npy_intp npy_shape[2] = { n, n };
    PyArrayRef result_array
        = PyArray_SimpleNew(2 /* num dims */,
                            npy_shape,
                            NPY_FLOAT);
    if (!result_array)
        return NULL;

    try {
        PyThreads threads(UNBLOCK);

        /* Copy into a boost multi array (TODO: avoid this copy) */
        boost::multi_array<float, 2> array(boost::extents[n][n]);

        const float * data_in = (const float *)PyArray_DATA(input_as_float32);

        std::copy(data_in, data_in + (n * n), array.data());

        input_as_float32.release();

        boost::multi_array<float, 2> result
            = distances_to_probabilities(array,
                                         tolerance,
                                         perplexity);
        
        if (result.shape()[0] != n || result.shape()[1] != n)
            throw Exception("wrong shapes");

        float * data_out = (float *)PyArray_DATA(result_array);

        std::copy(result.data(), result.data() + n * n, data_out);
    } catch (const std::exception & exc) {
        return to_python_exception(exc);
    } catch (...) {
        return to_python_exception();
    }

    return result_array.release<PyObject>();
}

bool tsne_callback(int signals_before, int, float, const char *)
{
    return signals_before == num_signals_received;
}

static PyObject *
tsne_tsne(PyObject *self, PyObject *args, PyObject * kwds)
{
    PyObject * in_array;
    TSNE_Params params;

    int num_dims = 2;

    static const char * const kwlist[] =
        { "array", "num_dims", "max_iter", "initial_momentum", "final_momentum",
          "eta", "min_gain", "min_prob", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                     "O!|iiddddd", (char **)kwlist,
                                     &PyArray_Type, &in_array,
                                     &num_dims,
                                     &params.max_iter,
                                     &params.initial_momentum,
                                     &params.final_momentum,
                                     &params.eta,
                                     &params.min_gain,
                                     &params.min_prob))
        return NULL;

    /* Convert the input array to a float array. */
    PyArrayRef input_as_float32
        = PyArray_FromAny(in_array,
                          PyArray_DescrFromType(NPY_FLOAT32),
                          2, 2,
                          NPY_C_CONTIGUOUS | NPY_FORCECAST | NPY_ALIGNED,
                          0);
    if (!input_as_float32)
        return NULL;
    
    int n = PyArray_DIM(input_as_float32, 0);
    int n2 = PyArray_DIM(input_as_float32, 1);

    if (n != n2) {
        PyErr_SetString(PyExc_RuntimeError, "array must be square");
        return NULL;
    }

    /* Allocate an object (in memory) for the result */
    npy_intp npy_shape[2] = { n, num_dims };
    PyArrayRef result_array
        = PyArray_SimpleNew(2 /* num dims */,
                            npy_shape,
                            NPY_FLOAT);
    if (!result_array)
        return NULL;
    
    try {
        PyThreads threads(UNBLOCK);

        /* Copy into a boost multi array (TODO: avoid this copy) */
        boost::multi_array<float, 2> array(boost::extents[n][n]);

        const float * data_in = (const float *)PyArray_DATA(input_as_float32);

        std::copy(data_in, data_in + (n * n), array.data());

        input_as_float32.release();

        boost::multi_array<float, 2> result
            = tsne(array, num_dims, params,
                   boost::bind(tsne_callback,
                               threads.signals_before,
                               _1, _2, _3));
        
        if (threads.interrupted())
            throw Interrupt_Exception();

        if (result.shape()[0] != n || result.shape()[1] != num_dims)
            throw Exception("wrong shapes");
        
        float * data_out = (float *)PyArray_DATA(result_array);
        
        std::copy(result.data(), result.data() + n * num_dims, data_out);
    } catch (const std::exception & exc) {
        return to_python_exception(exc);
    } catch (...) {
        return to_python_exception();
    }

    return result_array.release<PyObject>();
}

static PyMethodDef TsneMethods[] = {
    {"vectors_to_distances",  tsne_vectors_to_distances, METH_VARARGS,
     "Convert an array of vectors in a coordinate space to a symmetric square"
     "matrix of distances between them."},
    {"distances_to_probabilities",  tsne_distances_to_probabilities, METH_VARARGS,
     "Convert an array of distances between points to an array of joint "
     "probabilities by modelling as gaussians."},
    {"tsne",  (PyCFunction)tsne_tsne, METH_VARARGS | METH_KEYWORDS,
     "reduce the (n x d) matrix to a (n x num_dims) matrix using t-SNE."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

extern "C" {

PyMODINIT_FUNC
init_tsne(void)
{
    import_array();

    PyObject *m;

    m = Py_InitModule("_tsne", TsneMethods);
    if (m == NULL)
        return;

#if 0
    TsneError = PyErr_NewException("tsne.error", NULL, NULL);
    Py_INCREF(TsneError);
    PyModule_AddObject(m, "error", TsneError);
#endif
}

} // extern "C"
