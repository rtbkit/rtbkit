/* tsne_python.cc
   Jeremy Barnes, 20 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Python wrapper for TSNE.
*/

#include <Python.h>
#include "numpy/arrayobject.h"
#include "tsne.h"
#include <iostream>
#include "arch/exception.h"


using namespace std;
using namespace ML;

// For the following: /usr/include/python2.6/numpy/__multiarray_api.h:958: error: ‘int _import_array()’ defined but not used
int (*fn) () = &_import_array;


static PyObject *
tsne_vectors_to_distances(PyObject *self, PyObject *args)
{
    PyObject * in_array;

    int args_ok = PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array);
    if (!args_ok)
        return NULL;

    /* Convert the argument to a float array. */
    PyObject * input_as_float32
        = PyArray_ContiguousFromAny(in_array, PyArray_FLOAT32, 2, 2);
    if (!input_as_float32) {
        cerr << "don't know what to do if PyArray_ContiguousFromAny fails"
             << endl;
        abort();
    }

    // TODO: inc reference?

    int n = PyArray_DIM(input_as_float32, 0);
    int d = PyArray_DIM(input_as_float32, 1);

    npy_intp npy_shape[2] = { n, n };

    /* Allocate an object (in memory) for the result */
    PyObject * result_array
        = PyArray_SimpleNew(2 /* num dims */,
                            npy_shape,
                            PyArray_FLOAT32);
    if (!result_array)
        return NULL;

    // TODO: inc reference?

    Py_BEGIN_ALLOW_THREADS;
    try {
        /* Copy into a boost multi array (TODO: avoid this copy) */
        boost::multi_array<float, 2> array(boost::extents[n][d]);

        const float * data_in = (const float *)PyArray_DATA(input_as_float32);

        std::copy(data_in, data_in + (n * d), array.data());

        boost::multi_array<float, 2> result
            = ML::vectors_to_distances(array);

        if (result.shape()[0] != n || result.shape()[1] != n)
            throw Exception("wrong shapes");

        float * data_out = (float *)PyArray_DATA(result_array);

        std::copy(result.data(), result.data() + n * n, data_out);

    } catch (const std::exception & exc) {
        cerr << "vectors_to_distances failed: convert exception "
             << exc.what() << endl;
        abort();
    } catch (...) {
        cerr << "vectors_to_distances failed: convert unknown exception"
             << endl;
        abort();
    }

    Py_END_ALLOW_THREADS;

    return result_array;
}

static PyObject *
tsne_distances_to_probabilities(PyObject *self, PyObject *args)
{
    Py_BEGIN_ALLOW_THREADS;
    Py_END_ALLOW_THREADS;

    return Py_BuildValue("i", 0);
}

static PyObject *
tsne_tsne(PyObject *self, PyObject *args)
{
    Py_BEGIN_ALLOW_THREADS;
    Py_END_ALLOW_THREADS;

    return Py_BuildValue("i", 0);
}

static PyMethodDef TsneMethods[] = {
    {"vectors_to_distances",  tsne_vectors_to_distances, METH_VARARGS,
     "Convert an array of vectors in a coordinate space to a symmetric square"
     "matrix of distances between them."},
    {"distances_to_probabilities",  tsne_distances_to_probabilities, METH_VARARGS,
     "Convert an array of distances between points to an array of joint "
     "probabilities by modelling as gaussians."},
    {"tsne",  tsne_tsne, METH_VARARGS,
     "reduce the (n x d) matrix to a (n x num_dims) matrix using t-SNE."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
inittsne(void)
{
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

