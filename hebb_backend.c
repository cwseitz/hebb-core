#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "lif.h"

static PyMethodDef RNNMethods[] = {
    {"lif", lif, METH_VARARGS, "Python interface for lif network in C"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef hebb_backend = {
    PyModuleDef_HEAD_INIT,
    "hebb_backend",
    "Python interface for IF network in C",
    -1,
    RNNMethods
};

PyMODINIT_FUNC PyInit_hebb_backend(void) {
    import_array();
    return PyModule_Create(&hebb_backend);
}
