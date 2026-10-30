#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "symnmf.h"

/* Numpy Array -> C Array */
static double **pyarray_to_c_matrix(PyArrayObject *array, int *rows, int *cols) {
    int i, j;
    double **mat;
    npy_intp *dims;
    double *data;

    if (PyArray_NDIM(array) != 2) {
        return NULL;
    }

    dims = PyArray_DIMS(array);
    *rows = (int)dims[0];
    *cols = (int)dims[1];

    mat = allocate_matrix(*rows, *cols);
    if (mat == NULL) {
        return NULL;
    }

    data = (double *)PyArray_DATA(array);

    for (i = 0; i < *rows; i++) {
        for (j = 0; j < *cols; j++) {
            mat[i][j] = data[i * (*cols) + j];
        }
    }

    return mat;
}

/* C Array -> Numpy Array */
static PyObject *c_matrix_to_pyarray(double **mat, int rows, int cols) {
    int i, j;
    PyObject *array_obj;
    PyArrayObject *array;
    npy_intp dims[2];
    double *data;

    dims[0] = rows;
    dims[1] = cols;

    array_obj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (array_obj == NULL) {
        return NULL;
    }

    array = (PyArrayObject *)array_obj;
    data = (double *)PyArray_DATA(array);

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            data[i * cols + j] = mat[i][j];
        }
    }

    return array_obj;
}

static PyObject *py_sym(PyObject *self, PyObject *args) {
    PyObject *x_obj;
    PyArrayObject *x_array;
    double **X, **A;
    int n, d;
    PyObject *result;

    if (!PyArg_ParseTuple(args, "O", &x_obj)) {
        return NULL;
    }

    x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL) {
        return NULL;
    }

    X = pyarray_to_c_matrix(x_array, &n, &d);
    Py_DECREF(x_array);

    if (X == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid input matrix");
        return NULL;
    }

    A = allocate_matrix(n, n);
    if (A == NULL) {
        free_matrix(X, n);
        PyErr_SetString(PyExc_MemoryError, "Allocation failed");
        return NULL;
    }

    sym(X, A, n, d);

    result = c_matrix_to_pyarray(A, n, n);

    free_matrix(X, n);
    free_matrix(A, n);

    return result;
}

static PyObject *py_ddg(PyObject *self, PyObject *args) {
    PyObject *x_obj;
    PyArrayObject *x_array;
    double **X, **A, **D;
    double *D_diag;
    int n, d, i, j;
    PyObject *result;

    (void)self;

    if (!PyArg_ParseTuple(args, "O", &x_obj)) {
        return NULL;
    }

    x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL) {
        return NULL;
    }

    X = pyarray_to_c_matrix(x_array, &n, &d);
    Py_DECREF(x_array);

    if (X == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid input matrix");
        return NULL;
    }

    A = allocate_matrix(n, n);
    D = allocate_matrix(n, n);
    D_diag = (double *)calloc(n, sizeof(double));

    if (A == NULL || D == NULL || D_diag == NULL) {
        free_matrix(X, n);
        free_matrix(A, n);
        free_matrix(D, n);
        free(D_diag);
        PyErr_SetString(PyExc_MemoryError, "Allocation failed");
        return NULL;
    }

    sym(X, A, n, d);
    ddg(A, D_diag, n);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                D[i][j] = D_diag[i];
            } else {
                D[i][j] = 0.0;
            }
        }
    }

    result = c_matrix_to_pyarray(D, n, n);

    free_matrix(X, n);
    free_matrix(A, n);
    free_matrix(D, n);
    free(D_diag);

    return result;
}

static PyObject *py_norm(PyObject *self, PyObject *args) {
    PyObject *x_obj;
    PyArrayObject *x_array;
    double **X, **A, **W;
    double *D_diag;
    int n, d;
    PyObject *result;

    (void)self;

    if (!PyArg_ParseTuple(args, "O", &x_obj)) {
        return NULL;
    }

    x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL) {
        return NULL;
    }

    X = pyarray_to_c_matrix(x_array, &n, &d);
    Py_DECREF(x_array);

    if (X == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid input matrix");
        return NULL;
    }

    A = allocate_matrix(n, n);
    W = allocate_matrix(n, n);
    D_diag = (double *)calloc(n, sizeof(double));

    if (A == NULL || W == NULL || D_diag == NULL) {
        free_matrix(X, n);
        free_matrix(A, n);
        free_matrix(W, n);
        free(D_diag);
        PyErr_SetString(PyExc_MemoryError, "Allocation failed");
        return NULL;
    }

    sym(X, A, n, d);
    ddg(A, D_diag, n);
    norm(A, D_diag, W, n);

    result = c_matrix_to_pyarray(W, n, n);

    free_matrix(X, n);
    free_matrix(A, n);
    free_matrix(W, n);
    free(D_diag);

    return result;
}

static PyObject *py_symnmf(PyObject *self, PyObject *args) {
    PyObject *w_obj, *h_obj;
    PyArrayObject *w_array, *h_array;
    double **W, **H;
    int w_rows = 0, w_cols = 0, h_rows = 0, h_cols = 0;
    int max_iter = 300;
    double epsilon = 1e-4;
    PyObject *result;

    (void)self;

    if (!PyArg_ParseTuple(args, "OO", &w_obj, &h_obj)) {
        return NULL;
    }

    w_array = (PyArrayObject *)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (w_array == NULL) {
        return NULL;
    }

    h_array = (PyArrayObject *)PyArray_FROM_OTF(h_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (h_array == NULL) {
        Py_DECREF(w_array);
        return NULL;
    }

    W = pyarray_to_c_matrix(w_array, &w_rows, &w_cols);
    H = pyarray_to_c_matrix(h_array, &h_rows, &h_cols);

    Py_DECREF(w_array);
    Py_DECREF(h_array);

    if (W == NULL || H == NULL) {
        free_matrix(W, w_rows);
        free_matrix(H, h_rows);
        PyErr_SetString(PyExc_ValueError, "Invalid input matrix");
        return NULL;
    }

    if (w_rows != w_cols || w_rows != h_rows) {
        free_matrix(W, w_rows);
        free_matrix(H, h_rows);
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        return NULL;
    }

    update_H(W, H, h_rows, h_cols, max_iter, epsilon);

    result = c_matrix_to_pyarray(H, h_rows, h_cols);

    free_matrix(W, w_rows);
    free_matrix(H, h_rows);

    return result;
}

static PyMethodDef SymNMFMethods[] = {
    {"sym", py_sym, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Compute diagonal degree matrix"},
    {"norm", py_norm, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Optimize H matrix using SymNMF"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmf = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    SymNMFMethods
};

PyMODINIT_FUNC PyInit_symnmf(void)
{
    import_array();
    return PyModule_Create(&symnmf);
}