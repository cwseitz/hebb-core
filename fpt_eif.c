#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <gsl_randist.h>
#include <gsl_rng.h>
#include <sys/time.h>
#include "complex.h"

static PyObject* fpt_eif(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;

    int f, k, Nloop, kre;
    double V, w, pi;
    double df, fmax;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
      return NULL;

      double gL = PyFloat_AsDouble(PyList_GetItem(list, 0));
      double C = PyFloat_AsDouble(PyList_GetItem(list, 1));
      double Delta = PyFloat_AsDouble(PyList_GetItem(list, 2));
      double VT = PyFloat_AsDouble(PyList_GetItem(list, 3));
      double VL = PyFloat_AsDouble(PyList_GetItem(list, 4));
      double Vth = PyFloat_AsDouble(PyList_GetItem(list, 5));
      double Vlb = PyFloat_AsDouble(PyList_GetItem(list, 6));
      double dV = PyFloat_AsDouble(PyList_GetItem(list, 7));
      double Vr = PyFloat_AsDouble(PyList_GetItem(list, 8));
      double tref = PyFloat_AsDouble(PyList_GetItem(list, 9));
      double tau_x = PyFloat_AsDouble(PyList_GetItem(list, 10));
      double Vx = PyFloat_AsDouble(PyList_GetItem(list, 11));
      double gx = PyFloat_AsDouble(PyList_GetItem(list, 12));

      double x0 = PyFloat_AsDouble(PyList_GetItem(list, 13));
      double mu_in = PyFloat_AsDouble(PyList_GetItem(list, 14));
      double sigma2 = PyFloat_AsDouble(PyList_GetItem(list, 15));
      double u1 = PyFloat_AsDouble(PyList_GetItem(list, 16));
      double r0 = PyFloat_AsDouble(PyList_GetItem(list, 17));

      Nloop = (int) floor((Vth-Vlb)/dV); /* number of bins */
      kre = (int) round((Vr-Vlb)/dV); /* index of reset potential */
      pi = 3.14159265;

      //Print params
      printf("\n\n###################\n");
      printf("Parameters:\n\n");
      printf("Nloop = %d\n", Nloop);
      printf("gL =  %f\n",gL);
      printf("C = %f\n",C);
      printf("Vlb = %f\n",Vlb);
      printf("Vth = %f\n",Vth);
      printf("Delta = %f\n",Delta);
      printf("VT = %f\n",VT);
      printf("VL = %f\n",VL);
      printf("Vr = %f\n",Vr);
      printf("tref = %f\n",tref);
      printf("tau_x = %f\n",tau_x);
      printf("Vx = %f\n",Vx);
      printf("gx = %f\n",gx);
      printf("dV = %f\n",dV);

      printf("x0 = %f\n",x0);
      printf("mu_in = %f\n",mu_in);
      printf("sigma2 = %f\n",sigma2);
      printf("u1 = %f\n",u1);
      printf("r0 = %f\n",r0);
      printf("###################\n\n");

      double* freq = malloc(Nloop*sizeof(double));
      PyObject* _freq = PyList_GetItem(list, 18);
      Py_ssize_t _freq_size = PyList_Size(_freq);
      for (int j = 0; j < _freq_size; j++) {
        freq[j] = PyFloat_AsDouble(PyList_GetItem(_freq, j));
        if (PyErr_Occurred()) return NULL;
      }

    int Nfreq = PyFloat_AsDouble(PyList_GetItem(list, 19));
    int m = 1;

    double *f0r = malloc(Nfreq*sizeof(double));
    double *f0i = malloc(Nfreq*sizeof(double));

    double _Complex *pf = malloc(Nloop*sizeof(double _Complex));
    double _Complex *po = malloc(Nloop*sizeof(double _Complex));
    double _Complex *jf = malloc(Nloop*sizeof(double _Complex));
    double _Complex *jo = malloc(Nloop*sizeof(double _Complex));
    double *I0 = malloc(Nloop*sizeof(double));
    double *G = malloc(Nloop*sizeof(double));
    double *alpha = malloc(Nloop*sizeof(double));
    double *beta = malloc(Nloop*sizeof(double));


    V=Vlb;
    for (k=0; k<Nloop; k++){

        I0[k] = gL*(VL-V)+mu_in+gL*Delta*exp((V-VT)/Delta);
        G[k] = -I0[k]/(gL*sigma2);
        alpha[k] = exp(dV*G[k]);
        if (G[k]==0) beta[k] = 1/sigma2;
        else beta[k] = (alpha[k]-1)/(G[k]*sigma2);

        jf[k] = 0;
        jo[k] = 0;
        pf[k] = 0;
        po[k] = 0;

        V += dV;
    }


    for (f=0; f<Nfreq; f++) {

    /* initialize flux and probability to 0 */
    jf[Nloop-1] = 1+0*I; // boundary condition
    jo[Nloop-1] = 0+0*I;
    pf[Nloop-1] = 0+0*I;
    po[Nloop-1] = 0+0*I;

    w = freq[f]*2*pi;

    V = Vth;

    for (k = Nloop-2; k>=0; k--){

        /* boundary conditions */
        jf[k] = jf[k+1]+dV*I*w*pf[k+1];
        pf[k] = pf[k+1]*alpha[k+1]+C/gL*jf[k+1]*beta[k+1];

        /* inhomogenous component - current modulations */
        jo[k] = jo[k+1]+dV*I*w*po[k+1]; if (k==kre) jo[k] -= cexp(-I*w*tref);
        po[k] = po[k+1]*alpha[k+1]+C/gL*jo[k+1]*beta[k+1];

        V -= dV;

    }

    f0r[f] = creal(-jo[0]/jf[0]);
    f0i[f] = cimag(-jo[0]/jf[0]);

    }

    PyObject *f0r_list = PyList_New(Nfreq);
    PyObject *f0i_list = PyList_New(Nfreq);

    for (int i = 0; i < Nfreq; ++i) {
       PyList_SET_ITEM(f0r_list, i, PyFloat_FromDouble(f0r[i]));
       PyList_SET_ITEM(f0i_list, i, PyFloat_FromDouble(f0i[i]));
   }

  free(freq);
  free(f0r);
  free(f0i);
  free(pf);
  free(po);
  free(jf);
  free(jo);
  free(I0);
  free(G);
  free(alpha);
  free(beta);

  return Py_BuildValue("(OO)",f0r_list,f0i_list);
}
