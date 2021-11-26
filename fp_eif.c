#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <gsl_randist.h>
#include <gsl_rng.h>
#include <sys/time.h>
#include "complex.h"


static PyObject* fp_eif(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;

    int k,Nloop,kre;
    double V;
    double p0sum, P0sum;

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

    Nloop = (int)floor((Vth-Vlb)/dV);  /* number of bins */
    kre = (int)round((Vr-Vlb)/dV); /* index of reset potential */

    double mu_in = PyFloat_AsDouble(PyList_GetItem(list, 13));
    double sigma2 = PyFloat_AsDouble(PyList_GetItem(list, 14));
    double x0_in = 0;

    double *Psp = malloc(Nloop*sizeof(double));
    double *j0 = malloc(Nloop*sizeof(double));
    double *I0 = malloc(Nloop*sizeof(double));
    double *G = malloc(Nloop*sizeof(double));
    double *alpha = malloc(Nloop*sizeof(double));
    double *beta = malloc(Nloop*sizeof(double));

    double *P0 = malloc(Nloop*sizeof(double));
    double *p0 = malloc(Nloop*sizeof(double));
    double *J0 = malloc(Nloop*sizeof(double));

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
    printf("mu_in = %f\n",mu_in);
    printf("var = %f\n",sigma2);
    printf("###################\n\n");

    double* xi = malloc(Nloop*sizeof(double));
    PyObject* _xi = PyList_GetItem(list, 15);
    Py_ssize_t _xi_size = PyList_Size(_xi);
    for (Py_ssize_t j = 0; j < _xi_size; j++) {
      xi[j] = PyFloat_AsDouble(PyList_GetItem(_xi, j));
      if (PyErr_Occurred()) return NULL;
    }

    p0[Nloop-1] = 0;
    Psp[Nloop-1] = 0;
    j0[Nloop-1] = 1;

    /* integrate backwards from threshold */
    V = Vth;
    I0[Nloop-1] = gL*(VL-V)+gx*x0_in*(Vx-V)+mu_in+gL*Delta*exp((V-VT)/Delta);
    G[Nloop-1] = -I0[Nloop-1]/(gL*sigma2);
    alpha[Nloop-1] = exp(dV*G[Nloop-1]);

    // if (G[Nloop-1]<.000000000001) beta[Nloop-1] = dV/sigma2;
    if (G[Nloop-1]==0) beta[Nloop-1] = dV/sigma2;
    else beta[Nloop-1] = (alpha[Nloop-1]-1)/(G[Nloop-1]*sigma2);


    p0sum = 0;

    for (k = Nloop-2; k>0; k--){

        V = V-dV;

        I0[k] = gL*(VL-V)+gx*x0_in*(Vx-V)+mu_in+gL*Delta*exp((V-VT)/Delta);
        G[k] = -I0[k]/(gL*sigma2);
        alpha[k] = exp(dV*G[k]);
    //     if (G[k]<.000000000001) {beta[k] = dV/sigma2; mexPrintf("\naa=%f %f\n", dV/sigma2, (alpha[k]-1)/(G[k]*sigma2)); }
        if (G[k]==0) beta[k] = dV/sigma2;
        else beta[k] = (alpha[k]-1)/(G[k]*sigma2);


        j0[k] = j0[k+1];
        if (k==kre) j0[k] += -1;

        p0[k] = p0[k+1]*alpha[k+1]+C/gL*j0[k+1]*beta[k+1];

        p0sum += p0[k];
        Psp[k] = 0;

    }

    /* normalize and compute firing rate and mean gx activation */

    double r0=1/(dV*p0sum+tref);
    double x0 = 0;
    P0sum = 0;

    for (k=0; k<Nloop; k++) {
        J0[k] = (r0)*j0[k];
        if (k >= kre) Psp[k] = (r0)*tref/(Vth-Vr);

        P0[k] = (p0[k]*(r0))+Psp[k]; // linear spike shape
        P0sum += P0[k];
        x0 += P0[k]*xi[k]/tau_x;
    }

    x0 = dV*(x0)/(dV*P0sum/tau_x);

    PyObject *P0_list = PyList_New(Nloop);
    PyObject *p0_list = PyList_New(Nloop);
    PyObject *J0_list = PyList_New(Nloop);
    for (int i = 0; i < Nloop; ++i) {
       PyList_SET_ITEM(P0_list, i, PyFloat_FromDouble(P0[i]));
       PyList_SET_ITEM(p0_list, i, PyFloat_FromDouble(p0[i]));
       PyList_SET_ITEM(J0_list, i, PyFloat_FromDouble(J0[i]));
   }

   free(Psp);
   free(j0);
   free(I0);
   free(G);
   free(alpha);
   free(beta);
   free(P0);
   free(p0);
   free(J0);
   free(xi);

  PyObject *x0_out = PyFloat_FromDouble(x0);
  PyObject *r0_out = PyFloat_FromDouble(r0);
  return Py_BuildValue("(OOOOO)",P0_list,p0_list,J0_list,x0_out,r0_out);
}
