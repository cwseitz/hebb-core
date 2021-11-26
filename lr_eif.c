#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <gsl_randist.h>
#include <gsl_rng.h>
#include <sys/time.h>
#include "complex.h"

static PyObject* lr_eif(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;

    int f, k, Nloop, kre;
    double  V, gamx, w, pi, byt0;
    double _Complex xibytu,xibytx,x0bytu,x0bytx;


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
    gamx = gx/gL;

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

    printf("x0 = %f\n",x0);
    printf("mu_in = %f\n",mu_in);
    printf("sigma2 = %f\n",sigma2);
    printf("u1 = %f\n",u1);
    printf("r0 = %f\n",r0);
    printf("###################\n\n");


    double* P0 = malloc(Nloop*sizeof(double));
    PyObject* _P0 = PyList_GetItem(list, 18);
    Py_ssize_t _P0_size = PyList_Size(_P0);
    for (Py_ssize_t j = 0; j < _P0_size; j++) {
      P0[j] = PyFloat_AsDouble(PyList_GetItem(_P0, j));
      if (PyErr_Occurred()) return NULL;
    }

    double* xi = malloc(Nloop*sizeof(double));
    PyObject* _xi = PyList_GetItem(list, 19);
    Py_ssize_t _xi_size = PyList_Size(_xi);
    for (Py_ssize_t j = 0; j < _xi_size; j++) {
      xi[j] = PyFloat_AsDouble(PyList_GetItem(_xi, j));
      if (PyErr_Occurred()) return NULL;
    }

    double* p0 = malloc(Nloop*sizeof(double));
    PyObject* _p0 = PyList_GetItem(list, 20);
    Py_ssize_t _p0_size = PyList_Size(_p0);
    for (Py_ssize_t j = 0; j < _p0_size; j++) {
      p0[j] = PyFloat_AsDouble(PyList_GetItem(_p0, j));
      if (PyErr_Occurred()) return NULL;
    }

    double* freq = malloc(Nloop*sizeof(double));
    PyObject* _freq = PyList_GetItem(list, 21);
    Py_ssize_t _freq_size = PyList_Size(_freq);
    for (Py_ssize_t j = 0; j < _freq_size; j++) {
      freq[j] = PyFloat_AsDouble(PyList_GetItem(_freq, j));
      if (PyErr_Occurred()) return NULL;
    }

    int Nfreq = PyFloat_AsDouble(PyList_GetItem(list, 22));
    int m = 1;

    // if (Nfreq==1 && m!=1) {
    //     Nfreq = m;
    //     m = 1;
    // }

    // /* allocate outputs */
    // plhs[0] = mxCreateDoubleMatrix(Nfreq,1,mxCOMPLEX);
    // plhs[1] = mxCreateDoubleMatrix(Nfreq,1,mxCOMPLEX);
    // plhs[2] = mxCreateDoubleMatrix(Nfreq,1,mxCOMPLEX);
    //
    // Ar = mxGetPr(plhs[0]);
    // Ai = mxGetPi(plhs[0]);
    // x1r = mxGetPr(plhs[1]);
    // x1i = mxGetPi(plhs[1]);
    // V1r = mxGetPr(plhs[2]);
    // V1i = mxGetPi(plhs[2]);

    double *Ar = malloc(Nfreq*sizeof(double));
    double *Ai = malloc(Nfreq*sizeof(double));
    double *x1r = malloc(Nfreq*sizeof(double));
    double *x1i = malloc(Nfreq*sizeof(double));
    double *V1r = malloc(Nfreq*sizeof(double));
    double *V1i = malloc(Nfreq*sizeof(double));

    /* initialize arrays for integration */
    double *I0,*G,*alpha,*beta;
    double _Complex *ru,*rx,*jr,*ju,*jx,*pr,*pu,*px,*psp,*Pu,*Px,*P1;

    ru= malloc(Nfreq*sizeof(double _Complex));
    rx= malloc(Nfreq*sizeof(double _Complex));
    jr= malloc(Nloop*sizeof(double _Complex));
    ju= malloc(Nloop*sizeof(double _Complex));
    jx= malloc(Nloop*sizeof(double _Complex));
    pr = malloc(Nloop*sizeof(double _Complex));
    pu = malloc(Nloop*sizeof(double _Complex));
    px = malloc(Nloop*sizeof(double _Complex));
    psp = malloc(Nloop*sizeof(double _Complex));
    Pu = malloc(Nloop*sizeof(double _Complex));
    Px = malloc(Nloop*sizeof(double _Complex));
    P1 = malloc(Nloop*sizeof(double _Complex));
    I0 = malloc(Nloop*sizeof(double));
    G = malloc(Nloop*sizeof(double));
    alpha = malloc(Nloop*sizeof(double));
    beta = malloc(Nloop*sizeof(double));


    V=Vlb;

    for (k=0; k<Nloop; k++){

        I0[k] = gL*(VL-V)+gx*x0*(Vx-V)+mu_in+gL*Delta*exp((V-VT)/Delta);
        G[k] = -I0[k]/(gL*sigma2);
        alpha[k] = exp(dV*G[k]);
        if (G[k]<.000001) beta[k] = dV/sigma2;
        else beta[k] = (alpha[k]-1)/(G[k]*sigma2);

        jr[k] = 0;
        ju[k] = 0;
        jx[k] = 0;
        pr[k] = 0;
        pu[k] = 0;
        px[k] = 0;
        psp[k] = 0;
        Pu[k] = 0;
        Px[k] = 0;
        P1[k] = 0;

        V += dV;

    }


    for (f=0; f<Nfreq; f++) {
        /* initialize flux and probability to 0 */
        jr[Nloop-1] = 1+0*I; // boundary condition
        ju[Nloop-1] = 0+0*I;
        jx[Nloop-1] = 0+0*I;
        pr[Nloop-1] = 0+0*I;
        pu[Nloop-1] = 0+0*I;
        px[Nloop-1] = 0+0*I;

        w = freq[f]*2*pi;
        V = Vth;

        for (k = Nloop-2; k>=0; k--){

            /* boundary conditions */
            jr[k] = jr[k+1]+dV*I*w*pr[k+1]; if (k==kre) jr[k] -= cexp(-I*w*tref);
            pr[k] = pr[k+1]*alpha[k+1]+C/gL*jr[k+1]*beta[k+1];

            /* inhomogenous component - current modulations */
            ju[k] = ju[k+1]+dV*I*w*pu[k+1];
            pu[k] = pu[k+1]*alpha[k+1]+(C/gL*ju[k+1]-u1/gL*(r0*p0[k+1]))*beta[k+1];

            /* inhomogenous component - voltage-activated conductance modulations */
            jx[k] = jx[k+1]+dV*I*w*px[k+1];
            px[k] = px[k+1]*alpha[k+1]+(C/gL*jx[k+1]+(V-Vx)*(r0*p0[k+1]))*beta[k+1];

            V -= dV;
        }

        /* normalize and compute first-order activity */

        ru[f] = -ju[0]/jr[0];
        rx[f] = -jx[0]/jr[0];

        V = Vlb;
        xibytu = 0; xibytx = 0; x0bytu = 0; x0bytx = 0;
        for (k=0; k<Nloop; k++) {

            if (k<kre) psp[k] = 0;
            else psp[k] = cexp(-I*w*tref*((Vth-V)/(Vth-Vr)))/(Vth-Vr);

            Pu[k] = pu[k]+ru[f]*(pr[k]+tref*psp[k]);
            Px[k] = px[k]+rx[f]*(pr[k]+tref*psp[k]);

            xibytu += (Pu[k]*xi[k]);
            xibytx += (Px[k]*xi[k]);
            x0bytu += Pu[k];
            x0bytx += Px[k];
            byt0 += P0[k];

            V += dV;
        }

        xibytu = xibytu*dV/tau_x;
        xibytx = xibytx*dV/tau_x;
        x0bytu = x0bytu*x0*dV/tau_x;
        x0bytx = x0bytx*x0*dV/tau_x;
        byt0 = byt0*dV/tau_x;

        x1r[f] = creal((xibytu-x0bytu)/(byt0+I*w-gamx*(xibytx-x0bytx)));
        x1i[f] = cimag((xibytu-x0bytu)/(byt0+I*w-gamx*(xibytx-x0bytx)));
        Ar[f] = creal(ru[f]+gamx*(x1r[f]+x1i[f])*rx[f]);
        Ai[f] = cimag(ru[f]+gamx*(x1r[f]+x1i[f])*rx[f]);


        V = Vlb;
        for (k=0; k<Nloop; k++) {
            P1[k] = Pu[k]+(x1r[f]+x1i[f])*Px[k];
            V1r[f] += creal(P1[k]*V);
            V1i[f] += cimag(P1[k]*V);
            V += dV;
        }
        V1r[f] = V1r[f]*dV;
        V1i[f] = V1i[f]*dV;


    }

    PyObject *V1r_list = PyList_New(Nfreq);
    PyObject *V1i_list = PyList_New(Nfreq);
    PyObject *x1r_list = PyList_New(Nfreq);
    PyObject *x1i_list = PyList_New(Nfreq);
    PyObject *Ar_list = PyList_New(Nfreq);
    PyObject *Ai_list = PyList_New(Nfreq);

    //print_double_arr(x1i,Nfreq);

    for (int i = 0; i < Nfreq; ++i) {
       PyList_SET_ITEM(V1r_list, i, PyFloat_FromDouble(V1r[i]));
       PyList_SET_ITEM(V1i_list, i, PyFloat_FromDouble(V1i[i]));
       PyList_SET_ITEM(Ar_list, i, PyFloat_FromDouble(Ar[i]));
       PyList_SET_ITEM(Ai_list, i, PyFloat_FromDouble(Ai[i]));
       PyList_SET_ITEM(x1r_list, i, PyFloat_FromDouble(x1r[i]));
       PyList_SET_ITEM(x1i_list, i, PyFloat_FromDouble(x1i[i]));
   }

   free(Ar);
   free(Ai);
   free(x1r);
   free(x1i);
   free(V1r);
   free(V1i);
   free(ru);
   free(rx);
   free(jr);
   free(ju);
   free(jx);
   free(pr);
   free(pu);
   free(px);
   free(psp);
   free(Pu);
   free(Px);
   free(P1);
   free(I0);
   free(G);
   free(alpha);
   free(beta);
   free(P0);
   free(p0);
   free(xi);
   free(freq);

  return Py_BuildValue("(OOOOOO)",V1r_list,V1i_list,x1r_list,x1i_list,Ar_list,Ai_list);
  //Py_RETURN_NONE;

}
