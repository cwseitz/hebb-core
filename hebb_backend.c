#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <gsl_randist.h>
#include <gsl_rng.h>
#include <sys/time.h>
#include "complex.h"


static PyObject* FPT_EIF(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;

    int f, k, Nloop, kre;
    double V, w, pi;
    double df, fmax;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
      return NULL;

      int N = PyFloat_AsDouble(PyList_GetItem(list, 0));
      double gL = PyFloat_AsDouble(PyList_GetItem(list, 1));
      double C = PyFloat_AsDouble(PyList_GetItem(list, 2));
      double Delta = PyFloat_AsDouble(PyList_GetItem(list, 3));
      double VT = PyFloat_AsDouble(PyList_GetItem(list, 4));
      double VL = PyFloat_AsDouble(PyList_GetItem(list, 5));
      double Vth = PyFloat_AsDouble(PyList_GetItem(list, 6));
      double Vlb = PyFloat_AsDouble(PyList_GetItem(list, 7));
      double dV = PyFloat_AsDouble(PyList_GetItem(list, 8));
      double Vr = PyFloat_AsDouble(PyList_GetItem(list, 9));
      double tref = PyFloat_AsDouble(PyList_GetItem(list, 10));
      double tau_x = PyFloat_AsDouble(PyList_GetItem(list, 11));
      double Vx = PyFloat_AsDouble(PyList_GetItem(list, 12));
      double gx = PyFloat_AsDouble(PyList_GetItem(list, 13));

      double x0 = PyFloat_AsDouble(PyList_GetItem(list, 14));
      double mu_in = PyFloat_AsDouble(PyList_GetItem(list, 15));
      double sigma2 = PyFloat_AsDouble(PyList_GetItem(list, 16));
      double u1 = PyFloat_AsDouble(PyList_GetItem(list, 17));
      double r0 = PyFloat_AsDouble(PyList_GetItem(list, 18));

      Nloop = (int) floor((Vth-Vlb)/dV); /* number of bins */
      kre = (int) round((Vr-Vlb)/dV); /* index of reset potential */
      pi = 3.14159265;

      //Print params
      printf("\n\n###################\n");
      printf("Parameters:\n\n");
      printf("N = %d\n", N);
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
      PyObject* _freq = PyList_GetItem(list, 19);
      //Py_ssize_t _freq_size = PyList_Size(_freq);
      for (int j = 0; j < Nloop; j++) {
        freq[j] = PyFloat_AsDouble(PyList_GetItem(_freq, j));
        if (PyErr_Occurred()) return NULL;
      }

    int Nfreq = PyFloat_AsDouble(PyList_GetItem(list, 20));
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

static PyObject* LR_EIF(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;

    int f, k, Nloop, kre;
    double  V, gamx, w, pi, byt0;
    double _Complex xibytu,xibytx,x0bytu,x0bytx;


    int N = PyFloat_AsDouble(PyList_GetItem(list, 0));
    double gL = PyFloat_AsDouble(PyList_GetItem(list, 1));
    double C = PyFloat_AsDouble(PyList_GetItem(list, 2));
    double Delta = PyFloat_AsDouble(PyList_GetItem(list, 3));
    double VT = PyFloat_AsDouble(PyList_GetItem(list, 4));
    double VL = PyFloat_AsDouble(PyList_GetItem(list, 5));
    double Vth = PyFloat_AsDouble(PyList_GetItem(list, 6));
    double Vlb = PyFloat_AsDouble(PyList_GetItem(list, 7));
    double dV = PyFloat_AsDouble(PyList_GetItem(list, 8));
    double Vr = PyFloat_AsDouble(PyList_GetItem(list, 9));
    double tref = PyFloat_AsDouble(PyList_GetItem(list, 10));
    double tau_x = PyFloat_AsDouble(PyList_GetItem(list, 11));
    double Vx = PyFloat_AsDouble(PyList_GetItem(list, 12));
    double gx = PyFloat_AsDouble(PyList_GetItem(list, 13));

    double x0 = PyFloat_AsDouble(PyList_GetItem(list, 14));
    double mu_in = PyFloat_AsDouble(PyList_GetItem(list, 15));
    double sigma2 = PyFloat_AsDouble(PyList_GetItem(list, 16));
    double u1 = PyFloat_AsDouble(PyList_GetItem(list, 17));
    double r0 = PyFloat_AsDouble(PyList_GetItem(list, 18));

    Nloop = (int) floor((Vth-Vlb)/dV); /* number of bins */
    kre = (int) round((Vr-Vlb)/dV); /* index of reset potential */
    pi = 3.14159265;
    gamx = gx/gL;

    //Print params
    printf("\n\n###################\n");
    printf("Parameters:\n\n");
    printf("N = %d\n", N);
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
    PyObject* _P0 = PyList_GetItem(list, 19);
    Py_ssize_t _P0_size = PyList_Size(_P0);
    for (Py_ssize_t j = 0; j < _P0_size; j++) {
      P0[j] = PyFloat_AsDouble(PyList_GetItem(_P0, j));
      if (PyErr_Occurred()) return NULL;
    }

    double* xi = malloc(Nloop*sizeof(double));
    PyObject* _xi = PyList_GetItem(list, 20);
    Py_ssize_t _xi_size = PyList_Size(_xi);
    for (Py_ssize_t j = 0; j < _xi_size; j++) {
      xi[j] = PyFloat_AsDouble(PyList_GetItem(_xi, j));
      if (PyErr_Occurred()) return NULL;
    }

    double* p0 = malloc(Nloop*sizeof(double));
    PyObject* _p0 = PyList_GetItem(list, 21);
    Py_ssize_t _p0_size = PyList_Size(_p0);
    for (Py_ssize_t j = 0; j < _p0_size; j++) {
      p0[j] = PyFloat_AsDouble(PyList_GetItem(_p0, j));
      if (PyErr_Occurred()) return NULL;
    }

    double* freq = malloc(Nloop*sizeof(double));
    PyObject* _freq = PyList_GetItem(list, 22);
    Py_ssize_t _freq_size = PyList_Size(_freq);
    for (Py_ssize_t j = 0; j < _freq_size; j++) {
      freq[j] = PyFloat_AsDouble(PyList_GetItem(_freq, j));
      if (PyErr_Occurred()) return NULL;
    }

    int Nfreq = PyFloat_AsDouble(PyList_GetItem(list, 23));
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

static PyObject* FokkerPlanck_EIF(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;

    int k,Nloop,kre;
    double V;
    double p0sum, P0sum;

    int N = PyFloat_AsDouble(PyList_GetItem(list, 0));
    double gL = PyFloat_AsDouble(PyList_GetItem(list, 1));
    double C = PyFloat_AsDouble(PyList_GetItem(list, 2));
    double Delta = PyFloat_AsDouble(PyList_GetItem(list, 3));
    double VT = PyFloat_AsDouble(PyList_GetItem(list, 4));
    double VL = PyFloat_AsDouble(PyList_GetItem(list, 5));
    double Vth = PyFloat_AsDouble(PyList_GetItem(list, 6));
    double Vlb = PyFloat_AsDouble(PyList_GetItem(list, 7));
    double dV = PyFloat_AsDouble(PyList_GetItem(list, 8));
    double Vr = PyFloat_AsDouble(PyList_GetItem(list, 9));
    double tref = PyFloat_AsDouble(PyList_GetItem(list, 10));
    double tau_x = PyFloat_AsDouble(PyList_GetItem(list, 11));
    double Vx = PyFloat_AsDouble(PyList_GetItem(list, 12));
    double gx = PyFloat_AsDouble(PyList_GetItem(list, 13));

    Nloop = (int)floor((Vth-Vlb)/dV);  /* number of bins */
    kre = (int)round((Vr-Vlb)/dV); /* index of reset potential */

    double mu_in = PyFloat_AsDouble(PyList_GetItem(list, 14));
    double sigma2 = PyFloat_AsDouble(PyList_GetItem(list, 15));
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
    // printf("\n\n###################\n");
    // printf("Parameters:\n\n");
    // printf("N = %d\n", N);
    // printf("Nloop = %d\n", Nloop);
    // printf("gL =  %f\n",gL);
    // printf("C = %f\n",C);
    // printf("Vlb = %f\n",Vlb);
    // printf("Vth = %f\n",Vth);
    // printf("Delta = %f\n",Delta);
    // printf("VT = %f\n",VT);
    // printf("VL = %f\n",VL);
    // printf("Vr = %f\n",Vr);
    // printf("tref = %f\n",tref);
    // printf("tau_x = %f\n",tau_x);
    // printf("Vx = %f\n",Vx);
    // printf("gx = %f\n",gx);
    // printf("###################\n\n");

    double* xi = malloc(Nloop*sizeof(double));
    PyObject* _xi = PyList_GetItem(list, 16);
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

void sim(int N, int Nrecord, double T, int Nt, int Ne, int Ni, double q,
         double dt, double pee0, double pei0, double pie0, double pii0,
         double jee, double jei, double jie, double jii, double wee0,
         double wei0, double wie0, double wii0, int Kee, int Kei, int Kie,
         int Kii, double taux, double mxe, double mxi, double vxe, double vxi,
         double tausyne, double tausyni, double tausynx, double Jee, double Jei,
         double Jie, double Jii, double maxns, double *gl, double *C, double *vlb,
         double *vth, double *DeltaT, double *vT, double *vl, double *vre, double *tref,
         int *Irecord, double *v0, double *s, double *alphaxr,double *alphaer,
         double *alphair, double *vr, double *v, double *JnextE, double *JnextI,
         double *alphae, double *alphai, double *alphax, int *Wee,int *Wei,int
         *Wie,int *Wii,int *refstate, double *ffwd, double *ffwdr){

   int j, jj, k;

   // gsl_vector *mu_e = gsl_vector_alloc(N);
   // gsl_vector_set_all(mu_e, mxe);
   // gsl_vector *mu_i = gsl_vector_alloc(N);
   // gsl_vector_set_all(mu_i, mxi);
   // gsl_matrix *cov = gsl_matrix_alloc(N,N);
   // gsl_matrix_set_identity(cov);

  /* Inititalize v */
  for(j=0;j<N;j++){
      v[j]=v0[j];
      refstate[j]=0;
      JnextE[j]=0;
      JnextI[j]=0;
      alphae[j]=0;
      alphai[j]=0;
      alphax[j]=0;
      ffwd[j]=0;
     }


  for(jj=0;jj<Nrecord;jj++){
        j=(int)round(Irecord[jj]-1);
        alphaer[jj+Nrecord*0]=alphae[j];
        alphair[jj+Nrecord*0]=alphai[j];
        alphaxr[jj+Nrecord*0]=alphax[j];
        vr[jj+Nrecord*0]=v[j];
        ffwdr[jj+Nrecord*0]=ffwd[j];
  }


  /* Initialize connection matrix */
  for(j=0;j<Ne;j++){
      for(k=0;k<Kee;k++)
          Wee[j*Kee+k]=(int)floor(drand48()*((double)(Ne)));
      for(k=0;k<Kie;k++)
          Wie[j*Kie+k]=Ne+(int)floor(drand48()*((double)(Ni)));
       }

  for(j=0;j<Ni;j++){
      for(k=0;k<Kei;k++)
          Wei[j*Kei+k]=(int)floor(drand48()*((double)(Ne)));
      for(k=0;k<Kii;k++)
          Wii[j*Kii+k]=Ne+(int)floor(drand48()*((double)(Ni)));
       }

  int *Ntref = malloc(2*sizeof(int));
  Ntref[0]=(int)round(tref[0]/dt);
  Ntref[1]=(int)round(tref[1]/dt);

  int ns=0;
  int flag=0;
  int i;
  double Ixe, Ixi;
  struct timeval tv; // Seed generation based on time
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);

  // printf("Spikes: %f\n", ns);
  // printf("Max Spikes: %f\n", maxns);
  for(i=1;i<Nt && ns<maxns;i++){
      if (i % 100 == 0) {
      printf("Time step: %d/%d\n", i, Nt);
      }


       for(j=0;j<N;j++){

           alphae[j]-=alphae[j]*(dt/tausyne);
           alphai[j]-=alphai[j]*(dt/tausyni);
           alphax[j]-=alphax[j]*(dt/tausynx);

           gettimeofday(&tv,0);
           unsigned long mySeed = tv.tv_sec + tv.tv_usec;
           gsl_rng_set(r, mySeed);

           if(j<Ne){
               ffwd[j] = mxe+gsl_ran_gaussian(r, sqrt(vxe));
               //printf("%f\n", Ixe);
               if(refstate[j]<=0){
                  v[j]+=(alphae[j]+alphai[j]+alphax[j]-gl[0]*(v[j]-vl[0])+gl[0]*DeltaT[0]*exp((v[j]-vT[0])/DeltaT[0]))*dt/C[0];
                  v[j]+=ffwd[j]*dt/C[1];
                  // if(j<Ne1)
                  //     v[j]+=Ix1e[i]*dt/C[0];
                  // else
                  //     v[j]+=Ix2e[i]*dt/C[0];
                  v[j]=fmax(v[j],vlb[0]);
               }
               else{
                  if(refstate[j]>1)
                     v[j]=vth[0];/*-=(vth[0]-vre[0])/((double)Ntref[0]);*/
                  else
                     v[j]=vre[0];
                  refstate[j]--;
               }
               /* If a spike occurs */
               if(v[j]>=vth[0] && refstate[j]<=0 && ns<maxns){

                    refstate[j]=Ntref[0];
                    v[j]=vth[0];       /* reset membrane potential */
                    s[0+2*ns]=i*dt; /* spike time */
                    s[1+2*ns]=j+1;     /* neuron index 1 */
                    ns++;           /* update total number of spikes */

                    /* For each postsynaptic target */
                    for(k=0;k<Kee;k++)
                           JnextE[Wee[j*Kee+k]]+=Jee;
                    for(k=0;k<Kie;k++)
                           JnextE[Wie[j*Kie+k]]+=Jie;

                }
           }
           else{ /* If cell is inhibitory */
               ffwd[j] = mxi+gsl_ran_gaussian(r, sqrt(vxi));
               if(refstate[j]<=0){
                  v[j]+=(alphae[j]+alphai[j]+alphax[j]-gl[1]*(v[j]-vl[1])+gl[1]*DeltaT[1]*exp((v[j]-vT[1])/DeltaT[1]))*dt/C[1];
                  v[j]+=ffwd[j]*dt/C[1];
                  // if(j<Ne+Ni1)
                  //     v[j]+=Ix1i[i]*dt/C[1];
                  // else
                  //     v[j]+=Ix2i[i]*dt/C[1];
                  v[j]=fmax(v[j],vlb[1]);
               }
               else{
                  if(refstate[j]>1)
                     v[j]=vth[1];/*-=(vth[1]-vre[1])/((double)Ntref[1]);*/
                  else
                     v[j]=vre[1];
                  refstate[j]--;
               }

                /* If a spike occurs */
                if(v[j]>=vth[1] && refstate[j]<=0 && ns<maxns){

                    refstate[j]=Ntref[1];
                    v[j]=vth[1];       /* reset membrane potential */
                    s[0+2*ns]=i*dt; /* spike time */
                    s[1+2*ns]=j+1;     /* neuron index 1 */
                    ns++;           /* update total number of spikes */
                   /* For each postsynaptic target */
                  for(k=0;k<Kei;k++)
                          JnextI[Wei[(j-Ne)*Kei+k]]+=Jei;
                   for(k=0;k<Kii;k++)
                          JnextI[Wii[(j-Ne)*Kii+k]]+=Jii;
                 }
             }
          }


          /* Store recorded variables */
          for(jj=0;jj<Nrecord;jj++){
              /* Find index into local variables */
              j=(int)round(Irecord[jj]-1);
              alphaer[jj+Nrecord*i]=alphae[j];
              alphair[jj+Nrecord*i]=alphai[j];
              alphaxr[jj+Nrecord*i]=alphax[j];
              ffwdr[jj+Nrecord*i]=ffwd[j];
              vr[jj+Nrecord*i]=v[j];
          }
          /* Propagate spikes */
          for(j=0;j<N;j++){
            alphae[j]+=JnextE[j]/tausyne;
            alphai[j]+=JnextI[j]/tausyni;
            JnextE[j]=0;
            JnextI[j]=0;
          }

          // for(j=0;j<Ne;j++)
          //     if(drand48()<rxe*dt)
          //         alphax[j]+=Jex/tausynx;
          //
          // for(j=Ne;j<N;j++)
          //     if(drand48()<rxi*dt)
          //         alphax[j]+=Jix/tausynx;

  }

  if(ns>=maxns)
     printf("Maximum number of spikes reached (%d), simulation terminated.\n", ns);

  return 0;

}

  static PyObject* EIF(PyObject* Py_UNUSED(self), PyObject* args) {

    PyObject* list;

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
      return NULL;

      //Quantities that will be passed to the simulation directly
      int N = PyFloat_AsDouble(PyList_GetItem(list, 0));
      int Nrecord = PyFloat_AsDouble(PyList_GetItem(list, 1));
      float T = PyFloat_AsDouble(PyList_GetItem(list, 2));
      int Nt = PyFloat_AsDouble(PyList_GetItem(list, 3));
      int Ne = PyFloat_AsDouble(PyList_GetItem(list, 4));
      int Ni = PyFloat_AsDouble(PyList_GetItem(list, 5));
      double q = PyFloat_AsDouble(PyList_GetItem(list, 6));
      double dt = PyFloat_AsDouble(PyList_GetItem(list, 7));
      double pee0 = PyFloat_AsDouble(PyList_GetItem(list, 8));
      double pei0 = PyFloat_AsDouble(PyList_GetItem(list, 9));
      double pie0 = PyFloat_AsDouble(PyList_GetItem(list, 10));
      double pii0 = PyFloat_AsDouble(PyList_GetItem(list, 11));
      double jee = PyFloat_AsDouble(PyList_GetItem(list, 12));
      double jei = PyFloat_AsDouble(PyList_GetItem(list, 13));
      double jie = PyFloat_AsDouble(PyList_GetItem(list, 14));
      double jii = PyFloat_AsDouble(PyList_GetItem(list, 15));
      double wee0 = PyFloat_AsDouble(PyList_GetItem(list, 16));
      double wei0 = PyFloat_AsDouble(PyList_GetItem(list, 17));
      double wie0 = PyFloat_AsDouble(PyList_GetItem(list, 18));
      double wii0 = PyFloat_AsDouble(PyList_GetItem(list, 19));
      int Kee = (int) PyFloat_AsDouble(PyList_GetItem(list, 20));
      int Kei = (int) PyFloat_AsDouble(PyList_GetItem(list, 21));
      int Kie = (int) PyFloat_AsDouble(PyList_GetItem(list, 22));
      int Kii = (int) PyFloat_AsDouble(PyList_GetItem(list, 23));
      double taux = PyFloat_AsDouble(PyList_GetItem(list, 24));
      double mxe = PyFloat_AsDouble(PyList_GetItem(list, 25));
      double mxi = PyFloat_AsDouble(PyList_GetItem(list, 26));
      double vxe = PyFloat_AsDouble(PyList_GetItem(list, 27));
      double vxi = PyFloat_AsDouble(PyList_GetItem(list, 28));
      double tausyne = PyFloat_AsDouble(PyList_GetItem(list, 29));
      double tausyni = PyFloat_AsDouble(PyList_GetItem(list, 30));
      double tausynx = PyFloat_AsDouble(PyList_GetItem(list, 31));
      double Jee = PyFloat_AsDouble(PyList_GetItem(list, 32));
      double Jei = PyFloat_AsDouble(PyList_GetItem(list, 33));
      double Jie = PyFloat_AsDouble(PyList_GetItem(list, 34));
      double Jii = PyFloat_AsDouble(PyList_GetItem(list, 35));
      double maxns = PyFloat_AsDouble(PyList_GetItem(list, 36));

      // double rxe = PyFloat_AsDouble(PyList_GetItem(list, 53));
      // double rxi = PyFloat_AsDouble(PyList_GetItem(list, 54));
      // double Jex = PyFloat_AsDouble(PyList_GetItem(list, 55));
      // double Jix = PyFloat_AsDouble(PyList_GetItem(list, 56));
      // int Ne1 = (int) PyFloat_AsDouble(PyList_GetItem(list, 57));
      // int Ni1 = (int) PyFloat_AsDouble(PyList_GetItem(list, 58));

      //Chunks of memory passed to the function as pointers
      PyObject* _gl = PyList_GetItem(list, 37);
      PyObject* _Cm = PyList_GetItem(list, 38);
      PyObject* _vlb = PyList_GetItem(list, 39);
      PyObject* _vth = PyList_GetItem(list, 40);
      PyObject* _DeltaT = PyList_GetItem(list, 41);
      PyObject* _vT = PyList_GetItem(list, 42);
      PyObject* _vl = PyList_GetItem(list, 43);
      PyObject* _vre = PyList_GetItem(list, 44);
      PyObject* _tref = PyList_GetItem(list, 45);
      double* gl = malloc(2*sizeof(double));
      double* Cm = malloc(2*sizeof(double));
      double* vlb = malloc(2*sizeof(double));
      double* vth = malloc(2*sizeof(double));
      double* DeltaT = malloc(2*sizeof(double));
      double* vT = malloc(2*sizeof(double));
      double* vl = malloc(2*sizeof(double));
      double* vre = malloc(2*sizeof(double));
      double* tref = malloc(2*sizeof(double));
      gl[0] = PyFloat_AsDouble(PyList_GetItem(_gl, 0));
      gl[1] = PyFloat_AsDouble(PyList_GetItem(_gl, 1));
      Cm[0] = PyFloat_AsDouble(PyList_GetItem(_Cm, 0));
      Cm[1] = PyFloat_AsDouble(PyList_GetItem(_Cm, 1));
      vlb[0] = PyFloat_AsDouble(PyList_GetItem(_vlb, 0));
      vlb[1] = PyFloat_AsDouble(PyList_GetItem(_vlb, 1));
      vth[0] = PyFloat_AsDouble(PyList_GetItem(_vth, 0));
      vth[1] = PyFloat_AsDouble(PyList_GetItem(_vth, 1));
      DeltaT[0] = PyFloat_AsDouble(PyList_GetItem(_DeltaT, 0));
      DeltaT[1] = PyFloat_AsDouble(PyList_GetItem(_DeltaT, 1));
      vT[0] = PyFloat_AsDouble(PyList_GetItem(_vT, 0));
      vT[1] = PyFloat_AsDouble(PyList_GetItem(_vT, 1));
      vl[0] = PyFloat_AsDouble(PyList_GetItem(_vl, 0));
      vl[1] = PyFloat_AsDouble(PyList_GetItem(_vl, 1));
      vre[0] = PyFloat_AsDouble(PyList_GetItem(_vre, 0));
      vre[1] = PyFloat_AsDouble(PyList_GetItem(_vre, 1));
      tref[0] = PyFloat_AsDouble(PyList_GetItem(_tref, 0));
      tref[1] = PyFloat_AsDouble(PyList_GetItem(_tref, 1));

      //More chunks of memory passed to the function as pointers
      double* s = malloc(2 * maxns * sizeof(double));
      double* alphaxr = malloc(Nrecord * Nt * sizeof(double));
      double* alphaer = malloc(Nrecord * Nt * sizeof(double));
      double* alphair = malloc(Nrecord * Nt * sizeof(double));
      double* vr = malloc(Nrecord * Nt * sizeof(double));
      double* v = malloc(N*sizeof(double));
      double* JnextE = malloc(N*sizeof(double));
      double* JnextI = malloc(N*sizeof(double));
      double* alphae = malloc(N*sizeof(double));
      double* alphai = malloc(N*sizeof(double));
      double* alphax = malloc(N*sizeof(double));
      double* ffwd = malloc(N*sizeof(double));
      double* ffwdr = malloc(Nrecord * Nt * sizeof(double));
      int* Wee = malloc(Ne*Kee*sizeof(int));
      int* Wei = malloc(Ni*Kei*sizeof(int));
      int* Wie = malloc(Ne*Kie*sizeof(int));
      int* Wii = malloc(Ni*Kii*sizeof(int));
      int* refstate = malloc(N*sizeof(int));


      // double* Ixe = malloc(Nt*sizeof(double));
      // double* Ixi = malloc(Nt*sizeof(double));
      // // double* Ix1i = malloc(Nt*sizeof(double));
      // // double* Ix2i = malloc(Nt*sizeof(double));
      //
      // //Feedforward inputs passed to the sim function as pointers
      // PyObject* _Ixe = PyList_GetItem(list, 46);
      // PyObject* _Ixi = PyList_GetItem(list, 47);
      // // PyObject* _Ix1i = PyList_GetItem(list, 48);
      // // PyObject* _Ix2i = PyList_GetItem(list, 49);
      //
      // Py_ssize_t _Ixe_size = PyList_Size(_Ixe);
      // for (Py_ssize_t j = 0; j < _Ixe_size; j++) {
      //   Ixe[j] = PyFloat_AsDouble(PyList_GetItem(_Ixe, j));
      //   Ixi[j] = PyFloat_AsDouble(PyList_GetItem(_Ixi, j));
      //   // Ix1i[j] = PyFloat_AsDouble(PyList_GetItem(_Ix1i, j));
      //   // Ix2i[j] = PyFloat_AsDouble(PyList_GetItem(_Ix2i, j));
      //   if (PyErr_Occurred()) return NULL;
      // }

      double nrecord = PyFloat_AsDouble(PyList_GetItem(list, 46));
      PyObject* _Irecord = PyList_GetItem(list, 47);
      int* Irecord = malloc(nrecord*sizeof(int));

      Py_ssize_t _Irecord_size = PyList_Size(_Irecord);
      for (Py_ssize_t j = 0; j < _Irecord_size; j++) {
        Irecord[j] = (int) PyLong_AsDouble(PyList_GetItem(_Irecord, j));
        if (PyErr_Occurred()) return NULL;
      }

      double* V0 = malloc(N*sizeof(double));
      PyObject* _V0 = PyList_GetItem(list, 48);
      Py_ssize_t _V0_size = PyList_Size(_V0);
      for (Py_ssize_t j = 0; j < _V0_size; j++) {
        V0[j] = PyFloat_AsDouble(PyList_GetItem(_V0, j));
        if (PyErr_Occurred()) return NULL;
      }

      //Print params
      printf("\n\n###################\n");
      printf("Parameters:\n\n");
      printf("N = %d\n", N);
      printf("Nrecord = %d\n", Nrecord);
      printf("T = %f\n", T);
      printf("Nt = %d\n", Nt);
      printf("Ne = %d\n", Ne);
      printf("Ni = %d\n", Ni);
      printf("q = %f\n", q);
      printf("dt = %f\n", dt);

      printf("pee0 = %f\n",pee0);
      printf("pei0 = %f\n",pei0);
      printf("pie0 = %f\n",pie0);
      printf("pii0 = %f\n",pii0);
      printf("jee = %f\n",jee);
      printf("jei = %f\n",jei);
      printf("jie = %f\n",jie);
      printf("jii = %f\n",jii);

      printf("wee0 = %f\n",wee0);
      printf("wei0 = %f\n",wei0);
      printf("wie0 = %f\n",wie0);
      printf("wii0 = %f\n",wii0);
      printf("Kee = %d\n",Kee);
      printf("Kei = %d\n",Kei);
      printf("Kie = %d\n",Kie);
      printf("Kii = %d\n",Kii);
      printf("taux = %f\n",taux);
      printf("mxe = %f\n",mxe);
      printf("mxi = %f\n",mxi);
      printf("vxe = %f\n",vxe);
      printf("vxi = %f\n",vxi);
      printf("tausyne = %f\n",tausyne);
      printf("tausyni = %f\n",tausyni);

      printf("tausynx = %f\n",tausynx);
      printf("Jee = %f\n",Jee);
      printf("Jei = %f\n",Jei);
      printf("Jie = %f\n",Jie);
      printf("Jii = %f\n",Jii);
      // printf("Ne1 = %d\n",Ne1);
      // printf("Ni1 =  %d\n",Ni1);

      printf("gl =  %f,%f\n",gl[0],gl[1]);
      printf("Cm = %f,%f\n",Cm[0],Cm[1]);
      printf("vlb = %f,%f\n",vlb[0],vlb[1]);
      printf("vth = %f,%f\n",vth[0],vth[1]);
      printf("DeltaT = %f,%f\n",DeltaT[0],DeltaT[1]);
      printf("vT = %f,%f\n",vT[0],vT[1]);
      printf("vl = %f,%f\n",vl[0],vl[1]);
      printf("vre = %f,%f\n",vre[0],vre[1]);
      printf("tref = %f,%f\n",tref[0],tref[1]);
      printf("###################\n\n");

      sim(N,Nrecord,T,Nt,Ne,Ni,q,dt,pee0,pei0,pie0,pii0,jee,jei,jie,jii,
          wee0,wei0,wie0,wii0,Kee,Kei,Kie,Kii,taux,mxe,mxi,vxe,vxi,
          tausyne,tausyni,tausynx,Jee,Jei,Jie,Jii,maxns,gl,Cm,vlb,vth,
          DeltaT,vT,vl,vre,tref,Irecord,V0,s,alphaxr,alphaer,alphair,vr,v,JnextE,
          JnextI,alphae,alphai,alphax,Wee,Wei,Wie,Wii,refstate,ffwd,ffwdr);

    npy_intp dims[2] = {Nt, Nrecord}; //row major order
    //Copy data into python list objects and free mem
    PyObject *alphaer_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(alphaer_out), alphaer, Nrecord*Nt*sizeof(double));
    free(alphaer);

    PyObject *alphair_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(alphair_out), alphair, Nrecord*Nt*sizeof(double));
    free(alphair);

    PyObject *alphaxr_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(alphaxr_out), alphaxr, Nrecord*Nt*sizeof(double));
    free(alphaxr);

    PyObject *vr_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(vr_out), vr, Nrecord*Nt*sizeof(double));
    free(vr);

    PyObject *ffwdr_out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(ffwdr_out), ffwdr, Nrecord*Nt*sizeof(double));
    free(ffwdr);

    npy_intp sdims[2] = {maxns,2};
    PyObject *s_out = PyArray_SimpleNew(2, sdims, NPY_DOUBLE);
    memcpy(PyArray_DATA(s_out), s, 2*maxns*sizeof(double));
    free(s);


    return Py_BuildValue("(OOOOOO)", s_out, vr_out, alphaer_out, alphair_out, alphaxr_out, ffwdr_out);
}


// int* _multi_gauss_ind(int N, double mu0, double sigma0) {
//
//   struct timeval tv; // Seed generation based on time
//   gettimeofday(&tv,0);
//   unsigned long mySeed = tv.tv_sec + tv.tv_usec;
//
//   gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
//   gsl_rng_set(r, mySeed);
//
//   gsl_vector *mu = gsl_vector_alloc(N);
//   gsl_vector *res = gsl_vector_alloc(N);
//   gsl_vector_set_all(mu, mu0);
//
//   gsl_matrix *cov = gsl_matrix_alloc(N,N);
//   gsl_matrix_set_identity(cov);
//   gsl_ran_multivariate_gaussian(r, mu, cov, res);
//
//   int j;
//   //copy to new array
//   for(j = 0; j < N; j++) {
//       ptr[j] = gsl_vector_get(res, j);
//     }
//
//   // gsl_vector_free(mu);
//   // gsl_vector_free(res);
//   // gsl_matrix_free(cov);
//
//   return res;
//
// }

// static PyObject* multi_gauss_ind(PyObject* Py_UNUSED(self), PyObject* args) {
//
//   /*
//   Function for multivariate gaussian with independent variables
//   This is equivalent to N independent gaussian random variables with mean mu0
//   and variance sigma0.
//   */
//
//   int N;
//   double mu0;
//   double sigma0;
//
//   if (!PyArg_ParseTuple(args, "idd", &N, &mu0, &sigma0))
//     return NULL;
//
//   ptr = _multi_gauss_ind(N, mu0, sigma0);
//
//   npy_intp dims[1] = {N};
//   PyObject *mg_out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
//   memcpy(PyArray_DATA(mg_out), ptr, N*sizeof(double));
//   free(ptr);
//
//   return Py_BuildValue("O", mg_out);
//
// }

void print_double_arr(double arr[], int SIZE) {
    int j;
    for(j = 0; j < SIZE; j++) {
        printf("%f\n",arr[j]);
    }
}

static PyMethodDef RNNMethods[] = {
    {"EIF", EIF, METH_VARARGS, "Python interface for EIF network in C"},
    {"FokkerPlanck_EIF", FokkerPlanck_EIF, METH_VARARGS, "Python interface for EIF network in C"},
    {"LR_EIF", LR_EIF, METH_VARARGS, "Python interface for EIF network in C"},
    {"FPT_EIF", FPT_EIF, METH_VARARGS, "Python interface for EIF network in C"},
    {NULL, NULL, 0, NULL},
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
