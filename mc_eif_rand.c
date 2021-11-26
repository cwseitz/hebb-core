#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <gsl_randist.h>
#include <gsl_rng.h>
#include <sys/time.h>
#include "complex.h"

void mc_eif_rand_sim(int N, int Nrecord, double T, int Nt, int Ne, int Ni, double q,
         double dt, double taux, double mxe, double mxi, double vxe, double vxi,
         double tausyne, double tausyni, double tausynx, double maxns, double *gl, double *C, double *vlb,
         double *vth, double *DeltaT, double *vT, double *vl, double *vre, double *tref,
         int *Irecord, double *v0, double *s, double *alphaxr,double *alphaer,
         double *alphair, double *vr, double *v, double *JnextE, double *JnextI, double *J,
         double *alphae, double *alphai, double *alphax, int *refstate, double *ffwd, double *ffwdr){

   int j, jj, k;

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
      if (i % 1000 == 0) {
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
               if(refstate[j]<=0){
                  v[j]+=(alphae[j]+alphai[j]+alphax[j]-gl[0]*(v[j]-vl[0])+gl[0]*DeltaT[0]*exp((v[j]-vT[0])/DeltaT[0]))*dt/C[0];
                  v[j]+=ffwd[j]*dt/C[1];
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
                    for(k=0;k<N;k++)
                           JnextE[k] += J[j*N+k];

                }
           }
           else{ /* If cell is inhibitory */
               ffwd[j] = mxi+gsl_ran_gaussian(r, sqrt(vxi));
               if(refstate[j]<=0){
                  v[j]+=(alphae[j]+alphai[j]+alphax[j]-gl[1]*(v[j]-vl[1])+gl[1]*DeltaT[1]*exp((v[j]-vT[1])/DeltaT[1]))*dt/C[1];
                  v[j]+=ffwd[j]*dt/C[1];
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
                   /* For each postsynaptic target */
                   for(k=0;k<N;k++)
                          JnextI[k] += J[j*N+k];
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

  }

  if(ns>=maxns)
     printf("Maximum number of spikes reached (%d), simulation terminated.\n", ns);

  return 0;

}

static PyObject* mc_eif_rand(PyObject* Py_UNUSED(self), PyObject* args) {

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
    double taux = PyFloat_AsDouble(PyList_GetItem(list, 8));
    double mxe = PyFloat_AsDouble(PyList_GetItem(list, 9));
    double mxi = PyFloat_AsDouble(PyList_GetItem(list, 10));
    double vxe = PyFloat_AsDouble(PyList_GetItem(list, 11));
    double vxi = PyFloat_AsDouble(PyList_GetItem(list, 12));
    double tausyne = PyFloat_AsDouble(PyList_GetItem(list, 13));
    double tausyni = PyFloat_AsDouble(PyList_GetItem(list, 14));
    double tausynx = PyFloat_AsDouble(PyList_GetItem(list, 15));
    double maxns = PyFloat_AsDouble(PyList_GetItem(list, 16));


    //Chunks of memory passed to the function as pointers
    PyObject* _gl = PyList_GetItem(list, 17);
    PyObject* _Cm = PyList_GetItem(list, 18);
    PyObject* _vlb = PyList_GetItem(list, 19);
    PyObject* _vth = PyList_GetItem(list, 20);
    PyObject* _DeltaT = PyList_GetItem(list, 21);
    PyObject* _vT = PyList_GetItem(list, 22);
    PyObject* _vl = PyList_GetItem(list, 23);
    PyObject* _vre = PyList_GetItem(list, 24);
    PyObject* _tref = PyList_GetItem(list, 25);
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
    double* alphae = malloc(N*sizeof(double));
    double* alphai = malloc(N*sizeof(double));
    double* alphax = malloc(N*sizeof(double));
    double* ffwd = malloc(N*sizeof(double));
    double* ffwdr = malloc(Nrecord * Nt * sizeof(double));
    int* refstate = malloc(N*sizeof(int));
    double* JnextE = malloc(N*sizeof(double));
    double* JnextI = malloc(N*sizeof(double));

    double nrecord = PyFloat_AsDouble(PyList_GetItem(list, 26));
    PyObject* _Irecord = PyList_GetItem(list, 27);
    int* Irecord = malloc(nrecord*sizeof(int));

    Py_ssize_t _Irecord_size = PyList_Size(_Irecord);
    for (Py_ssize_t j = 0; j < _Irecord_size; j++) {
      Irecord[j] = (int) PyLong_AsDouble(PyList_GetItem(_Irecord, j));
      if (PyErr_Occurred()) return NULL;
    }

    double* V0 = malloc(N*sizeof(double));
    PyObject* _V0 = PyList_GetItem(list, 28);
    Py_ssize_t _V0_size = PyList_Size(_V0);
    for (Py_ssize_t j = 0; j < _V0_size; j++) {
      V0[j] = PyFloat_AsDouble(PyList_GetItem(_V0, j));
      if (PyErr_Occurred()) return NULL;
    }

    double* J = malloc(N*N*sizeof(double));
    PyObject* _J = PyList_GetItem(list, 29);
    Py_ssize_t _J_size = PyList_Size(_J);
    for (Py_ssize_t j = 0; j < _J_size; j++) {
      J[j] = PyFloat_AsDouble(PyList_GetItem(_J, j));
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

    printf("taux = %f\n",taux);
    printf("mxe = %f\n",mxe);
    printf("mxi = %f\n",mxi);
    printf("vxe = %f\n",vxe);
    printf("vxi = %f\n",vxi);
    printf("tausyne = %f\n",tausyne);
    printf("tausyni = %f\n",tausyni);
    printf("tausynx = %f\n",tausynx);


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

    mc_eif_rand_sim(N,Nrecord,T,Nt,Ne,Ni,q,dt,taux,mxe,mxi,vxe,vxi,
        tausyne,tausyni,tausynx,maxns,gl,Cm,vlb,vth,
        DeltaT,vT,vl,vre,tref,Irecord,V0,s,alphaxr,alphaer,alphair,vr,v,
        JnextE,JnextI,J,alphae,alphai,alphax,refstate,ffwd,ffwdr);

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
