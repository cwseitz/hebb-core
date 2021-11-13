#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <gsl_randist.h>
#include <gsl_rng.h>
#include <sys/time.h>


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
        printf("%.2f\n",arr[j]);
    }
}

static PyMethodDef RNNMethods[] = {
    {"EIF", EIF, METH_VARARGS, "Python interface for EIF network in C"},
    //{"multi_gauss_ind", multi_gauss_ind, METH_VARARGS, "Generating normally distributed nums" },
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
