#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

void sim(int N, int Nrecord, double T, int Nt, int Ne, int Ni, double q,
         double dt, double pee0, double pei0, double pie0, double pii0,
         double jee, double jei, double jie, double jii, double wee0,
         double wei0, double wie0, double wii0, int Kee, int Kei, int Kie,
         int Kii, double taux, double mxe0, double mxi0, double vxe, double vxi,
         double tausyne, double tausyni, double tausynx, double Jee, double Jei,
         double Jie, double Jii, double maxns, double *gl, double *C, double *vlb,
         double *vth, double *DeltaT, double *vT, double *vl, double *vre, double *tref,
         double *Ix1e, double *Ix2e, double *Ix1i, double* Ix2i, int nrecord,
         int *Irecord, double *v0, double rxe, double rxi, double Jex, double Jix,
         int Ne1, int Ni1, double *s, double *alphaxr,double *alphaer,double *alphair,
         double *vr, double *v, double *JnextE, double *JnextI, double *alphae,
         double *alphai, double *alphax, int *Wee,int *Wei,int *Wie,int *Wii,int *refstate){

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
     }


  for(jj=0;jj<Nrecord;jj++){
        j=(int)round(Irecord[jj]-1);
        alphaer[jj+Nrecord*0]=alphae[j];
        alphair[jj+Nrecord*0]=alphai[j];
        alphaxr[jj+Nrecord*0]=alphax[j];
        vr[jj+Nrecord*0]=v[j];
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

  // printf("Spikes: %f\n", ns);
  // printf("Max Spikes: %f\n", maxns);
  for(i=1;i<Nt && ns<maxns;i++){
       printf("Time step: %d/%d\n", i, Nt);
       for(j=0;j<N;j++){
           alphae[j]-=alphae[j]*(dt/tausyne);
           alphai[j]-=alphai[j]*(dt/tausyni);
           alphax[j]-=alphax[j]*(dt/tausynx);

           if(j<Ne){
               if(refstate[j]<=0){
                  v[j]+=(alphae[j]+alphai[j]+alphax[j]-gl[0]*(v[j]-vl[0])+gl[0]*DeltaT[0]*exp((v[j]-vT[0])/DeltaT[0]))*dt/C[0];
                  if(j<Ne1)
                      v[j]+=Ix1e[i]*dt/C[0];
                  else
                      v[j]+=Ix2e[i]*dt/C[0];
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

               if(refstate[j]<=0){
                  v[j]+=(alphae[j]+alphai[j]+alphax[j]-gl[1]*(v[j]-vl[1])+gl[1]*DeltaT[1]*exp((v[j]-vT[1])/DeltaT[1]))*dt/C[1];
                  if(j<Ne+Ni1)
                      v[j]+=Ix1i[i]*dt/C[1];
                  else
                      v[j]+=Ix2i[i]*dt/C[1];
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
              vr[jj+Nrecord*i]=v[j];
          }
          /* Propagate spikes */
          for(j=0;j<N;j++){
            alphae[j]+=JnextE[j]/tausyne;
            alphai[j]+=JnextI[j]/tausyni;
            JnextE[j]=0;
            JnextI[j]=0;
          }

          for(j=0;j<Ne;j++)
              if(drand48()<rxe*dt)
                  alphax[j]+=Jex/tausynx;

          for(j=Ne;j<N;j++)
              if(drand48()<rxi*dt)
                  alphax[j]+=Jix/tausynx;
  }

  if(ns>=maxns)
     printf("Maximum number of spikes reached (%d), simulation terminated.\n", ns);

  return 0;

}

  static PyObject* lif(PyObject* Py_UNUSED(self), PyObject* args) {

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
      double mxe0 = PyFloat_AsDouble(PyList_GetItem(list, 25));
      double mxi0 = PyFloat_AsDouble(PyList_GetItem(list, 26));
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

      double rxe = PyFloat_AsDouble(PyList_GetItem(list, 53));
      double rxi = PyFloat_AsDouble(PyList_GetItem(list, 54));
      double Jex = PyFloat_AsDouble(PyList_GetItem(list, 55));
      double Jix = PyFloat_AsDouble(PyList_GetItem(list, 56));
      int Ne1 = (int) PyFloat_AsDouble(PyList_GetItem(list, 57));
      int Ni1 = (int) PyFloat_AsDouble(PyList_GetItem(list, 58));

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
      int* Wee = malloc(Ne*Kee*sizeof(int));
      int* Wei = malloc(Ni*Kei*sizeof(int));
      int* Wie = malloc(Ne*Kie*sizeof(int));
      int* Wii = malloc(Ni*Kii*sizeof(int));
      int* refstate = malloc(N*sizeof(int));
      double* Ix1e = malloc(Nt*sizeof(double));
      double* Ix2e = malloc(Nt*sizeof(double));
      double* Ix1i = malloc(Nt*sizeof(double));
      double* Ix2i = malloc(Nt*sizeof(double));

      //Feedforward inputs passed to the sim function as pointers
      PyObject* _Ix1e = PyList_GetItem(list, 46);
      PyObject* _Ix2e = PyList_GetItem(list, 47);
      PyObject* _Ix1i = PyList_GetItem(list, 48);
      PyObject* _Ix2i = PyList_GetItem(list, 49);

      Py_ssize_t _Ix1e_size = PyList_Size(_Ix1e);
      for (Py_ssize_t j = 0; j < _Ix1e_size; j++) {
        Ix1e[j] = PyFloat_AsDouble(PyList_GetItem(_Ix1e, j));
        Ix2e[j] = PyFloat_AsDouble(PyList_GetItem(_Ix2e, j));
        Ix1i[j] = PyFloat_AsDouble(PyList_GetItem(_Ix1i, j));
        Ix2i[j] = PyFloat_AsDouble(PyList_GetItem(_Ix2i, j));
        if (PyErr_Occurred()) return NULL;
      }

      double nrecord = PyFloat_AsDouble(PyList_GetItem(list, 50));
      PyObject* _Irecord = PyList_GetItem(list, 51);
      int* Irecord = malloc(nrecord*sizeof(int));

      Py_ssize_t _Irecord_size = PyList_Size(_Irecord);
      for (Py_ssize_t j = 0; j < _Irecord_size; j++) {
        Irecord[j] = (int) PyLong_AsDouble(PyList_GetItem(_Irecord, j));
        if (PyErr_Occurred()) return NULL;
      }

      int* V0 = malloc(N*sizeof(double));
      PyObject* _V0 = PyList_GetItem(list, 52);
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
      printf("mxe0 = %f\n",mxe0);
      printf("mxi0 = %f\n",mxi0);
      printf("vxe = %f\n",vxe);
      printf("vxi = %f\n",vxi);
      printf("tausyne = %f\n",tausyne);
      printf("tausyni = %f\n",tausyni);

      printf("tausynx = %f\n",tausynx);
      printf("Jee = %f\n",Jee);
      printf("Jei = %f\n",Jei);
      printf("Jie = %f\n",Jie);
      printf("Jii = %f\n",Jii);
      printf("Ne1 = %d\n",Ne1);
      printf("Ni1 =  %d\n",Ni1);

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
          wee0,wei0,wie0,wii0,Kee,Kei,Kie,Kii,taux,mxe0,mxi0,vxe,vxi,
          tausyne,tausyni,tausynx,Jee,Jei,Jie,Jii,maxns,gl,Cm,vlb,vth,
          DeltaT,vT,vl,vre,tref,Ix1e,Ix2e,Ix1i,Ix2i,nrecord,Irecord,V0,
          rxe,rxi,Jex,Jix,Ne1,Ni1,s,alphaxr,alphaer,alphair,vr,v,JnextE,
          JnextI,alphae,alphai,alphax,Wee,Wei,Wie,Wii,refstate);


    npy_intp dims[2] = {Nrecord, Nt}; //row major order
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

    return Py_BuildValue("(OOOO)", vr_out, alphaer_out, alphair_out, alphaxr_out);
}

void print_double_arr(double arr[], int SIZE) {
    int j;
    for(j = 0; j < SIZE; j++) {
        printf("%.2f\n",arr[j]);
    }
}

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
