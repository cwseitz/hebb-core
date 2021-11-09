#include <Python.h>

void sim(){

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
        /* Find index into local variables */

          /* Find index into local variables */
          j=(int)round(Irecord[jj]-1);

        if(j>=N || j<0)
           mexErrMsgTxt("Irecord contains out of bounds indices.");

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


  Ntref[0]=(int)round(tref[0]/dt);
  Ntref[1]=(int)round(tref[1]/dt);

  ns=0;
  flag=0;

  for(i=1;i<Nt && ns<maxns;i++){
       for(j=0;j<N;j++){

           alphae[j]-=alphae[j]*(dt/tausyne);
           alphai[j]-=alphai[j]*(dt/tausyni);
           alphax[j]-=alphax[j]*(dt/tausynx);

           if(j<Ne){
               if(refstate[j]<=0){
                  v[j]+=(alphae[j]+alphai[j]+alphax[j]-gl[0]*(v[j]-Vleak[0])+gl[0]*DeltaT[0]*exp((v[j]-VT[0])/DeltaT[0]))*dt/C[0];
                  if(j<Ne1)
                      v[j]+=Ix1e[i]*dt/C[0];
                  else
                      v[j]+=Ix2e[i]*dt/C[0];
                  v[j]=fmax(v[j],Vlb[0]);
               }
               else{
                  if(refstate[j]>1)
                     v[j]=Vth[0];/*-=(Vth[0]-Vre[0])/((double)Ntref[0]);*/
                  else
                     v[j]=Vre[0];
                  refstate[j]--;
               }
               /* If a spike occurs */
               if(v[j]>=Vth[0] && refstate[j]<=0 && ns<maxns){

                    refstate[j]=Ntref[0];
                    v[j]=Vth[0];       /* reset membrane potential */
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
                  v[j]+=(alphae[j]+alphai[j]+alphax[j]-gl[1]*(v[j]-Vleak[1])+gl[1]*DeltaT[1]*exp((v[j]-VT[1])/DeltaT[1]))*dt/C[1];
                  if(j<Ne+Ni1)
                      v[j]+=Ix1i[i]*dt/C[1];
                  else
                      v[j]+=Ix2i[i]*dt/C[1];
                  v[j]=fmax(v[j],Vlb[1]);
               }
               else{
                  if(refstate[j]>1)
                     v[j]=Vth[1];/*-=(Vth[1]-Vre[1])/((double)Ntref[1]);*/
                  else
                     v[j]=Vre[1];
                  refstate[j]--;
               }

                /* If a spike occurs */
                if(v[j]>=Vth[1] && refstate[j]<=0 && ns<maxns){

                    refstate[j]=Ntref[1];
                    v[j]=Vth[1];       /* reset membrane potential */
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
              if(j>=N || j<0)
                  mexErrMsgTxt("Bad index in Irecord.");

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

}

static PyObject* lif(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;

    int N = PyFloat_AsDouble(PyList_GetItem(list, 0));
    int Nrecord = PyFloat_AsDouble(PyList_GetItem(list, 1));
    int T = PyFloat_AsDouble(PyList_GetItem(list, 2));
    int Nt = PyFloat_AsDouble(PyList_GetItem(list, 3));
    int N_e = PyFloat_AsDouble(PyList_GetItem(list, 4));
    int N_i = PyFloat_AsDouble(PyList_GetItem(list, 5));
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

    double Kee = PyFloat_AsDouble(PyList_GetItem(list, 20));
    double Kei = PyFloat_AsDouble(PyList_GetItem(list, 21));
    double Kie = PyFloat_AsDouble(PyList_GetItem(list, 22));
    double Kii = PyFloat_AsDouble(PyList_GetItem(list, 23));

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
    double* Wee = malloc(N_e*Kee*sizeof(int));
    double* Wei = malloc(N_i*Kei*sizeof(int));
    double* Wie = malloc(N_e*Kie*sizeof(int));
    double* Wii = malloc(N_i*Kii*sizeof(int));
    double* refstate = malloc(N*sizeof(int));

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

    double* Ix1e = malloc(Nt*sizeof(int));
    double* Ix2e = malloc(Nt*sizeof(int));
    double* Ix1i = malloc(Nt*sizeof(int));
    double* Ix2i = malloc(Nt*sizeof(int));

    PyObject* _Ix1e = PyList_GetItem(list, 46);
    PyObject* _Ix2e = PyList_GetItem(list, 47);
    PyObject* _Ix1i = PyList_GetItem(list, 48);
    PyObject* _Ix2i = PyList_GetItem(list, 49);

    // Py_ssize_t _Ix1e_size = PyList_Size(_Ix1e);
    // for (Py_ssize_t j = 0; j < _Ix1e_size; j++) {
    //   Ix1e[j] = PyFloat_AsDouble(PyList_GetItem(_Ix1e, j));
    //   Ix2e[j] = PyFloat_AsDouble(PyList_GetItem(_Ix2e, j));
    //   Ix1i[j] = PyFloat_AsDouble(PyList_GetItem(_Ix1i, j));
    //   Ix2i[j] = PyFloat_AsDouble(PyList_GetItem(_Ix2i, j));
    //   if (PyErr_Occurred()) return NULL;
    // }

    // double nrecord = PyFloat_AsDouble(PyList_GetItem(list, 50));
    // PyObject* _Irecord = PyList_GetItem(list, 51);
    // double* Irecord = malloc(nrecord*sizeof(int));
    //
    // Py_ssize_t _Irecord_size = PyList_Size(_Irecord);
    // for (Py_ssize_t j = 0; j < _Irecord_size; j++) {
    //   Irecord[j] = PyFloat_AsDouble(PyList_GetItem(_Irecord, j));
    //   if (PyErr_Occurred()) return NULL;
    // }

    // sim(Ix1e,Ix2e,Ix1i,Ix2i,Ne,Ni,Ne1,Ni1,Jex,Jix,Jee,Jei,Jie,Jii,
    //                  rxe,rxi,Kee,Kei,Kie,Kii,Cm,gl,vl,DeltaT,vT,tref,vth,vre,vlb,
    //                  tausynx,tausyne,tausyni,V0,T,dt,maxns,Irecord)

  Py_RETURN_NONE;
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
    return PyModule_Create(&hebb_backend);
}
