#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#ifdef _WIN32
#include <conio.h>
//for now only for 1 chain(!)
//works with CUDA neighbor list
#endif
//#include <omp.h>
#include <time.h>
#include <cuda_runtime_api.h>
//#include <windows.h>
bool debug = false; //turns true after some crushes
bool UPOROT = false; // for special fucked up potentials, don't you touch
//// GLOBALS
float SIZE[3]={500,500,500};// SIMULATION CELL SIZE, PBC in all dimensions
 const float OBR_SIZE[3]={0.002,0.002,0.002}; // reverse sizes, should be written
 float *cudaSize;
 float *cudaObrSize;
 
 
 
const double BONDMIN=0.8; const double BONDMAX=1.25; // bond length bounds
int Nchains; // works only for ONE chain now
const int MAXNEIGB = 200; // number of neighbor list each monomer
const int NMaxChains = 2;
const int NMax = 512; //Should be changed to real number of monomers, is critical due to CUDA-memcpys.
const int FIXED_POINT = -100500; //set fixed monomer if necessary
#define Nmax NMax

//For communication with CUDA
float coords[3*NMax*NMaxChains]; 
int neighborList[NMax*MAXNEIGB];
int neighborLen[NMax];
float *cudaCoords;
int *cudaneighborList;
int *cudaneighborLen;
int *cudaBonds;
int Bonds[NMax*10];
float *cudaList;
float testlist[NMax*NMax];
//

int NParticles = 0;
// max monomers in chain
const int MONTYPES = 3; // monomer types for interaction and stiffness matrix
double EPS[MONTYPES][MONTYPES]; //Van-der-Waals epsilons
double EPSST[MONTYPES][MONTYPES][MONTYPES]; //stiffness epsilons
double MON_SIZES[MONTYPES]; // not yet used
const double POT_CUT = 2;// potential cutoff
const double POT_CUT_SQ = 4;
const double NOT_POSSIBLE=-150500; // Kostyl'!:)
const double PARTICLE_SIZE = 1;
const double NEIGHBOR_CUT = 3; // neighbor cutoff
const float NEIGHBOR_CUTSQ = 9;
#define NEIGBCUT 9

double CurrentE;
long int STEPS = 200000000;

//const bool PERIODIC_BOUNDARIES[3]; // not yet used
class SAMC;
struct Chain;
struct Mon;
struct Vector;


double DEnergyNV(const Mon&, int , int);
double DEnergyStiff(const Mon&, int, int);
double ECos(double);
bool CheckAllBonds();

void BuildNeighborTable(double Cut);

void outputVrml(char* filename);
void outputPeriodicVrml(char* filename);
void outputGradVrml(char* filename);
void PrintDistMatrix(int ChNum, FILE* f);
void GenerateHomopolymer(int, int);
void GenerateAmphiphilic(int len, int Typ1, int Typ2, bool RandomGrafting = false, int LenGrafts = 1, int GraftingInterval = 1);
void GenerateGraft(int len, int ch, int startMon, int Typ1);
void GenerateDiblock(int len1, int len2, int Typ1, int Typ2);


void GenerateGradient(int len, int Typ1, int Typ2,int Typ3, double frequency_typ3);
void GenerateRing(int len, int Typ1); 
void GenerateStar(int numChains, int *Lens, int *Typs, int MidType); // TODO : not yet written

void ReadConf();
void StoreConf();

void parseCoordsFromDendrimer(char filename[40], int length);
////chain stats

void ConfigurationalBias(int nChain, int Typ, bool (* Accept)(double,double),int length, bool end);
void ReverseChain(int);

void InitStats(int Len, int StpLen, int minlen, int maxlen,int);
void CalculateStats();
void outputStats(FILE *f);



//some shit for fractal globule statistics

double rgs[5000]; //starts with 2
double rs[5000];
double conts[5000];
int rgstep;
int rgmax;
int rglen;
int rgmin;
int rgcounter;
int rgstart = 0;
//acception rates
int CBAccrate;
int simpleaccrate;
bool firstTable;
char dump[400];

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int testmons[4000];
int ntestmons = -100;
const int NCELLS = 100;
//unsigned short LinkedCellChainList[NCELLS][NCELLS][NCELLS][100];
//unsigned short LinkedCellList[NCELLS][NCELLS][NCELLS][100];


// HSV to RGB for gradient coloring



struct rgb{
    double r;       // percent
    double g;       // percent
    double b;       // percent
} ;
 struct hsv{
    double h;       // angle in degrees
    double s;       // percent
    double v;       // percent
} ;

hsv   rgb2hsv(rgb in);
rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0              
            // s = 0, v is undefined
        out.s = 0.0;
        out.h = 0;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}


rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;     
}











//simple SAMC class with linear weights; Static initialization, no dynarrays, works
class SAMC{

private:
double **GE;
int **HIST;
int NVMin;
int STMin,STMax,NVMax;
double GSAMC;
int TSAMC;
double MinusShift;
bool initialized;
double LastENV;
double LastEST;
long long counter_T;
int TABLE_SIZE;
int NVWindowMin,NVWindowMax;

public:
void init(int ENVMin, int ENVMax, int ESTMin, int ESTMax, double Gamma, int t, double BeginENV, double BeginESTIFF,int MIN, int MAX) // MIN and MAX - energy window boundaries.
{
    NVMin = ENVMin;
     NVMax = ENVMax;
     STMax= ESTMax;
    STMin = ESTMin;
   GSAMC = Gamma;
   LastENV = BeginENV;
    LastEST = BeginESTIFF;
     NVWindowMin = MIN;
	NVWindowMax = MAX;
    TABLE_SIZE = (ENVMax-ENVMin)*(ESTMax-ESTMin);
    GE = new double* [ENVMax-ENVMin];
    HIST = new int* [ENVMax-ENVMin];
    for(int i = 0; i < ENVMax-ENVMin; i++)
    {
        GE[i] = new double[ESTMax-ESTMin];
        HIST[i] = new int[ESTMax-ESTMin];
        for(int j = 0; j < ESTMax-ESTMin; j++)
        {
            GE[i][j] = 0; HIST[i][j] = 0;
        }
    }
   TSAMC = t;
    MinusShift = 0; // -gamma on each step;
    initialized = true;
    counter_T = 1;
}

bool AcceptStep( double denv, double destiff)
{
    double rnd = (double)rand()/(double)RAND_MAX;
    double prb = (GE[(int)round(LastENV)-NVMin][(int)round(LastEST)-STMin])-(GE[(int)round(LastENV+denv)-NVMin][(int)round(LastEST+destiff)-STMin]);
    //if((int)round(LastENV+denv) > )
    if(LastENV+denv > NVWindowMax || LastEST+denv < NVWindowMin)
    {
        return false;
    }
    if(rnd < exp(prb) || prb > 30)
    {
        if(LastENV+denv > 0)
        {
        //    printf("shit fuck crap");
        }
        GE[(int)round(LastENV+denv)-NVMin][(int)round(LastEST+destiff)-STMin] += GSAMC;
        HIST[(int)round(LastENV+denv)-NVMin][(int)round(LastEST+destiff)-STMin] ++;
        MinusShift += GSAMC/(double)TABLE_SIZE;
        LastENV = LastENV+denv;
        LastEST = LastEST+destiff;
        ++counter_T;
        GSAMC = fmin(GSAMC,(double)TSAMC/(double)counter_T);
        return true;
    }
    else
    {
        GE[(int)round(LastENV)-NVMin][(int)round(LastEST)-STMin] += GSAMC;
        HIST[(int)round(LastENV)-NVMin][(int)round(LastEST)-STMin] ++;
        MinusShift += GSAMC/(double)TABLE_SIZE;
        ++counter_T;
        GSAMC = fmin(GSAMC,(double)TSAMC/(double)counter_T);
        return false;
    }




}

void printHist(const char* filename,const char* mode)
{
    FILE* fp = fopen(filename,mode);
    fprintf(fp,"\n");
    fprintf(fp,"HIST ");
    for(int k = 0; k < STMax-STMin; k++)
    {
        fprintf(fp,"%i ",STMin+k);
    }
    fprintf(fp,"\n");
    for(int i = 0; i < NVMax-NVMin; i++)
    {
        fprintf(fp,"%i ",NVMin+i);
        for(int j = 0; j < STMax-STMin; j ++)
        {
            fprintf(fp,"%i ", HIST[i][j]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);

}
void printGE(const char* filename,const char *mode)
{
    FILE* fp = fopen(filename,mode);
    fprintf(fp,"\n");
    fprintf(fp,"GE ");
    for(int k = 0; k < STMax-STMin; k++)
    {
        fprintf(fp,"%i ",STMin+k);
    }
    fprintf(fp,"\n");
    for(int i = 0; i < NVMax-NVMin; i++)
    {
        fprintf(fp,"%i ",NVMin+i);
        for(int j = 0; j < STMax-STMin; j ++)
        {
            fprintf(fp,"%f ", GE[i][j]-MinusShift);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}




~SAMC()
{

    for(int i = 0; i < NVMax-NVMin; i++)
    {
        delete[] GE[i] ;
        delete[] HIST[i];




    }

    delete[] GE;
    delete[] HIST;



}
} Samc;


bool AcceptSAMC(double env, double est)
{
    return Samc.AcceptStep(env,est);
}






struct Vector
{
public: double x[3];
       double L;
       Vector(double x1,double y1,double z1)
       {
                  x[0] = x1;
                  x[1] = y1;
                  x[2] = z1;

       //           L = sqrt(x1*x1 + y1*y1 + z1*z1);

       }
       Vector()
       {
       }


       Vector& operator = (const Vector &n)
       {
              x[0] = n.x[0];
              x[1] = n.x[1];
              x[2] = n.x[2];

 //                 L = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
       return *this;
       }
       bool operator == (const Vector &n)
       {
       if(x[0] == n.x[0] && x[1] == n.x[1] && x[2] == n.x[2])
       {
       return true;
       }
       else
       {
       return false;
       }
       }
       double len()
       {
       return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
       }
       double sqlen()
       {
           return x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
       }

        Vector getReal()
        {
            Vector tmp;
            for(int i = 0; i < 3; i++)
            {

            tmp.x[i]=x[i]-SIZE[i]*floor(x[i]/SIZE[i]);


            }

        return tmp;
        }




};

Vector operator+(const Vector &a,const Vector &b)
       {

          Vector c(a.x[0]+b.x[0],a.x[1]+b.x[1],a.x[2]+b.x[2]);
          return c;
       }
Vector operator-(const Vector &a,const Vector &b)
       {
          Vector c(a.x[0]-b.x[0],a.x[1]-b.x[1],a.x[2]-b.x[2]);
          return c;
       }



Vector getReal(const Vector &un)
        {
        Vector tmp;
        for(int i = 0; i < 3; i++)
        {

        tmp.x[i]=un.x[i]-SIZE[i]*floor(un.x[i]/SIZE[i]);


        }

        return tmp;
        }

double operator*(const Vector &a,const Vector &b)
{
     double c = a.x[0]*b.x[0] +a.x[1]*b.x[1]+a.x[2]*b.x[2];
          return c;

}

Vector operator*(const double a,const Vector &b)
{
 Vector tmp;
 tmp = b;
 for(int i = 0; i < 3; i++)
        {

        tmp.x[i]=a*tmp.x[i];


        }
return tmp;
}



///// monomer class

struct Mon
{
int coor;
Vector X;
Vector XReal;
//double CurrentE;
int Typ;
int Bonds[5];
int NumBonds;
int BondChains[5]; // will be deprecated someday
int NeighborList[200];
int NeighborChainList[200]; //will be deprecated
int Neighbors;
int number;
//int StiffType[5];
//int ParentChain;
double Size;
bool Moved;
Mon()
{
}

Mon(const Mon &v)
{
    coor = v.coor;
	X = v.X;
    XReal = v.XReal;
    NumBonds = v.NumBonds;
    Size = v.Size;
    Typ = v.Typ;
    Moved = v.Moved;
    for(int j=0 ; j< NumBonds;j++)
    {
        Bonds[j] = v.Bonds[j];
        BondChains[j]=v.BondChains[j];
       // StiffType[j] = v.StiffType[j];
    }
    Neighbors = v.Neighbors;
    for(int j=0 ;j < Neighbors; j++)
    {
        NeighborList[j] = v.NeighborList[j];
        NeighborChainList[j] = v.NeighborChainList[j];
    }

}

Mon operator = (const Mon &v)
{
    coor = v.coor;
	X = v.X;
    XReal = v.XReal;
    NumBonds = v.NumBonds;
    Typ = v.Typ;
    Moved = v.Moved;
    for(int j=0; j< NumBonds;j++)
    {
        Bonds[j] = v.Bonds[j];
        BondChains[j]=v.BondChains[j];
      //  StiffType[j] = v.StiffType[j];
    }
    Neighbors = v.Neighbors;
    for(int j=0;j < Neighbors; j++)
    {
        NeighborList[j] = v.NeighborList[j];
        NeighborChainList[j] = v.NeighborChainList[j];
    }
    return *this;
}

void update() // pushing coordinates to CUDA array
{
   // printf("mon %i coords %f %f %f\n",coor,XReal.x[0],XReal.x[1],XReal.x[2]);
	coords[coor*3] = (float)XReal.x[0];
    coords[coor*3+1] = (float)XReal.x[1];
    coords[coor*3+2] = (float)XReal.x[2];
}

bool bondedWith(const int nChain,const int nMon)
{
 int i;
 for(i = 0 ; i < NumBonds; i++)
 {
    if(Bonds[i] == nMon &&  BondChains[i] == nChain)
    {
        return true;
    }


 }

return false;
}

bool neighborWith(const int nChain,const int nMon)
{
 int i;
 for(i = 0 ; i < neighborLen[coor]; i++)
 {
    if(neighborList[i] == nMon )
    {
        return true;
    }


 }

return false;
}


void bondWith(const int nChain,const int nMon)
{
      Bonds[NumBonds] = nMon;
      BondChains[NumBonds] = nChain;
      NumBonds++;





}



Mon operator+ (const Vector &v)
{
      Mon c = *this;
        c.XReal=c.XReal+v;
      //  c.X = getReal(c.XReal);
        return c;
}


  bool checkBonds();

};





///// end monomer class


double nearestImageR(const Mon &a,const Mon &b)
{
    Vector coor;
    for(int i=0; i < 3; i++)
    {
        coor.x[i] = a.XReal.x[i]-b.XReal.x[i];
        coor.x[i] = coor.x[i] - SIZE[i]*round(coor.x[i]/SIZE[i]);
    }
    return coor.len();
}




double nearestImageSq(const Mon &a,const Mon &b)
{
    Vector coor;

        coor.x[0] = a.XReal.x[0]-b.XReal.x[0];
	 coor.x[1] = a.XReal.x[1]-b.XReal.x[1];
	 coor.x[2] = a.XReal.x[2]-b.XReal.x[2];
//if(SIZE[i]*round(coor.x[i]/SIZE[i])!=0)
        //{
          //  printf ("%f real dist %f per dist \n", coor.x[i],coor.x[i] - SIZE[i]*round(coor.x[i]/SIZE[i]));
           // getch();
        //}
    //    coor.x[i] = coor.x[i] ;


    coor.x[0] -= SIZE[0]*round(coor.x[0]*OBR_SIZE[0]);
    coor.x[1] -= SIZE[1]*round(coor.x[1]*OBR_SIZE[1]);
    coor.x[2] -= SIZE[2]*round(coor.x[2]*OBR_SIZE[2]);

return coor.sqlen();
}
Vector operator-(const Mon &a, const Mon &b)
{

    Vector coor;
    for(int i=0; i<3; i++)
    {
        coor.x[i] = a.XReal.x[i]-b.XReal.x[i];
        coor.x[i] = coor.x[i] - SIZE[i]*round(coor.x[i]/SIZE[i]);
    }
    return coor;
}


/////////////////////////////////////////////////////////////////////////////////

struct Chain
{
public:
Mon M[Nmax];
int N;




};

Chain C[NMaxChains];

////////////////////////////////////////////////////////////////////

bool Mon :: checkBonds()
  {

     for(int i=0; i < NumBonds; i++)
     {
         // printf("checking bonds %f bond n %i \n ",nearestImageR(*this,C[BondChains[i]].M[Bonds[i]]),i);
        //  getch();
          if((XReal - C[BondChains[i]].M[Bonds[i]].XReal).len() > BONDMAX || (XReal - C[BondChains[i]].M[Bonds[i]].XReal).len() < BONDMIN)
          { //printf("bond fail\n"); r
          return false;
          }
     }

     return true;
  }







//random chain and monomer choice

void SimpleStep(bool (* Accept)(double envf, double estf) )
{
Vector rndVect;




for(int i = 0; i < 3 ; i++)
{

    rndVect.x[i] = 0.1 - 0.2*(double)rand()/RAND_MAX;
}



double TestE,DENV,DESTIFF;


bool accepted = false;
int rndChain = rand()%Nchains;
int rndMono = rand()%C[rndChain].N ;
if(rndMono == FIXED_POINT) 
{
	
	return;
}
Mon Montest = C[rndChain].M[rndMono]+ rndVect;
if(Montest.checkBonds() == true)
{


     TestE = DEnergyNV(Montest,rndChain,rndMono);
     if( TestE != NOT_POSSIBLE)
     {
        // DESTIFF = DEnergyStiff(C[rndChain].M[rndMono]+rndVect,rndChain,rndMono);
         DENV = TestE;
        // printf( "%f , %f %i- energnv enerst typ check %d\n", DENV,DESTIFF,C[rndChain].M[rndMono].Typ,Montest.checkBonds() );
       //  getch();
	accepted = true;
	 if(Accept(DENV,0))
         {
            if(debug)
            {


            printf("%i mono %f %f %f was plus",rndMono,C[rndChain].M[rndMono].XReal.x[0],C[rndChain].M[rndMono].XReal.x[1],C[rndChain].M[rndMono].XReal.x[2]);
            printf("%f %f %f\n",rndVect.x[0],rndVect.x[1],rndVect.x[2]);
            }

             C[rndChain].M[rndMono] = C[rndChain].M[rndMono]+ rndVect;
             simpleaccrate++;
             C[rndChain].M[rndMono].update();
             if(debug)
             {
                 printf("is %f %f %f \n",C[rndChain].M[rndMono].XReal.x[0],C[rndChain].M[rndMono].XReal.x[1],C[rndChain].M[rndMono].XReal.x[2]);

             }

            // getch();
         }
     }



}
//else
//{
//SimpleStep(Accept);
//return;
//}
if(accepted == false)
{
//accepted = Accept(0,0);
}
}


void AllStep(bool (* Accept)(double envf, double estf))
{

   Vector rndVect;
   double DENV,TestE;
    int rndChain, rndMono;
   for(int i = 0; i < Nchains; i++)
   {

    for(int s = 0; s < C[i].N; s++)
    {


    rndChain = i;
    rndMono = s;


    for(int i = 0; i < 3 ; i++)
    {

    rndVect.x[i] = 0.2 - 0.4*(double)rand()/RAND_MAX;
    }

    Mon Montest = C[rndChain].M[rndMono]+ rndVect;
    if(Montest.checkBonds() == true)
    {


     TestE = DEnergyNV(Montest,rndChain,rndMono);
     if( TestE != NOT_POSSIBLE)
     {
        // DESTIFF = DEnergyStiff(C[rndChain].M[rndMono]+rndVect,rndChain,rndMono);
         DENV = TestE;
        // printf( "%f , %f %i- energnv enerst typ check %d\n", DENV,DESTIFF,C[rndChain].M[rndMono].Typ,Montest.checkBonds() );
       //  getch();
       // accepted = true;
        if(Accept(DENV,0))
            {
            if(debug)
            {


            printf("%i mono %f %f %f was plus",rndMono,C[rndChain].M[rndMono].XReal.x[0],C[rndChain].M[rndMono].XReal.x[1],C[rndChain].M[rndMono].XReal.x[2]);
            printf("%f %f %f\n",rndVect.x[0],rndVect.x[1],rndVect.x[2]);
            }

             C[rndChain].M[rndMono] = C[rndChain].M[rndMono]+ rndVect;
             simpleaccrate++;

             if(debug)
             {
                 printf("is %f %f %f \n",C[rndChain].M[rndMono].XReal.x[0],C[rndChain].M[rndMono].XReal.x[1],C[rndChain].M[rndMono].XReal.x[2]);
             }

            // getch();
         }
     }



}
}
   }
}


//should be tested, non-ergodic
void TargetedStep(bool (* Accept)(double envf, double estf) , int Ch, int RangeBegin, int RangeEnd)
{

Vector rndVect;
for(int i = 0; i < 3 ; i++)
{

    rndVect.x[i] = 0.2 - 0.4*(double)rand()/RAND_MAX;
}

double TestE,DENV,DESTIFF;



int rndChain = Ch;
int rndMono = rand()%(RangeEnd-RangeBegin) + RangeBegin ;
Mon Montest = C[rndChain].M[rndMono]+ rndVect;
if(Montest.checkBonds() == true)
{


     TestE = DEnergyNV(Montest,rndChain,rndMono);
     if( TestE != NOT_POSSIBLE)
     {
         DESTIFF = 0;//DEnergyStiff(C[rndChain].M[rndMono]+rndVect,rndChain,rndMono);
         DENV = TestE;
        // printf( "%f , %f %i- energnv enerst typ check %d\n", DENV,DESTIFF,C[rndChain].M[rndMono].Typ,Montest.checkBonds() );
       //  getch();
         if(Accept(DENV,DESTIFF))
         {
         //   printf("%f %f %f was plus",C[rndChain].M[rndMono].X.x[0],C[rndChain].M[rndMono].X.x[1],C[rndChain].M[rndMono].X.x[2]);
          //   printf("%f %f %f\n",rndVect.x[0],rndVect.x[1],rndVect.x[2]);
             C[rndChain].M[rndMono] = C[rndChain].M[rndMono]+ rndVect;
             simpleaccrate++;
          //  printf("is %f %f %f \n",C[rndChain].M[rndMono].X.x[0],C[rndChain].M[rndMono].X.x[1],C[rndChain].M[rndMono].X.x[2]);
            // getch();
         }
     }



}
}


//energy

double DEnergyNV(const Mon& newMon,int Ch, int Mono)
{
    double newE = 0,oldE = 0;
    for(int i = 0; i < neighborLen[Mono]; i++)
    {
       if(C[0].M[Mono].bondedWith(0,neighborList[i+MAXNEIGB*Mono]) == false)
	   {
	   double neighbLen = nearestImageSq( C[0].M[neighborList[i+MAXNEIGB*Mono]], newMon) ;
	if(debug) printf("%i mon with %i neighbors with new sqlength %f old sqlength %f \n",Mono,C[Ch].M[Mono].NeighborList[i],nearestImageSq( C[C[Ch].M[Mono].NeighborChainList[i]].M[C[Ch].M[Mono].NeighborList[i]], newMon),nearestImageSq( C[C[Ch].M[Mono].NeighborChainList[i]].M[C[Ch].M[Mono].NeighborList[i]], C[Ch].M[Mono]));
        if(neighbLen <  PARTICLE_SIZE )
        {
           if(debug) printf("not possible step\n");
            return NOT_POSSIBLE;
        }
			//if(neighborLen[Mono] > 0) printf("true");

           if(debug) printf("possible step\n");



          if(nearestImageSq( C[0].M[neighborList[i+MAXNEIGB*Mono]], C[Ch].M[Mono]) <  POT_CUT_SQ){


        if(UPOROT) oldE = oldE +  EPS[C[0].M[Mono].Typ][C[0].M[neighborList[i+MAXNEIGB*Mono]].Typ]*64.0/(16.0+2.0*abs(Mono-neighborList[i+MAXNEIGB*Mono]));

        oldE = oldE +EPS[C[Ch].M[Mono].Typ][C[0].M[neighborList[i+MAXNEIGB*Mono]].Typ]; }
        if(neighbLen <  POT_CUT_SQ){

        if(UPOROT) newE = newE +  EPS[C[0].M[Mono].Typ][C[0].M[neighborList[i+MAXNEIGB*Mono]].Typ]*64.0/(16.0+2.0*abs(Mono-neighborList[i+MAXNEIGB*Mono]));


        newE = newE +EPS[newMon.Typ][C[0].M[neighborList[i+MAXNEIGB*Mono]].Typ]; }

	}
    }



return newE-oldE;
}

double EnergyNV()
{
double ENV = 0;
for(int Ch = 0; Ch < Nchains; Ch ++)
{
    for(int Mono = 0; Mono < C[Ch].N; Mono++)
    {

        for(int i = 0; i < neighborLen[Mono]; i++)
    {

          if(nearestImageSq( C[0].M[neighborList[i+MAXNEIGB*Mono]] ,C[Ch].M[Mono]) <  POT_CUT_SQ && C[0].M[Mono].bondedWith(0,neighborList[i+MAXNEIGB*Mono]) == false ){
        ENV = ENV +  EPS[C[Ch].M[Mono].Typ][C[0].M[neighborList[i+MAXNEIGB*Mono]].Typ];

             if(UPOROT)   ENV = ENV +  EPS[C[Ch].M[Mono].Typ][C[0].M[neighborList[i+MAXNEIGB*Mono]].Typ]*64.0/(16.0+2.0*abs(Mono-neighborList[i+MAXNEIGB*Mono]));


        }


    }



    }



}

return ENV/2;
}


double DEnergyStiff(const Mon& newMon,int Ch, int Mono)
{
double oldE=0,newE = 0,dE;
double COS,COS2;

Mon MonOld = C[Ch].M[Mono];

if(newMon.NumBonds > 0)
{
    for(int i = 0; i < newMon.NumBonds; i++)
    {

        for(int j = i+1; j < newMon.NumBonds; j++)
        {

            COS = (newMon.XReal - C[newMon.BondChains[i]].M[newMon.Bonds[i]].XReal)*(newMon.XReal - C[newMon.BondChains[j]].M[newMon.Bonds[j]].XReal)/((newMon.XReal - C[newMon.BondChains[i]].M[newMon.Bonds[i]].XReal).len()*(newMon.XReal - C[newMon.BondChains[j]].M[newMon.Bonds[j]].XReal).len());


            COS2 = (MonOld.XReal - C[MonOld.BondChains[i]].M[MonOld.Bonds[i]].XReal)*(MonOld.XReal - C[MonOld.BondChains[j]].M[MonOld.Bonds[j]].XReal)/((MonOld.XReal - C[MonOld.BondChains[i]].M[MonOld.Bonds[i]].XReal).len()*(MonOld.XReal - C[MonOld.BondChains[j]].M[MonOld.Bonds[j]].XReal).len());
       //     COS = -COS; COS2 = -COS2;
            oldE += EPSST[C[MonOld.BondChains[i]].M[MonOld.Bonds[i]].Typ][MonOld.Typ][C[MonOld.BondChains[j]].M[MonOld.Bonds[j]].Typ] * ECos(COS2);
            newE += EPSST[C[MonOld.BondChains[i]].M[MonOld.Bonds[i]].Typ][newMon.Typ][C[newMon.BondChains[j]].M[newMon.Bonds[j]].Typ] * ECos(COS);
        //    printf("Stiffness in middle  cos %f cosold %f\n", COS,COS2);
        }

        if(C[newMon.BondChains[i]].M[newMon.Bonds[i]].NumBonds > 1)
        {
            Mon tmp = C[newMon.BondChains[i]].M[newMon.Bonds[i]];

        for(int j = 0; j < tmp.NumBonds ; j++)
        {
                if(Ch != tmp.BondChains[j] || Mono != tmp.Bonds[j])
                {

                    Vector bondnew = newMon.XReal - tmp.XReal;
                    Vector bondold = C[Ch].M[Mono].XReal - tmp.XReal;
                    Vector bond = tmp.XReal - C[tmp.BondChains[j]].M[tmp.Bonds[j]].XReal;
                    COS = -(bondold*bond)/(bondold.len()*bond.len());
                    COS2 = -(bondnew*bond)/(bondnew.len()*bond.len());

                   // printf("Stiffness in end  cos %f cosold %f\n", COS,COS2);
                    oldE += EPSST[MonOld.Typ][tmp.Typ][C[MonOld.BondChains[i]].M[MonOld.Bonds[i]].Typ] * ECos(COS);
                    newE += EPSST[newMon.Typ][tmp.Typ][C[newMon.BondChains[i]].M[newMon.Bonds[i]].Typ] * ECos(COS2);


                }


        }

        }
    }
}
return newE-oldE;
}


double EnergyStiff()
{
    double COS;
    double ESTF = 0;
    for(int i = 0; i < Nchains; i++)
    {
        for(int j = 0; j < C[i].N; j++)
        {
            if(C[i].M[j].NumBonds > 1)
            {
                for(int k = 0; k < C[i].M[j].NumBonds; k++)
                {
                    for(int l = k+1; l < C[i].M[j].NumBonds; l++)
                    {
                         COS = (C[i].M[j].XReal - C[C[i].M[j].BondChains[k]].M[C[i].M[j].Bonds[k]].XReal)*(C[i].M[j].XReal - C[C[i].M[j].BondChains[l]].M[C[i].M[j].Bonds[l]].XReal)/((C[i].M[j].XReal - C[C[i].M[j].BondChains[k]].M[C[i].M[j].Bonds[k]].XReal).len()*(C[i].M[j].XReal - C[C[i].M[j].BondChains[l]].M[C[i].M[j].Bonds[l]].XReal).len());

                         ESTF+= EPSST[C[C[i].M[j].BondChains[k]].M[C[i].M[j].Bonds[k]].Typ][C[i].M[j].Typ][C[C[i].M[j].BondChains[l]].M[C[i].M[j].Bonds[l]].Typ] * ECos(COS);
                    //    printf ("bad cos %f destf %f\n",COS,ESTF);
                    }
                }


            }




        }
    }




//printf("%f ESTF\n");
//getch();
return ESTF;

}

inline double ECos(double Cos)
{
  //  if(Cos > -0.9  ) return 1;
  //  else return 0;
    return (1+Cos)*0.5;
}

// CUDA

__global__ void buildList(float* coor, float* list , float *Size, float *ObrSize)
{


  
  
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  float r[3];
  float d = 0;
  for(int i = 0; i < 3; i++)
  {
  r[i] = coor[3*x+i] - coor[3*y+i];
  r[i] = r[i] - Size[i]*rintf(r[i]*ObrSize[i]);
  d = d + r[i]*r[i];
 // printf("%i %i %f %rdist\n",x,y,d);
  }

  list[x*NMax+y] = d;
 
 
 }

 __global__ void buildNeigb(float *list, int *reallist,int *realcount,float NEIGHBOR_CUT,int NMax)
 {
	 int x = blockIdx.x*blockDim.x + threadIdx.x;
	 int count = 0;
	 for(int i=0; i < NMax; i++)
	 {
		// printf("%f list\n",list[x*NMax+i]);
		 if(list[x*NMax+i] < NEIGBCUT && i!=x)
		 {
			 
			 reallist[x*MAXNEIGB+count] = i;
			 count++;
		 }

	 }
	 realcount[x] = count;

 }





void CudaNeighborTable()
{
	dim3 gridSize = dim3(NMax/16, NMax/16, 1);
	dim3 blockSize = dim3(16, 16, 1);
	
	gpuErrchk(cudaMemcpy(cudaCoords,coords,NMax*3*sizeof(float),cudaMemcpyHostToDevice));
	//cudaThreadSynchronize();
	buildList <<<gridSize,blockSize>>> (cudaCoords,cudaList,cudaSize,cudaObrSize);
	
	cudaThreadSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	buildNeigb <<<NMax/128,128>>> (cudaList,cudaneighborList,cudaneighborLen,NEIGHBOR_CUTSQ,NMax);
	//cudaThreadSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaMemcpy(neighborList,cudaneighborList, NMax*MAXNEIGB*sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(neighborLen,cudaneighborLen, NMax*sizeof(int), cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaMemcpy(testlist,cudaList, NMax*NMax*sizeof(float), cudaMemcpyDeviceToHost));
   //cudaMalloc((void**)&cudaCoords, NMax*3*sizeof(float));
  //  cudaMalloc((void**)&cudaList, NMax*NMax*sizeof(float));
   // cudaMalloc((void**)&cudaneighborLen, NMax*sizeof(int));
//	cudaMalloc((void**)&cudaneighborList, NMax*MAXNEIGB*sizeof(int));
}





inline void BuildNeighborTable(double Cut)
{
    int i,k,l,j;
    double d;
    int tmp = -991991;

	for(i = 0; i < Nchains; i++)
    {
        for( j = 0; j < C[i].N ; j++)
        {
            C[i].M[j].Neighbors = 0;
		}
	}
 //   #pragma omp parallel for
    for(i = 0; i < Nchains; i++)
    {

        for( j = 0; j < C[i].N ; j++)
        {

					k = i;

					for(l = j+1; l < C[k].N ; l++)
					{
                        if( (i != k || l != j) && C[i].M[j].bondedWith(k,l) == false)
                        {

                        d = nearestImageSq(C[i].M[j],C[k].M[l]);
                        if(d < NEIGHBOR_CUTSQ)
                        {
                        C[i].M[j].NeighborChainList[C[i].M[j].Neighbors] = k;
                        C[i].M[j].NeighborList[C[i].M[j].Neighbors] = l;
                        C[k].M[l].NeighborChainList[C[k].M[l].Neighbors] = i;
						C[k].M[l].NeighborList[C[k].M[l].Neighbors] = j;
						C[i].M[j].Neighbors++;
						C[k].M[l].Neighbors++;
						if(j == ntestmons)
                            {
                                printf ("debugging %i %i monomers sqlen = %f\n",j,l,d);
                            }
                        }
                        if(d < PARTICLE_SIZE)
                        {
                            debug = true;
                            printf("Merde between %i and %i distance %f !\n",j,l,nearestImageR(C[i].M[j],C[k].M[l]));
                            tmp = j;
                            //printf("%s",dump);
                            StoreConf();
                            getchar();
                        }
                        }

                    }

					for(k = i+1; k < Nchains; k++)
					{

						for(l = 0; l < C[k].N ; l++)
						{
                        if( (i != k || l != j) && C[i].M[j].bondedWith(k,l) == false)
                        {

                        d = nearestImageSq(C[i].M[j],C[k].M[l]);
                        if(d < NEIGHBOR_CUTSQ)
                        {
                        C[i].M[j].NeighborChainList[C[i].M[j].Neighbors] = k;
                        C[i].M[j].NeighborList[C[i].M[j].Neighbors] = l;
                        C[k].M[l].NeighborChainList[C[k].M[l].Neighbors] = i;
						C[k].M[l].NeighborList[C[k].M[l].Neighbors] = j;
						C[i].M[j].Neighbors++;
						C[k].M[l].Neighbors++;
                            if(j == ntestmons)
                            {
                                printf ("debugging %i %i monomers sqlen = %f\n",j,l,d);
                            }
                        }
                        if(d < PARTICLE_SIZE)
                        {
                            debug = true;
                            printf("Merde between %i and %i distance %f !\n",j,l,nearestImageR(C[i].M[j],C[k].M[l]));
                            tmp = j;
                            //printf("%s",dump);
                            StoreConf();
                            getchar();
                        }
                        }

						}



            }


       // if(C[i].M[j].Neighbors > 50)
       // {
         //   printf("too many neighbors %i\n",C[i].M[j].Neighbors);
       // }

        }



    }








}

bool AcceptMetropolis(double DE1, double DE2)
{

    double prob = (double)rand()/RAND_MAX;
    if(prob < exp(-(DE1+DE2)))
    {
    return true;
    }

else return false;

}


double GyrationRadius(int Chain, int Start, int End);



int main()
{
double etest = 0;

Nchains = 0;
FILE *fmat = fopen("Matr.txt","w");
FILE *fstat = fopen("stat.txt","w");
setvbuf(stdout,NULL,_IONBF,0);
setvbuf(fstat,NULL,_IONBF,0);
char fnDend[40];
sprintf(fnDend,"position_machine.txt");
//InitStats(1024, 1, 3, 300,512);

//omp_set_dynamic(0);      // запретить библиотеке openmp менять число потоков во время исполнения
 // omp_set_num_threads(10); // установить число потоков в 10

simpleaccrate = 0;
CBAccrate = 0;
//outputStats(fstat);
firstTable = true;
srand(time(NULL));
//OBR_SIZE[0] = 1.0/SIZE[0];
//OBR_SIZE[1] = 1.0/SIZE[1];
//OBR_SIZE[2] = 1.0/SIZE[2];
for(int j = 0; j < MONTYPES; j++)
{

    for(int f = 0; f < MONTYPES; f++)
    {
        EPS[j][f] = 0;
        for(int ms = 0; ms < MONTYPES; ms++)
        {
            EPSST[j][f][ms] = 0;
        }
    }
}
//stiffness parameters, weren't used for some time, maybe broken.
//EPSST[0][0][0] = 8;
//energy parameters
EPS[0][1] = 0;
EPS[0][0] = 0;
EPS[1][1] = -0.5;
EPS[1][0] = 0;
/*
C[0].N = 3;

Vector vec(3,3,3);
Vector sh(1,0,0);e
Vector sh2(2,0,0);
C[0].M[0].X = Vector(3,3,3);
C[0].M[1].X = Vector(4,3,3);
C[0].M[2].X = Vector(5,3,3);

for(int j = 0; j < 3;j++)
{
C[0].M[j].Typ = 0;

 C[0].M[j].NumBonds = 0;
C[0].M[j].Neighbors = 0;
C[0].M[j].XReal = C[0].M[j].X;

}
C[0].M[0].bondWith(0,1);

C[0].M[1].bondWith(0,0);

C[0].M[1].bondWith(0,2);

C[0].M[2].bondWith(0,1);
*/
//initial conf generation

//void GenerateAmphiphilic(int len, int Typ1, int Typ2, bool RandomGrafting = false, int LenGrafts = 1, int GraftingInterval = 1);
//void GenerateGraft(int len, int ch, int startMon, int Typ1);
//void GenerateDiblock(int len1, int len2, int Typ1, int Typ2);
//parseCoordsFromDendrimer(fnDend, 202);
//ReadConf();
GenerateGradient(512,0,1,2,0.5);
//C[0].M[FIXED_POINT].Typ = 1;
/*
for(int sfa = 64;sfa < 63+ 60; sfa++)
{
if(sfa%4!=0)
{
GenerateGraft(1,0,sfa,1);
}
}
*/
//SAMC Samc(-1000, 1000, 0,2000, 1, 1000, EnergyNV(), EnergyStiff());
//GenerateHomopolymer(10,0);


//float coords[3*Nmax*NMaxChains];
//int neighborList[Nmax*MAXNEIGB];
//int neighborLen[Nmax];
//cudaCoords = new float[3*Nmax];
//cudaneighborList = new int[Nmax*MAXNEIGB];
//cudaneighborLen = new int[Nmax];

// dim3 gridSize = dim3(size/8, size/8, 1);
// dim3 blockSize = dim3(8, 8, 1);
//const float *cudaSize;
//const float *cudaObrSize;
printf("before memory allocation");
   gpuErrchk( cudaMalloc((void**)&cudaCoords, NMax*3*sizeof(float)));
   gpuErrchk(  cudaMalloc((void**)&cudaList, NMax*NMax*sizeof(float)));
   gpuErrchk(  cudaMalloc((void**)&cudaneighborLen, NMax*sizeof(int)));
	gpuErrchk( cudaMalloc((void**)&cudaneighborList, NMax*MAXNEIGB*sizeof(int)));
 gpuErrchk(  cudaMalloc((void**)&cudaSize, 3*sizeof(float)));
 gpuErrchk(  cudaMalloc((void**)&cudaObrSize, 3*sizeof(float)));
gpuErrchk(  cudaMemcpy(cudaSize,SIZE,3*sizeof(float),cudaMemcpyHostToDevice));
 gpuErrchk( cudaMemcpy(cudaObrSize,OBR_SIZE,3*sizeof(float),cudaMemcpyHostToDevice));
cudaDeviceSynchronize();
CudaNeighborTable();
printf("after memory allocation");
//BuildNeighborTable(NEIGHBOR_CUT);
//Samc.init(-7000, 4000, -1, 1, 1, 100000, EnergyNV(), 0, -6000,100);

if(CheckAllBonds() == false)
{
    printf("Bad generation!\n");
}
outputVrml("start.vrml");
//getchar();

//main cycla

long int i;
long int ggg = 4;
for(i = 0; i < STEPS; i ++)
{
//printf("into main cycle");

/*if(i%1000==0)
{
 //  ReverseChain(0);
   // ConfigurationalBias(0,0, AcceptMetropolis,rand()%20+2,false);
}
*/
//*dump='\0';
//printf("distmatr\n");
/*
for(int a = 0; a < Nmax; a++)
{
	for( int b = 0; b < neighborLen[a]; b++)
	{
		
		printf("%i ",neighborList[a*MAXNEIGB+b]);
	}
	printf("\n");
	
}
*/
//printf("before first table");
CudaNeighborTable();
//printf("after first table");
for(int k = 0; k < C[0].N; k++ )
{
	SimpleStep(AcceptMetropolis);

	SimpleStep(AcceptMetropolis);

	SimpleStep(AcceptMetropolis);

	SimpleStep(AcceptMetropolis);


}

/*
if(i%5000==0)
{


for(int p = 0; p < C[0].N; p++)
{
    for(int q = 0; q < C[0].N; q++)
    {
        if(p!=q && C[0].M[p].bondedWith(0,q)==false && nearestImageR(C[0].M[p],C[0].M[q]) < POT_CUT && C[0].M[p].neighborWith(0,q)==false)
        {
            printf("error\n");
            getchar();
        }
    }
}
}
*/
/*for(int j = 0; j < Nchains; j++)
{
    if(C[j].N > 9)
    {
   //    TargetedStep(AcceptMetropolis, j , 0, C[j].N/3);
     //   TargetedStep(AcceptMetropolis, j , C[j].N/3 , 2*C[j].N/3);
      // TargetedStep(AcceptMetropolis, j , 2*C[j].N/3 , C[j].N);
    }
    else
    {
    //    TargetedStep(AcceptMetropolis, j, 0, C[j].N);
    }

  //  for(int k = 0; k < C[j].N ; k++)
   // {
 //      TargetedStep(AcceptSAMC, j, k,k+1);
   // }
}

*/
if(i%5000 == 0){
    //printf ("step %i  Energynv: %f  Energystiff: %f checkallbonds %i\n" ,i, EnergyNV(),EnergyStiff(),CheckAllBonds());
//	for(int h = 0; h < 64; h++)
	//{
	//	printf("%i %f %f %f\n",h,coords[h*3], coords[h*3+1], coords[h*3+2]);
		
	//}

    if( EnergyNV() < etest || i%200000 == 0)
    {
     char f[50];
    sprintf(f,"conf_%ld_%f_energy.vrml",i,EnergyNV());

    char ppp[50];
    fprintf(fmat,"Step %ld\n",i);
    PrintDistMatrix(0, fmat);
    sprintf(ppp,"confreal_%ld.vrml",i);
    outputPeriodicVrml(ppp);
    outputVrml(f);
    etest = EnergyNV();
    StoreConf();
    }
}
   if(i%5000 == 0 && EnergyNV() < -2000)
    {
       // CalculateStats();
    }
if(i%20000==0)
{



  Samc.printGE("GE.txt","w");
 printf ("step %ld  Energynv %f  Eps %f Rg %f Rendtoend %f checkallbonds %i simplerate %i cbrate %i\n" ,i, EnergyNV(),EPS[0][0],GyrationRadius(0,0,C[0].N-1),(C[0].M[0].XReal-C[0].M[C[0].N-1].XReal).sqlen(),CheckAllBonds(),simpleaccrate, CBAccrate);
}

if(i>10000 && i%20000==0)
{
   //outputStats(fstat);
}







//SimpleStep(AcceptMetropolis);


}

//end main cycle


return 0;
}






void outputGradVrml(char* filename)
{
    #ifndef _WIN32
char str[1000] = "cp vrml_header.txt ";
strcat(str,filename);
system(str);
#endif

#ifdef _WIN32
char str[1000] = "copy vrml_header.txt ";
strcat(str,filename);
system(str);
#endif

//system("y");
FILE *F = fopen(filename,"a");
//printf("%lld",mAcc);
//getch();
for(int i = 0; i < Nchains; i++)
{
for(int j = 0; j < C[i].N; j ++)
{
if(C[i].M[j].Typ == 1)
{

    fprintf(F,"Transform { \n translation %f %f %f  children [ USE Ball1]}\n",C[i].M[j].XReal.x[0],C[i].M[j].XReal.x[1],C[i].M[j].XReal.x[2]);
}
else
{
 hsv color;
 color.h = ((double)j/(double)C[i].N)*360.0;
 color.s = 0.9;
 color.v = 0.8;
 rgb clrrgb = hsv2rgb(color);
 
 
 fprintf(F,"Transform { \n translation %f %f %f  children [ Shape {geometry Sphere {radius 0.5} \n appearance Appearance {material Material {diffuseColor %lf %lf %lf}}}]}\n",C[i].M[j].XReal.x[0],C[i].M[j].XReal.x[1],C[i].M[j].XReal.x[2], clrrgb.r , clrrgb.g, clrrgb.b );
}


}
}

for(int y = 0; y < Nchains; y++)
{
fprintf(F,"Shape {appearance Appearance {material Material {emissiveColor 0 0 0}}geometry IndexedLineSet {coord Coordinate {point [");
for(int a = 0; a < C[y].N;a++)
{
   /*if(C[y].M[a].rigid == true)
    {
    printf("1\n");
    }
    else
    {
    printf("0\n");
    }
    getch();*/
    fprintf(F,"%f %f %f , \n",C[y].M[a].XReal.x[0],C[y].M[a].XReal.x[1],C[y].M[a].XReal.x[2]);
}
fprintf(F,"]}coordIndex [");
for(int a = 0; a< C[y].N; a++)
{
for(int yy = 0; yy < C[y].M[a].NumBonds ; yy++)
{


fprintf(F,"%i,%i,-1\n",a,C[y].M[a].Bonds[yy]);
}
}
fprintf(F,"]}}");
}


fprintf(F,"Shape {appearance Appearance {material Material {emissiveColor 0 0 0}}geometry IndexedLineSet {coord Coordinate {point [");
fprintf(F,"0 0 %f\n",SIZE[2]);
fprintf(F,"0 %f %f \n",SIZE[1],SIZE[2]);
fprintf(F,"0 %f 0\n",SIZE[1]);
fprintf(F,"0 0 0\n");
fprintf(F,"%f 0 0\n",SIZE[0]);
fprintf(F,"%f 0 %f\n",SIZE[0],SIZE[2]);
fprintf(F,"%f %f %f\n",SIZE[0],SIZE[1],SIZE[2]);
fprintf(F,"%f %f 0\n",SIZE[0],SIZE[1]);
fprintf(F,"]}coordIndex [");


fprintf(F,"0,1,-1\n");
fprintf(F,"1,2,-1\n");
fprintf(F,"2,3,-1\n");
fprintf(F,"0,3,-1\n");

fprintf(F,"3,4,-1\n");
fprintf(F,"1,6,-1\n");
fprintf(F,"2,7,-1\n");
fprintf(F,"0,5,-1\n");

fprintf(F,"4,5,-1\n");
fprintf(F,"5,6,-1\n");
fprintf(F,"6,7,-1\n");
fprintf(F,"7,4,-1\n");



fprintf(F,"]}}");



fclose(F);





}




void outputPeriodicVrml(char *filename)
{
//printf("in vrml output");
//getch();

#ifndef _WIN32
char str[1000] = "cp vrml_header.txt ";
strcat(str,filename);
system(str);
#endif

#ifdef _WIN32
char str[1000] = "copy vrml_header.txt ";
strcat(str,filename);
system(str);
#endif

//system("y");
FILE *F = fopen(filename,"a");
//printf("%lld",mAcc);
//getch();
for(int i = 0; i < Nchains; i++)
{
for(int j = 0; j < C[i].N; j ++)
{
if(C[i].M[j].Typ == 1)
{

    fprintf(F,"Transform { \n translation %f %f %f  children [ USE Ball1]}\n",C[i].M[j].XReal.getReal().x[0],C[i].M[j].XReal.getReal().x[1],C[i].M[j].XReal.getReal().x[2]);
}
else
{
 fprintf(F,"Transform { \n translation %f %f %f  children [ USE Ball2]}\n",C[i].M[j].XReal.getReal().x[0],C[i].M[j].XReal.getReal().x[1],C[i].M[j].XReal.getReal().x[2]);
}


}
}

for(int y = 0; y < Nchains; y++)
{
fprintf(F,"Shape {appearance Appearance {material Material {emissiveColor 0 0 0}}geometry IndexedLineSet {coord Coordinate {point [");
for(int a = 0; a < C[y].N;a++)
{
   /*if(C[y].M[a].rigid == true)
    {
    printf("1\n");
    }
    else
    {
    printf("0\n");
    }
    getch();*/
    fprintf(F,"%f %f %f , \n",C[y].M[a].XReal.getReal().x[0],C[y].M[a].XReal.getReal().x[1],C[y].M[a].XReal.getReal().x[2]);
}
fprintf(F,"]}coordIndex [");
for(int a = 0; a< C[y].N; a++)
{
for(int yy = 0; yy < C[y].M[a].NumBonds ; yy++)
{


fprintf(F,"%i,%i,-1\n",a,C[y].M[a].Bonds[yy]);
}
}
fprintf(F,"]}}");
}








fprintf(F,"Shape {appearance Appearance {material Material {emissiveColor 0 0 0}}geometry IndexedLineSet {coord Coordinate {point [");
fprintf(F,"0 0 %f\n",SIZE[2]);
fprintf(F,"0 %f %f \n",SIZE[1],SIZE[2]);
fprintf(F,"0 %f 0\n",SIZE[1]);
fprintf(F,"0 0 0\n");
fprintf(F,"%f 0 0\n",SIZE[0]);
fprintf(F,"%f 0 %f\n",SIZE[0],SIZE[2]);
fprintf(F,"%f %f %f\n",SIZE[0],SIZE[1],SIZE[2]);
fprintf(F,"%f %f 0\n",SIZE[0],SIZE[1]);
fprintf(F,"]}coordIndex [");


fprintf(F,"0,1,-1\n");
fprintf(F,"1,2,-1\n");
fprintf(F,"2,3,-1\n");
fprintf(F,"0,3,-1\n");

fprintf(F,"3,4,-1\n");
fprintf(F,"1,6,-1\n");
fprintf(F,"2,7,-1\n");
fprintf(F,"0,5,-1\n");

fprintf(F,"4,5,-1\n");
fprintf(F,"5,6,-1\n");
fprintf(F,"6,7,-1\n");
fprintf(F,"7,4,-1\n");



fprintf(F,"]}}");














fclose(F);




}



void outputVrml(char* filename)
{
    #ifndef _WIN32
char str[1000] = "cp vrml_header.txt ";
strcat(str,filename);
system(str);
#endif

#ifdef _WIN32
char str[1000] = "copy vrml_header.txt ";
strcat(str,filename);
system(str);
#endif

//system("y");
FILE *F = fopen(filename,"a");
//printf("%lld",mAcc);
//getch();
for(int i = 0; i < Nchains; i++)
{
for(int j = 0; j < C[i].N; j ++)
{
if(C[i].M[j].Typ == 1)
{

    fprintf(F,"Transform { \n translation %f %f %f  children [ USE Ball1]}\n",C[i].M[j].XReal.x[0],C[i].M[j].XReal.x[1],C[i].M[j].XReal.x[2]);
}
else if(C[i].M[j].Typ ==2)
{
 fprintf(F,"Transform { \n translation %f %f %f  children [ Shape {geometry Sphere {radius 0.5} \n appearance Appearance {material Material {diffuseColor %lf %lf %lf}}}]}\n",C[i].M[j].XReal.x[0],C[i].M[j].XReal.x[1],C[i].M[j].XReal.x[2], 0, 0, 1 );
	
	
}
else
{
 fprintf(F,"Transform { \n translation %f %f %f  children [ USE Ball2]}\n",C[i].M[j].XReal.x[0],C[i].M[j].XReal.x[1],C[i].M[j].XReal.x[2]);
}


}
}

for(int y = 0; y < Nchains; y++)
{
fprintf(F,"Shape {appearance Appearance {material Material {emissiveColor 0 0 0}}geometry IndexedLineSet {coord Coordinate {point [");
for(int a = 0; a < C[y].N;a++)
{
   /*if(C[y].M[a].rigid == true)
    {
    printf("1\n");
    }
    else
    {
    printf("0\n");
    }
    getch();*/
    fprintf(F,"%f %f %f , \n",C[y].M[a].XReal.x[0],C[y].M[a].XReal.x[1],C[y].M[a].XReal.x[2]);
}
fprintf(F,"]}coordIndex [");
for(int a = 0; a< C[y].N; a++)
{
for(int yy = 0; yy < C[y].M[a].NumBonds ; yy++)
{


fprintf(F,"%i,%i,-1\n",a,C[y].M[a].Bonds[yy]);
}
}
fprintf(F,"]}}");
}


fprintf(F,"Shape {appearance Appearance {material Material {emissiveColor 0 0 0}}geometry IndexedLineSet {coord Coordinate {point [");
fprintf(F,"0 0 %f\n",SIZE[2]);
fprintf(F,"0 %f %f \n",SIZE[1],SIZE[2]);
fprintf(F,"0 %f 0\n",SIZE[1]);
fprintf(F,"0 0 0\n");
fprintf(F,"%f 0 0\n",SIZE[0]);
fprintf(F,"%f 0 %f\n",SIZE[0],SIZE[2]);
fprintf(F,"%f %f %f\n",SIZE[0],SIZE[1],SIZE[2]);
fprintf(F,"%f %f 0\n",SIZE[0],SIZE[1]);
fprintf(F,"]}coordIndex [");


fprintf(F,"0,1,-1\n");
fprintf(F,"1,2,-1\n");
fprintf(F,"2,3,-1\n");
fprintf(F,"0,3,-1\n");

fprintf(F,"3,4,-1\n");
fprintf(F,"1,6,-1\n");
fprintf(F,"2,7,-1\n");
fprintf(F,"0,5,-1\n");

fprintf(F,"4,5,-1\n");
fprintf(F,"5,6,-1\n");
fprintf(F,"6,7,-1\n");
fprintf(F,"7,4,-1\n");



fprintf(F,"]}}");



fclose(F);





}

/////GENERATORS


void GenerateHomopolymer(int len, int Typ1)
{

    int timestry = 0;
    Vector rndVect;
    Mon testMon;

    bool retry;
    // finding first spot
    do
    {

    retry = false;

    for(int i = 0; i < 3 ; i++)
    {
    rndVect.x[i] = (double)(rand()%(int)SIZE[i]);
    }
	
    testMon.X = rndVect;
    testMon.XReal = rndVect;
        for(int i = 0; i < Nchains; i++)
        {
            for(int j = 0; j < C[i].N ; j++ )
            {
                if((C[i].M[j]-testMon).len() < PARTICLE_SIZE)
                {
                    retry = true;
                }
            }
        }

    }while(retry == true);
    testMon.Typ = Typ1;
    
    testMon.NumBonds = 0;
    testMon.Neighbors = 0;
	testMon.coor = NParticles;
	NParticles++;
	testMon.update();
    C[Nchains].M[0] = testMon;
    
	int num;
    Vector oldshift;
    for(num = 1; num < len; num++)
    {
        // building homopolymer chain Typ1
         timestry = 0;
        Vector shift;
         do{
         timestry++;
         if(timestry > 2000)
         {
          //  GenerateHomopolymer(len,Typ1);
          printf("too long homopolymer generation chain %i mon %i\n",Nchains,num);
          // getch();
            // return;
         }
         retry = false;
         for(int i = 0; i < 3 ; i++)
        {
        shift.x[i] = 1.0 - 2.0*(double)rand()/RAND_MAX;
        }
        if(shift.len() < BONDMIN || shift.len() > BONDMAX)
         { retry = true;
       //  printf("problem 4");
        }

	if(num > 1)
	{
		if( (oldshift*shift) < 0.6)
		{
		retry = true;
		}
	}
         for(int i = 0; i < Nchains; i++)
         {
            for(int j = 0; j < C[i].N ; j++ )
            {
                if(((C[Nchains].M[num-1]+shift)-C[i].M[j]).len() < PARTICLE_SIZE)
                {
              //      printf("problem 1");
                    retry = true;
                }
            }
        }


            for(int j=0; j < num; j++)
            {
                if((C[Nchains].M[j] - (C[Nchains].M[num-1]+shift)).sqlen() < PARTICLE_SIZE && (num-j) > 1) {
                retry = true;
               // printf("problem 2 %f %f ",shift.len(),(C[Nchains].M[j] - (C[Nchains].M[num-1]+shift)).len());
                }
            }
            for(int j=0; j < 3; j++)
            {


            if((C[Nchains].M[num-1]+shift).XReal.x[j] < 0 || ((C[Nchains].M[num-1]+shift).XReal.x[j] > SIZE[j])) {retry = true;
           // printf("problem 3");
            }
            }
        } while( retry == true);
        oldshift = shift;
		
	 C[Nchains].M[num] = C[Nchains].M[num-1] + shift;
        C[Nchains].M[num].NumBonds = 0;
        C[Nchains].M[num].Neighbors = 0;
		C[Nchains].M[num].coor = NParticles;
		NParticles++;
	    C[Nchains].M[num].update();
        C[Nchains].M[num].bondWith(Nchains,num-1);
        C[Nchains].M[num-1].bondWith(Nchains,num);

    }
    C[Nchains].N = len;
    Nchains++;


}



void GenerateGraft(int len, int ch, int startMon, int Typ1)
{
    int timestry = 0;
    Mon start = C[ch].M[startMon];
    Mon testMon;
    int num;
    bool retry;
     for(num = C[ch].N; num < len+C[ch].N; num++)
    {
        // building homopolymer chain Typ1
        Vector shift;

        timestry = 0;
         do{
         timestry++;
         if(timestry > 2000)
         {
        printf("too long graft generation\n");
         //  GenerateGraft(len,ch,startMon,Typ1);
           //  return;
         }
         retry = false;
         for(int i = 0; i < 3 ; i++)
        {
        shift.x[i] = 1.0 - 2.0*(double)rand()/RAND_MAX;
        }
        testMon = start+shift;
        testMon.NumBonds = 0;
        testMon.Neighbors = 0;
        testMon.Moved = false;
        if(num == C[ch].N) {testMon.bondWith(ch,startMon); }
        else { testMon.bondWith(ch,num-1);}
        testMon.Typ = Typ1;
        if(shift.len() < BONDMIN || shift.len() > BONDMAX) {retry = true; }
         for(int i = 0; i < Nchains; i++)
         {
            for(int j = 0; j < C[i].N ; j++ )
            {
                if((testMon-C[i].M[j]).len() < PARTICLE_SIZE)
                {
                    retry = true;
                }
            }
        }


            for(int j=0; j < num; j++)
            {
                if((C[ch].M[j] - testMon).len() < PARTICLE_SIZE && testMon.bondedWith(ch,j)==false) { retry = true; }
            }
            for(int j=0; j < 3; j++)
            {


            if((start+shift).XReal.x[j] < 0 || ((start+shift).XReal.x[j] > SIZE[j])) {retry = true;}
            }

        } while( retry == true);
        C[ch].M[num] = testMon;
        C[testMon.BondChains[0]].M[testMon.Bonds[0]].bondWith(ch,num);
		C[ch].M[num].coor = NParticles;
		NParticles++;
	    C[ch].M[num].update();
        start = testMon;

    }
    C[ch].N = len+C[ch].N;


}


void GenerateAmphiphilic(int len, int Typ1, int Typ2, bool RandomGrafting , int LenGrafts , int GraftingInterval )
{




GenerateHomopolymer(len,Typ1);
for(int i = 0; i < len; i++)
{
    if(RandomGrafting == true)
    {
        double rnd = rand()/RAND_MAX;
        if(rnd <  1.0/(double)GraftingInterval)
        {

            GenerateGraft(LenGrafts,Nchains-1,i,Typ2);
        }
    }
    else
    {
        if(i%GraftingInterval == 0)
        {
            GenerateGraft(LenGrafts,Nchains-1,i,Typ2);
        }
    }
}


}

void GenerateDiblock(int len1,int len2, int Typ1, int Typ2)
{
    GenerateHomopolymer(len1,Typ1);
    GenerateGraft(len2,Nchains-1,len1-1,Typ2);
}

// END GENERATORS

bool CheckAllBonds()
{
    bool chk = true;
    for(int i = 0; i < Nchains; ++i)
    {
        for(int j = 0; j < C[i].N; ++j)
        {
            for(int k = 0; k < C[i].M[j].NumBonds; ++k)
            {

                  if(C[C[i].M[j].BondChains[k]].M[C[i].M[j].Bonds[k]].bondedWith(i,j) == false)
                {
                    printf("Not bonded properly!\n");
                    chk =  false;
                }
                if((C[i].M[j].XReal - C[C[i].M[j].BondChains[k]].M[C[i].M[j].Bonds[k]].XReal).len() > BONDMAX || (C[i].M[j].XReal - C[C[i].M[j].BondChains[k]].M[C[i].M[j].Bonds[k]].XReal).len()  < BONDMIN)
                {
                    printf("Bond c %i  m %i, c2 %i m2 %i BAD length %f\n",i,j,C[i].M[j].BondChains[k],C[i].M[j].Bonds[k],nearestImageR(C[i].M[j], C[C[i].M[j].BondChains[k]].M[C[i].M[j].Bonds[k]]));
                    StoreConf();
                    //getch();
                    chk = false;
                }

            }
        }
    }

    return chk;
}





void PrintDistMatrix(int ChNum, FILE* f)
{
//for homopolymer chains
fprintf(f,"** ");
for(int j = 0; j < C[ChNum].N; j++)
{
fprintf(f,"%i ", j);
}
fprintf(f,"\n");
for(int i = 0; i < C[ChNum].N; i++)
{
	fprintf(f,"%i ",i);
	for(int j = 0; j < C[ChNum].N; j++)
	{
		fprintf(f,"%f ", nearestImageR(C[ChNum].M[j], C[ChNum].M[i]));


	}
	fprintf(f,"\n");
}




}




double GyrationRadius(int Chain, int Start, int End)
{
    double Rg = 0;
    Vector CMass,Vg;
    for(int j = 0; j < 3; j++)
    {
        CMass.x[j] = 0;//υσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσισυισυισυισυισυισυισυισυυσιυσυυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσιυσυισυυσιυσιυσιυσι
    }
    for(int i = Start; i <= End; i ++)
    {
        for(int j = 0; j < 3; j++)
        {
        CMass.x[j] += C[Chain].M[i].XReal.x[j];
        }
    }
    for(int j = 0; j < 3; j++)
    {
        CMass.x[j] = CMass.x[j]/(double)(End-Start+1);
    }


    for(int i = Start; i <= End; i ++)
    {
        Rg = Rg + (CMass - C[Chain].M[i].XReal).sqlen();
    }

    return Rg/(double)(End-Start+1);

}


void InitStats(int Len, int StpLen, int minlen, int maxlen,int StartMon = 0)
{
////chain stats
rglen = Len;
rgcounter = 0;
rgmin = minlen;
rgmax = maxlen;
rgstep = StpLen;
rgstart = StartMon;
for(int i = 0; i < rglen; i ++)
{
    rgs[i] = 0;
    rs[i] = 0;
    conts[i] = 0;
}




}


void CalculateStats()
{

    for(int i = rgmin; i < rgmax; i+=rgstep)
    {
        double meanrs = 0;
        double meanrg = 0;
        int cnt = 0;
        int lag = i;
        int cnts = 0;
        for(int j = rand()%lag+rgstart; j < rglen - i; j+=lag)
        {
            meanrs += nearestImageR(C[0].M[j],C[0].M[j+i]);
            if(nearestImageR(C[0].M[j],C[0].M[j+i]) < 1.5)
            {
                cnts++;
            }
            meanrg +=GyrationRadius(0,j,i+j-1);
            cnt++;
        }
        rs[i] += meanrs/(double)cnt;
        rgs[i] += meanrg/(double)cnt;
        conts[i] += cnts/(double)cnt;



    }

    rgcounter++;
}


void outputStats(FILE *ft)
{
    for(int i = 0; i < rgmax; i++)
    {
        if(rs[i]!=0)
        {
            fprintf(ft,"%i %f %f %f \n", i,rs[i]/rgcounter,rgs[i]/rgcounter,conts[i]/rgcounter);


        }
    }
    fprintf(ft,"\n");


}






void StoreConf()
{
    FILE *f = fopen("SavedConf.conf","w");
    fprintf(f,"%i\n",Nchains);
    for(int i= 0; i < Nchains; i++)
    {
        fprintf(f,"%i %i \n",i,C[i].N);
        for(int j = 0; j < C[i].N ; j++)
        {
            fprintf(f,"%i %i %f %f %f %i\n",j,C[i].M[j].Typ,C[i].M[j].XReal.x[0],C[i].M[j].XReal.x[1],C[i].M[j].XReal.x[2],C[i].M[j].NumBonds);
            for(int b = 0; b < C[i].M[j].NumBonds; b++)
            {
                fprintf(f,"%i %i \n", C[i].M[j].BondChains[b], C[i].M[j].Bonds[b]);
            }

        }
    }
    fclose(f);
}


void ReadConf()
{

FILE *f = fopen("SavedConf.conf","r");
    fscanf(f,"%i\n",&Nchains);
    int temp;
    for(int i= 0; i < Nchains; i++)
    {
        fscanf(f,"%i %i \n",&temp,&C[i].N);
        for(int j = 0; j < C[i].N ; j++)
        {
            fscanf(f,"%i %i %lf %lf %lf %i\n",&temp,&C[i].M[j].Typ,&C[i].M[j].XReal.x[0],&C[i].M[j].XReal.x[1],&C[i].M[j].XReal.x[2],&C[i].M[j].NumBonds);
           //  printf("%i %i %f %f %f %i\n",j,C[i].M[j].Typ,C[i].M[j].XReal.x[0],C[i].M[j].XReal.x[1],C[i].M[j].XReal.x[2],C[i].M[j].NumBonds);
            //getch();

            for(int b = 0; b < C[i].M[j].NumBonds; b++)
            {
                fscanf(f,"%i %i \n", &C[i].M[j].BondChains[b], &C[i].M[j].Bonds[b]);
            }
			
			C[i].M[j].coor = NParticles;
			NParticles++;
			C[i].M[j].update();

        }
    }
    fclose(f);
}


void ConfigurationalBias(int nChain, int Typ, bool (* Accept)(double,double),int length, bool end)
{
    Chain temp = C[nChain];
    double oldENV, oldEST, newENV, newEST;
    oldENV = EnergyNV();
    oldEST = EnergyStiff();
    StoreConf();
    //getch();
    if (end)
    {
    ReverseChain(nChain);
    C[nChain].N = C[nChain].N - length;
    C[nChain].M[C[nChain].N-1].NumBonds = 0;
    C[nChain].M[C[nChain].N-1].bondWith(nChain,C[nChain].N-2);
    GenerateGraft(length, nChain, C[nChain].N-1, Typ);
    }
    else
    {
    C[nChain].N = C[nChain].N - length;
      C[nChain].M[C[nChain].N-1].NumBonds = 0;
    C[nChain].M[C[nChain].N-1].bondWith(nChain,C[nChain].N-2);
    GenerateGraft(length, nChain, C[nChain].N-1, Typ);


    }
    if(!Accept(EnergyNV()-oldENV,EnergyStiff()-oldEST))
    {
        C[nChain] = temp;
    }
    else
    {
    CBAccrate++;
    BuildNeighborTable(NEIGHBOR_CUT);
    }
    //StoreConf();
    //getch();

}




void ReverseChain(int nChain)
{
Chain temp = C[nChain];
for(int i = 0; i < temp.N; i ++)
{
    C[nChain].M[i] = temp.M[temp.N-i-1];
    for(int j = 0; j < C[nChain].M[i].NumBonds; j++)
    {
        C[nChain].M[i].Bonds[j] = C[nChain].N - C[nChain].M[i].Bonds[j] - 1;
    }

}
BuildNeighborTable(NEIGHBOR_CUT);

}



void Reptation(int Chain,bool (* Accept)(double,double))
{



}



void GenerateRing(int len, int Typ1)
{

    int timestry = 0;
    Vector rndVect;
    Mon testMon;
	Vector oldshift;
    bool retry;
    // finding first spot
    do
    {

    retry = false;

    for(int i = 0; i < 3 ; i++)
    {
    rndVect.x[i] = (double)(rand()%(int)SIZE[i]);
    }
	
    testMon.X = rndVect;
    testMon.XReal = rndVect;
        for(int i = 0; i < Nchains; i++)
        {
            for(int j = 0; j < C[i].N ; j++ )
            {
                if((C[i].M[j]-testMon).len() < PARTICLE_SIZE)
                {
                    retry = true;
                }
            }
        }

    }while(retry == true);
    testMon.Typ = Typ1;
	testMon.coor = NParticles;
	NParticles++;
	testMon.update();
    testMon.NumBonds = 0;
    testMon.Neighbors = 0;
    C[Nchains].M[0] = testMon;
    int num;
    for(num = 1; num < len; num++)
    {
        // building homopolymer chain Typ1
         timestry = 0;
        Vector shift;
         do{
         timestry++;
         if(timestry > 2000)
         {
          //  GenerateHomopolymer(len,Typ1);
          printf("too long homopolymer generation chain %i mon %i\n",Nchains,num);
          // getch();
            // return;
         }
         retry = false;
         for(int i = 0; i < 3 ; i++)
        {
        shift.x[i] = 1.0 - 2.0*(double)rand()/RAND_MAX;
        }
		if(num < len-40) shift.x[0] = 0.2 - 0.4*(double)rand()/RAND_MAX;
        if(shift.len() < BONDMIN || shift.len() > BONDMAX)
         { retry = true;
       //  printf("problem 4");
        }
         for(int i = 0; i < Nchains; i++)
         {
            for(int j = 0; j < C[i].N ; j++ )
            {
                if(((C[Nchains].M[num-1]+shift)-C[i].M[j]).len() < PARTICLE_SIZE)
                {
              //      printf("problem 1");
                    retry = true;
                }
            }
        }
		 if(num > 1 && num < len-15)
		{
		if( (oldshift*shift) < 0.6)
		{
		retry = true;
		}
		}
        if(((C[Nchains].M[num-1]+shift).XReal-C[Nchains].M[0].XReal).len() > len-num+0.2)
        {
            retry = true;
        }


            for(int j=0; j < num; j++)
            {
                if((C[Nchains].M[j] - (C[Nchains].M[num-1]+shift)).len() < PARTICLE_SIZE && (num-j) > 1) {
                retry = true;
               // printf("problem 2 %f %f ",shift.len(),(C[Nchains].M[j] - (C[Nchains].M[num-1]+shift)).len());
                }
            }
            for(int j=0; j < 3; j++)
            {


            if((C[Nchains].M[num-1]+shift).XReal.x[j] < 0 || ((C[Nchains].M[num-1]+shift).XReal.x[j] > SIZE[j])) {retry = true;
           // printf("problem 3");
            }
            }
        } while( retry == true);
        oldshift = shift;
		C[Nchains].M[num] = C[Nchains].M[num-1] + shift;
        C[Nchains].M[num].NumBonds = 0;
        C[Nchains].M[num].Neighbors = 0;
        C[Nchains].M[num].bondWith(Nchains,num-1);
        C[Nchains].M[num-1].bondWith(Nchains,num);
		C[Nchains].M[num].coor = NParticles;
		NParticles++;
	    C[Nchains].M[num].update();

    }
    C[Nchains].N = len;
    C[Nchains].M[0].bondWith(Nchains,len-1);
    C[Nchains].M[len-1].bondWith(Nchains,0);
    Nchains++;



}


void GenerateGradient(int len, int Typ1, int Typ2,int Typ3, double frequency_typ3)
{

    int timestry = 0;
    Vector rndVect;
    Mon testMon;
	
    bool retry;
    // finding first spot
    do
    {

    retry = false;

    for(int i = 0; i < 3 ; i++)
    {
    rndVect.x[i] = (double)(rand()%(int)SIZE[i]);
    }
	
    testMon.X = rndVect;
    testMon.XReal = rndVect;
        for(int i = 0; i < Nchains; i++)
        {
            for(int j = 0; j < C[i].N ; j++ )
            {
                if((C[i].M[j]-testMon).len() < PARTICLE_SIZE)
                {
                    retry = true;
                }
            }
        }

    }while(retry == true);
    testMon.Typ = Typ1;
    
    testMon.NumBonds = 0;
    testMon.Neighbors = 0;
	testMon.coor = NParticles;
	NParticles++;
	testMon.update();
    C[Nchains].M[0] = testMon;
    
	int num;
    Vector oldshift;
    for(num = 1; num < len; num++)
    {
        // building homopolymer chain Typ1
         timestry = 0;
        Vector shift;
         do{
         timestry++;
         if(timestry > 2000)
         {
          //  GenerateHomopolymer(len,Typ1);
          printf("too long homopolymer generation chain %i mon %i\n",Nchains,num);
          // getch();
            // return;
         }
         retry = false;
         for(int i = 0; i < 3 ; i++)
        {
        shift.x[i] = 1.0 - 2.0*(double)rand()/RAND_MAX;
        }
        if(shift.len() < BONDMIN || shift.len() > BONDMAX)
         { retry = true;
       //  printf("problem 4");
        }

	if(num > 1)
	{
		if( (oldshift*shift) < 0.6)
		{
		retry = true;
		}
	}
         for(int i = 0; i < Nchains; i++)
         {
            for(int j = 0; j < C[i].N ; j++ )
            {
                if(((C[Nchains].M[num-1]+shift)-C[i].M[j]).len() < PARTICLE_SIZE)
                {
              //      printf("problem 1");
                    retry = true;
                }
            }
        }


            for(int j=0; j < num; j++)
            {
                if((C[Nchains].M[j] - (C[Nchains].M[num-1]+shift)).sqlen() < PARTICLE_SIZE && (num-j) > 1) {
                retry = true;
               // printf("problem 2 %f %f ",shift.len(),(C[Nchains].M[j] - (C[Nchains].M[num-1]+shift)).len());
                }
            }
            for(int j=0; j < 3; j++)
            {


            if((C[Nchains].M[num-1]+shift).XReal.x[j] < 0 || ((C[Nchains].M[num-1]+shift).XReal.x[j] > SIZE[j])) {retry = true;
           // printf("problem 3");
            }
            }
        } while( retry == true);
        oldshift = shift;
	 	
		double prob2 = (double)(num )/(len);
		double ts2 = (double)rand()/RAND_MAX;
		double ts = (double)rand()/RAND_MAX;
		
		printf("%lf prob2 %lf ts\n",prob2,ts);
		
	   C[Nchains].M[num] = C[Nchains].M[num-1] + shift;
        C[Nchains].M[num].NumBonds = 0;
        C[Nchains].M[num].Neighbors = 0;
		C[Nchains].M[num].coor = NParticles;
		NParticles++;
	    C[Nchains].M[num].update();
        C[Nchains].M[num].bondWith(Nchains,num-1);
        C[Nchains].M[num-1].bondWith(Nchains,num);
		if(ts2 > frequency_typ3)
		{
		if(ts < prob2)
	    {
			C[Nchains].M[num].Typ = Typ2;
			//printf("typ2 %i chosen",Typ2);
		}
		else
		{
			
			C[Nchains].M[num].Typ = Typ1;
		}
		}
		else
		{
		C[Nchains].M[num].Typ = Typ3;
		}
    }
    C[Nchains].N = len;
    Nchains++;


}

void parseCoordsFromDendrimer(char filename[40], int length)
{
	FILE* f = fopen(filename,"r");
	printf("reading");
	Vector x;
	int Typp;
	int a,b,c,d;
	for(int i = 0; i < length; i++)
	{
	  fscanf(f,"%lf %lf %lf %i %i %i %i",&C[0].M[i].XReal.x[0],&C[0].M[i].XReal.x[1],&C[0].M[i].XReal.x[2],&Typp,&a,&b,&c);
	  for(int j = 0; j < 3; j++) C[0].M[i].XReal.x[j] = C[0].M[i].XReal.x[j]*1.6; 
	  if(Typp == 1)
	  {
		  C[0].M[i].Typ = 1;
	  }
	  else C[0].M[i].Typ = 0;
	  if(i!=0) C[0].M[i].bondWith(0,i-1);
	  if(i!=length-1) C[0].M[i].bondWith(0,i+1);
	  
		C[0].M[i].coor = NParticles;
		NParticles++;
		C[0].M[i].update(); 
	
	}
	C[0].N = length;
	Nchains++;
	
	
	
	
}