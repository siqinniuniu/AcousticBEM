C***************************************************************
C    Test program for subroutine RIM3 by Stephen Kirkup       
C***************************************************************
C
C  Copyright 2004- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/RIM3T2.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C
C This program is a test for the subroutine RIM3. The program computes
C  the solution to an acoustic/Helmholtz problem exterior to a square
C  plate lying in a rigid baffle.
C
C Background
C ----------
C
C The Helmholtz problem arises when harmonic solutions of the wave 
C  equation
C                                     2
C         __ 2                 1     d   {\Psi}(p,t)
C         \/   {\Psi}(p,t) -  ----   ---------------   =  0
C                               2        2
C                              c      d t
C                
C  are sought, where {\Psi}(p,t) is the scalar time-dependent velocity
C  potential. In the cases where {\Psi} is periodic, it may be 
C  approximated as a set of frequency components that may be analysed
C  independently. For each frequency a component of the form
C
C                      {\phi}(p) exp(i {\omega} t)
C
C  (where {\omega} = 2 * {\pi} * frequency) the wave equation can be
C  reduced to the Helmholtz equation
C
C                  __ 2                2
C                  \/    {\phi}   +   k  {\phi}   =  0  
C
C  where k (the wavenumber) = {\omega}/c (c=speed of sound in the 
C  medium. {\phi} is known as the velocity potential.
C
C For the exterior problem, the domain lies exterior to a closed 
C  plate \Pi. The plate condition may be Dirichlet, Robin or 
C  Neumann. It is assumed to have the following general form
C
C            {\alpha}(q) {\phi}(q) + {\beta}(q) v(q) = f(q)
C    
C  where {\phi}(q) is the velocity potential at the point q on \Pi, v(q) 
C  is the derivative of {\phi} with respect to the outward normal to \Pi 
C  at q and {\alpha}, {\beta} and f are complex-values functions defined
C   on \Pi. 
C
C Subroutine RIM3 accepts the wavenumber, a description of the 
C  plate of the domain and the position of the exterior points
C  where the solution ({\phi}) is sought, the plate condition and
C  returns the solution ({\phi} and v) on \Pi and the value of {\phi}
C  at the exterior points.
C

C The test problems
C -----------------
C
C In this test the domain is a square plate with side of length 1m
C  lying in an infinite rigid baffle. The acoustic medium is air (at 20 
C  celcius and 1 atmosphere, c=344.0 (metres per second), density 
C  {\rho}=1.205 (kilograms per cubic metre) and the solution to the 
C  problem with a Neumann plate condition v=1 (that is \alpha=1, \beta=0, 
C  f=1).

C The plate is described by a set of NPI=32 planar triangular elements
C  of equal size. The radiation ratio is computed for a range of
C  frequencies.

C----------------------------------------------------------------------

C The PARAMETER statement
C -----------------------
C There are four components in the PARAMETER statement.
C integer MAXNS   : The limit on the number of plate elements.
C integer MAXNV   : The limit on the number of vertices.
C integer MAXNFR  : The limit on the number of frequencies.
C integer MAXNPE  : The limit on the number of exterior points.


C External modules related to the package
C ---------------------------------------
C subroutine RIM3: Subroutine for solving the exterior Helmholtz
C  equation (file EBEM2.FOR contains EBEM2 and subordinate routines)
C subroutine H3LC: Returns the individual discrete Helmholtz integral
C  operators. (file H3LC.FOR contains H3LC and subordinate routines)
C subroutine CGLS: Solves a general linear system of equations.
C  (file CGLS.FOR contains CGSL and subordinate routines)
C file GEOM3D.FOR contains the set of relevant geometric subroutines


C The program 

      PROGRAM  RIM3T
      IMPLICIT NONE

C VARIABLE DECLARATION
C --------------------

C  PARAMETERs for storing the limits on the dimension of arrays
C   Limit on the number of elements
      INTEGER    MAXNPI
      PARAMETER (MAXNPI=40)
C   Limit on the number of vertices (equal to the number of elements)
      INTEGER    MAXNV
      PARAMETER (MAXNV=40)
C   Limit on the number of points exterior to the plate, where 
C    acoustic properties are sought
      INTEGER    MAXNPE
      PARAMETER (MAXNPE=6)

C  Constants
C   Real scalars: 0, 1, 2, pi
      REAL*8 ZERO,ONE,TWO,THREE,FOUR,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG


C  Properties of the acoustic medium
C   The speed of sound [standard unit: metres per second]
      REAL*8     CVAL
C   The density [standard unit: kilograms per cubic metre]
      REAL*8     RHOVAL

C   Wavenumber parameter for RIM3
      REAL*8     K
C   Angular frequency 
      COMPLEX*16 OMEGA

C  Geometrical description of the plate(ies)
C   Number of elements and counter
      INTEGER    NPI,IPI
C   Number of collocation points (on \Pi) and counter
      INTEGER    NPIP,ISP
C   Number of vetices and counter
      INTEGER    NV,IV
C   Index of nodal coordinate for defining boundaries (standard unit is 
C    metres)
      REAL*8     VERTEX(MAXNV,3)
C   The three nodes that define each element on the boundaries
      INTEGER    PIELV(MAXNPI,3)
C   The points exterior to the plate(ies) where the acoustic 
C    properties are sought and the directional vectors at those points.
C    [Only necessary if an exterior solution is sought.]
C    Number of exterior points
      INTEGER    NPE
C    Coordinates of the exterior points
      REAL*8     PEXT(MAXNPE,3)

C  Data structures that contain the parameters that define the test
C   problems
C   The acoustic frequency for each test. FRVAL is assigned the
C    acoustic frequency of the i-th test problem.
      REAL*8     FRVAL

C   Data structures that are used to define each test problem in turn
C    and are input parameters to RIM3.
C    PIALPHA(j) is assigned the value of {\alpha} at the centre of the 
C     j-th element.
      COMPLEX*16 PIALPHA(MAXNPI)
C    PIBETA(j) is assigned the value of {\beta} at the centre of the 
C     j-th element.
      COMPLEX*16 PIBETA(MAXNPI)
C    PIF(j) is assigned the value of f at the centre of the j-th element.
      COMPLEX*16 PIF(MAXNPI)


C  Validation and control parameters for RIM3
C   Switch for particular solution
      LOGICAL    LSOL
C   Validation switch
      LOGICAL    LVALID
C   The maximum absolute error in the parameters that describe the
C    geometry of the plate.
      REAL*8     EGEOM

C Output from subroutine RIM3
C  The velocity potential (phi - the solution) at the centres of the 
C   elements
      COMPLEX*16 PIPHI(MAXNPI)
C  The normal derivative of the velocity potential at the centres of the
C    elements
      COMPLEX*16 PIV(MAXNPI)
C  The velocity potential (phi - the solution) at exterior points
      COMPLEX*16 PEPHI(MAXNPE)

C Workspace for RIM3
      COMPLEX*16 WKSPC1(MAXNPI,MAXNPI)
      COMPLEX*16 WKSPC2(MAXNPE,MAXNPI)
      COMPLEX*16 WKSPC3(MAXNPI,MAXNPI)
      COMPLEX*16 WKSPC4(MAXNPI)
      COMPLEX*16 WKSPC5(MAXNPI)
      LOGICAL    WKSPC6(MAXNPI)


C   Acoustic properties. These data structures are appended after each
C    execution of RIM3 and contain the numerical solution to the test
C    problems. 
C   Sound intensity [standard unit: watts per square metre]
      REAL*8     PIINTY(MAXNPI)
      REAL*8     POWER,BAFFPOW

C  Counter through the x,y coordinates
      INTEGER    ICOORD

C  Local storage of pressure, pressure/velocity 
      COMPLEX*16 PRESSURE

C  The coordinates of the centres of the elements  
      REAL*8     SELCNT(MAXNPI,3)

      REAL*8     EPS

C  Number of test problems (wavenumbers) and counter
      INTEGER    NTEST,ITEST


C INITIALISATION
C --------------

C Set constants
      ZERO=0.0D0
      ONE=1.0D0
      TWO=2.0D0
      THREE=3.0D0
      FOUR=4.0D0
      PI=4.0D0*ATAN(ONE)
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)
      CIMAG=CMPLX(ZERO,ONE)

      EPS=1.0E-10


C Describe the nodes and elements that make up the plate
C  :The square is divided into NPI=32 uniform elements. VERTEX and 
C  : PIELV are defined anti-clockwise around the plate so that the 
C  : normal to the plate is assumed to be outward
C  :Set up nodes
C  : Set NPI, the number of elements
      NPI=32
C  : Set NV, the number of vertices (equal to the number of elements)
      NV=25


C  : Set up VERTEX, the coordinates of the vertices of the elements
      NV=25
      DATA ((VERTEX(IV,ICOORD),ICOORD=1,3),IV=1,25)
     * / 0.000D0, 0.000D0, 0.000D0,
     *   0.250D0, 0.000D0, 0.000D0,
     *   0.500D0, 0.000D0, 0.000D0,
     *   0.750D0, 0.000D0, 0.000D0,
     *   1.000D0, 0.000D0, 0.000D0,
     *   0.000D0, 0.250D0, 0.000D0,
     *   0.250D0, 0.250D0, 0.000D0,
     *   0.500D0, 0.250D0, 0.000D0,
     *   0.750D0, 0.250D0, 0.000D0,
     *   1.000D0, 0.250D0, 0.000D0,
     *   0.000D0, 0.500D0, 0.000D0,
     *   0.250D0, 0.500D0, 0.000D0,
     *   0.500D0, 0.500D0, 0.000D0,
     *   0.750D0, 0.500D0, 0.000D0,
     *   1.000D0, 0.500D0, 0.000D0,
     *   0.000D0, 0.750D0, 0.000D0,
     *   0.250D0, 0.750D0, 0.000D0,
     *   0.500D0, 0.750D0, 0.000D0,
     *   0.750D0, 0.750D0, 0.000D0,
     *   1.000D0, 0.750D0, 0.000D0,
     *   0.000D0, 1.000D0, 0.000D0,
     *   0.250D0, 1.000D0, 0.000D0,
     *   0.500D0, 1.000D0, 0.000D0,
     *   0.750D0, 1.000D0, 0.000D0,
     *   1.000D0, 1.000D0, 0.000D0/


C  : Set nodal indices that describe the elements of the plate.
C  :  The indices refer to the nodes in VERTEX. The order of the
C  :  nodes in PIELV dictates that the normal is outward from the 
C  :  plate into the acoustic domain.
      DATA ((PIELV(IPI,ICOORD),ICOORD=1,3),IPI=1,32)
     * / 1, 2, 6,    2, 7, 6,    2, 3, 8,    2, 8, 7,
     *   3, 4, 8,    4, 9, 8,    4, 5,10,    4,10, 9,
     *   6, 7,12,    6,12,11,    7, 8,12,    8,13,12,    
     *   8, 9,14,    8,14,13,    9,10,14,   10,15,14,
     *   11,12,16,   12,17,16,   12,13,18,   12,18,17,
     *   13,14,18,   14,19,18,   14,15,20,   14,20,19,
     *   16,17,22,   16,22,21,   17,18,22,   18,23,22,
     *   18,19,24,   18,24,23,   19,20,24,   20,25,24/


C Set the centres of the elements, the collocation points
      DO 100 IPI=1,NPI
        SELCNT(IPI,1)=(VERTEX(PIELV(IPI,1),1)
     *   +VERTEX(PIELV(IPI,2),1)+VERTEX(PIELV(IPI,3),1))/THREE
        SELCNT(IPI,2)=(VERTEX(PIELV(IPI,1),2)
     *   +VERTEX(PIELV(IPI,2),2)+VERTEX(PIELV(IPI,3),2))/THREE
        SELCNT(IPI,3)=(VERTEX(PIELV(IPI,1),3)
     *   +VERTEX(PIELV(IPI,2),3)+VERTEX(PIELV(IPI,3),3))/THREE
100   CONTINUE


C Set NPE=1
      NPE=1
      PEXT(1,1)=0.5D0
      PEXT(1,2)=0.5D0
      PEXT(1,3)=0.1D0

C The number of points on the plate is equal to the number of 
C  elements
      NPIP=NPI
        
C Set up test problems
C  :Set the number of test problems
      NTEST=100

C  Properties of the acoustic medium. C the propagation velocity
C  and RHO the density of the acoustic medium. C>0, RHO>0
C  :Acoustic medium is air at 20 celcius and 1 atmosphere. 
C  [C in metres per second, RHO in kilograms per cubic metre.]
      CVAL=344.0D0
      RHOVAL=1.205D0

C  Open file for the output data
      OPEN(UNIT=20,FILE='RIM3T2.OUT')

      DO 200 ITEST=1,NTEST

C  :Set acoustic frequency value (hertz) in FRVAL
        FRVAL=10.0D0*DFLOAT(ITEST)

C  : Set the wavenumber in K
        K=TWO*PI*FRVAL/CVAL

C   Set up particular alpha and beta functions for this wavenumber
C    and type of plate condition
        DO 310 ISP=1,NPIP
          PIALPHA(ISP)=CZERO
          PIBETA(ISP)=CONE
          PIF(ISP)=CONE
310     CONTINUE
       

C  :Switch for particular solution
        LSOL=.TRUE.
C  :Switch on the validation of RIM3
        LVALID=.TRUE.
C  :Set EGEOM
        EGEOM=1.0D-6


C  Set OMEGA, the angular frequency omega and K, the wavenumber
        OMEGA=2.0D0*PI*FRVAL

C  Call RIM3
        CALL RIM3(K,
     *            MAXNV,NV,VERTEX,MAXNPI,NPI,PIELV,
     *            MAXNPE,NPE,PEXT,
     *            PIALPHA,PIBETA,PIF,
     *            LSOL,LVALID,EGEOM,
     *            PIPHI,PIV,PEPHI,
     *            WKSPC1,WKSPC2,WKSPC3,WKSPC4,WKSPC5,WKSPC6)

C  Compute the pressure and the intensity at points on the plate
        DO 400 ISP=1,NPIP
          PRESSURE=CIMAG*RHOVAL*OMEGA*PIPHI(ISP)
          PIINTY(ISP)=DBLE(CONJG(PRESSURE)*PIV(ISP))/TWO
400     CONTINUE

C Compute the power and radiation ratio
        POWER=0.0D0
        BAFFPOW=0.0D0
        DO 695 ISP=1,NPIP
          POWER=POWER+PIINTY(ISP)
          BAFFPOW=BAFFPOW+RHOVAL*CVAL*ABS(CONJG(PIV(ISP))*PIV(ISP))/TWO
695     CONTINUE
C Output the radiation ratio
        WRITE(20,*) K,POWER/BAFFPOW

C  Close loop(ITEST) through the test problems
200   CONTINUE

      CLOSE(20)
      END
