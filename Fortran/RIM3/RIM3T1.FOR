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
C   www.boundary-element-method.com/fortran/RIM3T1.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C This program is a test for the subroutine RIM3. The program computes
C  the solution to an acoustic/Helmholtz problem exterior to a circle
C  plate with uniform motion lying in a rigid baffle.
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
C   Limit on the number of acoustic frequencies
      INTEGER    MAXTEST
      PARAMETER (MAXTEST=1000)
C   Limit on the number of points exterior to the plate, where 
C    acoustic properties are sought
      INTEGER    MAXNPE
      PARAMETER (MAXNPE=100)

C  Constants
C   Real scalars: 0, 1, 2, pi
      REAL*8 ZERO,ONE,TWO,THREE,FOUR,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG

C  The reference pressure, used to convert units to decibels.
      REAL*8     PREREF


C  Properties of the acoustic medium
C   The speed of sound [standard unit: metres per second]
      REAL*8     CVAL(MAXTEST)
C   The density [standard unit: kilograms per cubic metre]
      REAL*8     RHOVAL(MAXTEST)

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
C    Number of exterior points and counter
      INTEGER    NPE,IPE
C    Coordinates of the exterior points
      REAL*8     PEXT(MAXNPE,3)

C  Number of test problems and counter
      INTEGER    NTEST,ITEST

C  Data structures that contain the parameters that define the test
C   problems
C   The acoustic frequency for each test. FRVAL(i) is assigned the
C    acoustic frequency of the i-th test problem.
      REAL*8     FRVAL(MAXTEST)
C   The wavenumber for each test. KVAL(i) is assigned the wavenumber
C    of the i-th test problem.
      REAL*8     KVAL(MAXTEST)
C   The nature of the plate condition is specified by assigning 
C    values to the data structures PIALVAL and PIBEVAL. 
C    PIALVAL(i,j) is assigned the value of {\alpha} at the center of the
C     j-th element for the i-th test problem.
      COMPLEX*16 PIALVAL(MAXTEST,MAXNPI)
C    PIBEVAL(i,j) is assigned the value of {\beta} at the center of the
C     j-th element for the i-th test problem.
      COMPLEX*16 PIBEVAL(MAXTEST,MAXNPI)      
C   The actual plate condition is specified by assigning values to 
C    the data structure PIFVAL. 
C    PIFVAL(i,j) is assigned the value of f at the center of the j-th 
C    element for the i-th test problem.
      COMPLEX*16 PIFVAL(MAXTEST,MAXNPI)

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
      COMPLEX*16 PIVEL(MAXNPI)
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
C    At the centres of the elements
C     Sound pressure [standard unit: newtons per square metre (or
C      pascals) and phase] 
      COMPLEX*16 PIPRESS(MAXTEST,MAXNPI)
C     Velocity potential phi
      COMPLEX*16 PIPHIVAL(MAXTEST,MAXNPI)
C  Velocity (v) [standard unit: metres per second (and phase)]
C   At the centres of the elements 
      COMPLEX*16 PIV(MAXTEST,MAXNPI)
      REAL*8 SINTY(MAXTEST,MAXNPI)
      
C  Counter through the x,y coordinates
      INTEGER    ICOORD

C  Local storage of pressure, pressure/velocity 
      COMPLEX*16 PRESSURE

C  The coordinates of the centres of the elements  
C      REAL*8     SELCNT(MAXNPI,3)

      REAL*8     EPS


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

C  Reference for decibel scales
      PREREF=2.0D-05


C Describe the nodes and elements that make up the plate
C  :The unit circle, centred at the point (1.0,0.0,0.0) and lying in the
C  : y-z plane is divided into NPI=32 uniform elements. VERTEX and PIELV 
C  : are defined anti-clockwise around the plate so that the normal to 
C  : the plate is assumed to be outward
C  :Set up nodes
C  : Set NPI, the number of elements
      NPI=24
C  : Set NV, the number of vertices (equal to the number of elements)
      NV=25
C  : Set coordinates of the nodes


C  : Set up VERTEX, the coordinates of the vertices of the elements
      NV=19
      DATA ((VERTEX(IV,ICOORD),ICOORD=1,3),IV=1,19)
     * / 1.000D0, 0.000D0, 0.000D0,
     *   1.000D0, 0.025D0, 0.043D0,
     *   1.000D0, 0.050D0, 0.000D0,
     *   1.000D0, 0.025D0,-0.043D0,
     *   1.000D0,-0.025D0,-0.043D0,
     *   1.000D0,-0.050D0, 0.000D0,
     *   1.000D0,-0.025D0, 0.043D0,
     *   1.000D0, 0.050D0, 0.087D0,
     *   1.000D0, 0.087D0, 0.050D0,
     *   1.000D0, 0.100D0, 0.000D0,
     *   1.000D0, 0.087D0,-0.050D0,
     *   1.000D0, 0.050D0,-0.087D0,
     *   1.000D0, 0.000D0,-0.100D0,
     *   1.000D0,-0.050D0,-0.087D0,
     *   1.000D0,-0.087D0,-0.050D0,
     *   1.000D0,-0.100D0, 0.000D0,
     *   1.000D0,-0.087D0, 0.050D0,
     *   1.000D0,-0.050D0, 0.087D0,
     *   1.000D0, 0.000D0, 0.100D0/


C  : Set nodal indices that describe the elements of the plate
C  :  The indices refer to the nodes in VERTEX. The order of the
C  :  nodes in PIELV dictates that the normal is outward from the 
C  :  plate into the acoustic domain.
      DATA ((PIELV(IPI,ICOORD),ICOORD=1,3),IPI=1,24)
     * / 1, 2, 7,    1, 3, 2,    1, 4, 3,    1, 5, 4,
     *   1, 6, 5,    1, 7, 6,    8, 2, 9,    9, 2, 3,
     *   9, 3,10,   10, 3,11,   11, 3, 4,   11, 4,12,    
     *  12, 4,13,   13, 4, 5,   13, 5,14,   14, 5,15,
     *  15, 5, 6,   15, 6,16,   16, 6,17,   17, 6, 7,
     *  17, 7,18,   18, 7,19,   19, 7, 2,   19, 2, 8/


C Set NPE=100 and set the exterior points to be the points on the axis
C  of the vibrating piston
      NPE=100
      DO 110 IPE=1,NPE
        PEXT(IPE,1)=1.0D0+DFLOAT(IPE)/100.0D0
        PEXT(IPE,2)=ZERO
        PEXT(IPE,3)=ZERO
110   CONTINUE

C The number of points on the plate is equal to the number of 
C  elements
      NPIP=NPI
        
C Set up test problems
C  :Set the number of test problems
      NTEST=2

C  : Set the wavenumber in KVAL
      KVAL(1)=10.0D0
      KVAL(2)=25.0D0

      DO 200 ITEST=1,NTEST

C  Properties of the acoustic medium. C the propagation velocity
C  and RHO the density of the acoustic medium. C>0, RHO>0
C  :Acoustic medium is air at 20 celcius and 1 atmosphere. 
C  [C in metres per second, RHO in kilograms per cubic metre.]
      CVAL(ITEST)=344.0D0
      RHOVAL(ITEST)=1.205D0

C  :Set acoustic frequency value (hertz) in FRVAL
      FRVAL(ITEST)=CVAL(ITEST)*KVAL(ITEST)/2.0D0/PI

C  :Set nature of the plate condition by prescribing the values of
C   the plate functions PIALVAL and PIBEVAL at the collocation points
C   :In this case a Dirichlet (phi-valued) plate condition
      DO 160 ISP=1,NPIP
        PIALVAL(ITEST,ISP)=CZERO
        PIBEVAL(ITEST,ISP)=CONE
        PIFVAL(ITEST,ISP)=1.0D0        
160   CONTINUE


200   CONTINUE

C  :Switch for particular solution
      LSOL=.TRUE.
C  :Switch on the validation of RIM3
      LVALID=.TRUE.
C  :Set EGEOM
      EGEOM=1.0D-6

C  Open file for the output data
      OPEN(UNIT=20,FILE='RIM3T1.OUT')

C Loop(ITEST) through the test problems
      DO 500 ITEST=1,NTEST
C  Set OMEGA, the angular frequency omega and K, the wavenumber
        K=KVAL(ITEST)
        OMEGA=2.0D0*PI*FRVAL(ITEST)

C   Set up particular alpha and beta functions for this wavenumber
C    and type of plate condition
        DO 510 ISP=1,NPIP
          PIALPHA(ISP)=PIALVAL(ITEST,ISP)
          PIBETA(ISP)=PIBEVAL(ITEST,ISP)
          PIF(ISP)=PIFVAL(ITEST,ISP)
510     CONTINUE
       
        CALL RIM3(K,
     *            MAXNV,NV,VERTEX,MAXNPI,NPI,PIELV,
     *            MAXNPE,NPE,PEXT,
     *            PIALPHA,PIBETA,PIF,
     *            LSOL,LVALID,EGEOM,
     *            PIPHI,PIVEL,PEPHI,
     *            WKSPC1,WKSPC2,WKSPC3,WKSPC4,WKSPC5,WKSPC6)


C Compute the sound pressure at the exterior points. Also compute
C  the velocity and intensity at the points for each type of plate
C  condition and each related input function f and at each point.
        DO 520 ISP=1,NPIP
          PIPHIVAL(ITEST,ISP)=PIPHI(ISP)
          PRESSURE=CIMAG*RHOVAL(ITEST)*OMEGA*PIPHI(ISP)
          PIPRESS(ITEST,ISP)=PRESSURE
          PIV(ITEST,ISP)=PIVEL(ISP)
          SINTY(ITEST,ISP)=
     *         DBLE(CONJG(PRESSURE)*PIVEL(ISP))/TWO

520     CONTINUE


C Output the solutions

C  Formats for output
 2850   FORMAT(5X,'index',7X,'Potential',19X,'Pressure',24X,
     *       'Velocity',17X,'Intensity'/)
 2860   FORMAT(4X,I4,2X,E10.4,'+ ',E10.4,' i    ',
     *       E10.4, '+ ',E10.4,' i    ',4X,
     *       E10.4, '+ ',E10.4,7X,F10.4)
        WRITE(20,*) 'k = ',KVAL(ITEST)
        WRITE(20,2850)
        WRITE(20,*)
C      Loop(ISP) through the points on the boundary
        DO 2030 ISP=1,NPIP
C       Output the sound pressure, velocity and intensity at each point
          WRITE(20,2860) ISP,DBLE(PIPHIVAL(ITEST,ISP)),
     *     AIMAG(PIPHIVAL(ITEST,ISP)),DBLE(PIPRESS(ITEST,ISP)),
     *     AIMAG(PIPRESS(ITEST,ISP)),DBLE(PIV(ITEST,ISP)),
     *     AIMAG(PIV(ITEST,ISP)), SINTY(ITEST, ISP)
2030    CONTINUE
        WRITE(20,*) '   x-coordinate  real phi     imaginary phi'
        DO 600 IPE=1,NPE
          WRITE(20,999) PEXT(IPE,1),DBLE(PEPHI(IPE)),DIMAG(PEPHI(IPE))
600    CONTINUE

C  Close loop(ITEST) through the test problems
500   CONTINUE

      CLOSE(20)

999   FORMAT(3F14.8)

      END
