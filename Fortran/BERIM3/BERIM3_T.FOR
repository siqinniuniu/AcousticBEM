C***************************************************************
C    Test program for subroutine BERIM3 by Stephen Kirkup     
C***************************************************************
C
C
C  Copyright 2004- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/BERIM3_T.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************

C This program is a test for the subroutine BERIM3. The program computes
C  the solution to an acoustic/Helmholtz problem interior to a truncated
C  sphere with an opening by a hybrid of the boundary element method and
C  the Rayleigh Integral Method.
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
C  medium). {\phi} is known as the velocity potential.
C
C For the interior problem, the domain lies interior to a closed 
C  boundary S. The boundary condition may be Dirichlet, Robin or 
C  Neumann. It is assumed to have the following general form
C
C            {\alpha}(q) {\phi}(q) + {\beta}(q) v(q) = f(q)
C    
C  where {\phi}(q) is the velocity potential at the point q on S, v(q) 
C  is the derivative of {\phi} with respect to the outward normal to S 
C  at q and {\alpha}, {\beta} and f are complex-valued functions defined
C   on S. 
C
C Subroutine BERIM3 accepts the wavenumber, a description of the 
C  boundary of the domain and the position of the interior points
C  where the solution ({\phi}) is sought, the boundary condition and
C  returns the solution ({\phi} and v) on S and the value of {\phi}
C  at the interior points.
C

C The test problems
C -----------------
C
C In this test the domain is a truncated sphere of radius 1 (metre) but 
C  with an open top. The acoustic
C  medium is air (at 20 celcius and 1 atmosphere, c=344.0 (metres per
C  second), density {\rho}=1.205 (kilograms per cubic metre) and the 
C  solution to the problem with a Dirichlet boundary condition 
C  ({\alpha}=1, {\beta}=0) and with a Neumann boundary condition 
C  ({\alpha}=0, beta=1) are sought. For the test problem the frequency is
C  4000Hz (hence specifying k).
C
C The boundary is described by a set of NS=36 planar triangular elements
C  of approximately equal size. The boundary solution points are the 
C  centres of the elements. 
C The solution is sought at the interior points (0,0), (0,0.5), 
C  (0.25,0.25).

C----------------------------------------------------------------------

C The PARAMETER statement
C -----------------------
C There are four components in the PARAMETER statement.
C integer MAXNS   : The limit on the number of boundary elements.
C integer MAXNV   : The limit on the number of vertices.
C integer MAXNPI  : The limit on the number of interior points.


C External modules related to the package
C ---------------------------------------
C subroutine BERIM3: Subroutine for solving the interior Helmholtz
C  equation (file BERIM3.FOR contains the subroutine BERIM3)
C subroutine H3LC: Returns the individual discrete Helmholtz integral
C  operators. (file H3LC.FOR contains H3LC and subordinate routines)
C subroutine CGLS: Solves a general linear system of equations.
C  (file CGLS.FOR contains CGSL and subordinate routines)
C file GEOM3D.FOR contains the set of relevant geometric subroutines


C The program 

      PROGRAM B3T
      IMPLICIT NONE

C VARIABLE DECLARATION
C --------------------

C  PARAMETERs for storing the limits on the dimension of arrays
C   Limit on the number of elements
      INTEGER    MAXNS
      PARAMETER (MAXNS=40)
C   Limit on the number of vertices (equal to the number of elements)
      INTEGER    MAXNV
      PARAMETER (MAXNV=30)
C   Limit on the number of points interior to the cavity, where 
C    acoustic properties are sought
      INTEGER    MAXNPI
      PARAMETER (MAXNPI=6)
C   Limit on the number of points exterior to the cavity, where 
C    acoustic properties are sought
      INTEGER    MAXNPE
      PARAMETER (MAXNPE=6)

C  Constants
C   Real scalars: 0, 1, 2, pi
      REAL*8 ZERO,ONE,TWO,THREE,FOUR,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG

C  Properties of the acoustic medium
C   Wavenumber parameter for BERIM3
      REAL*8     K
C   Angular frequency 
      COMPLEX*16 OMEGA
C   The reference pressure, used to convert units to decibels.
      REAL*8     PREREF

C  Geometrical description of the boundary(ies)
C   Number of elements and counter
      INTEGER    NS,IS
C   Number of collocation points (on S) and counter
      INTEGER    NSP,ISP
C   Number of vetices and counter
      INTEGER    NV,IV
C   Index of nodal coordinate for defining boundaries (standard unit is 
C    metres)
      REAL*8     VERTEX(MAXNV,3)
C   The three nodes that define each element on the boundaries
      INTEGER    SELV(MAXNS,3)
C   The points interior to the boundary(ies) where the acoustic 
C    properties are sought and the directional vectors at those points.
C    [Only necessary if an interior solution is sought.]
C    Number of interior points and counter
      INTEGER    NPI,IPI
C    Coordinates of the interior points
      REAL*8     PINT(MAXNPI,3)
C    Areas of the elements
      REAL*8     ELAREA(MAXNS)


C   Data structures that are used to define each test problem in turn
C    and are input parameters to BERIM3.
C    SALPHA(j) is assigned the value of {\alpha} at the centre of the 
C     j-th element.
      COMPLEX*16 SALPHA(MAXNS)
C    SBETA(j) is assigned the value of {\beta} at the centre of the 
C     j-th element.
      COMPLEX*16 SBETA(MAXNS)
C    SF(j) is assigned the value of f at the centre of the j-th element.
      COMPLEX*16 SF(MAXNS)


C  Validation and control parameters for BERIM3
C   Switch for particular solution
      LOGICAL    LSOL
C   Validation switch
      LOGICAL    LVALID
C   The maximum absolute error in the parameters that describe the
C    geometry of the boundary.
      REAL*8     EGEOM

C Output from subroutine BERIM3
C  The velocity potential (phi - the solution) at the centres of the 
C   elements
      COMPLEX*16 SPHI(MAXNS)
C  The normal derivative of the velocity potential at the centres of the
C    elements
      COMPLEX*16 SVEL(MAXNS)
C  The velocity potential (phi - the solution) at cavity interior 
C   points
      COMPLEX*16 PIPHI(MAXNPI)
C  The velocity potential (phi - the solution) at exterior points
      COMPLEX*16 PEPHI(MAXNPI)

C  Exterior points at which the solution is to be observed
C   The number of interior points and counter
      INTEGER    NPE,IPE
C   Coordinates of the interior points
      REAL*8     PEXT(MAXNPE,3)

C  Working space
C   For BERIM3 routine
      COMPLEX*16 BIGMAT(2*MAXNS,2*MAXNS)
      COMPLEX*16 INPVEC(2*MAXNS)
      COMPLEX*16 SOLVEC(2*MAXNS)
      COMPLEX*16 LO(MAXNPE,MAXNS)
      COMPLEX*16 LS(MAXNPI,MAXNS)
      COMPLEX*16 MS(MAXNPI,MAXNS)


C   Acoustic properties. These data structures are appended after each
C    execution of BERIM3 and contain the numerical solution to the test
C    problems. 
C    At the centres of the elements
C     Sound pressure [standard unit: newtons per square metre (or
C      pascals) and phase] 
      COMPLEX*16 SPRESS(MAXNS)
C     Sound intensity [standard unit: watts per square metre]
      REAL*8     SINTY(MAXNS)
C     Sound power, as measured on the cavity surface
      REAL*8     CPOWER
C     Sound power, as measured at the opening
      REAL*8     OPOWER

C    At the interior points
C     Sound pressure [standard unit: newtons per square metre (or
C      pascals) and phase]
c      Interior 
      COMPLEX*16 IPRESS(MAXNPI)
C      Exterior
      COMPLEX*16 EPRESS(MAXNPI)

C  Counter through the x,y coordinates
      INTEGER    ICOORD

      REAL*8     C,RHO,FREQ

C  Local storage of pressure, pressure/velocity 
      COMPLEX*16 PRESSURE


C  Other variables used in specifying the boundary condition
      REAL*8     QA(3),QB(3),QC(3),AREA

      REAL*8     EPS

C      REAL*8     SIZE3,DOT3,DRBDN

      INTEGER    OELEND

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

      EPS=1.0D-10

C  Reference for decibel scales
      PREREF=2.0D-05


C Describe the nodes and elements that make up the boundary
C  :The unit sphere, centred at the origin is divided 
C  : into NS=36 uniform elements. VERTEX and SELV are defined 
C  : anti-clockwise around the boundary so that the normal to the 
C  : boundary is assumed to be outward
C  :Set up nodes
C  : Set NS, the number of elements
C  : Set coordinates of the nodes


C   Set up VERTEX, the coordinates of the vertices of the elements
      NV=20
      DATA ((VERTEX(IV,ICOORD),ICOORD=1,3),IV=1,20)
     * / 0.000D0, 0.000D0, 0.667D0,
     *   0.000D0, 0.745D0, 0.667D0,
     *   0.645D0, 0.372D0, 0.667D0,
     *   0.645D0,-0.372D0, 0.667D0,
     *   0.000D0,-0.745D0, 0.667D0,
     *  -0.645D0,-0.372D0, 0.667D0,
     *  -0.645D0, 0.372D0, 0.667D0,
     *   0.500D0, 0.866D0, 0.000D0,
     *   1.000D0, 0.000D0, 0.000D0,
     *   0.500D0,-0.866D0, 0.000D0,
     *  -0.500D0,-0.866D0, 0.000D0,
     *  -1.000D0, 0.000D0, 0.000D0,
     *  -0.500D0, 0.866D0, 0.000D0,
     *   0.000D0, 0.745D0,-0.667D0,
     *   0.645D0, 0.372D0,-0.667D0,
     *   0.645D0,-0.372D0,-0.667D0,
     *   0.000D0,-0.745D0,-0.667D0,
     *  -0.645D0,-0.372D0,-0.667D0,
     *  -0.645D0, 0.372D0,-0.667D0,
     *   0.000D0, 0.000D0,-1.000D0/


C  : Set nodal indices that describe the elements of the boundarys.
C  :  The indices refer to the nodes in VERTEX. The order of the
C  :  nodes in SELV dictates that the normal is outward from the 
C  :  boundary into the acoustic domain.
      NS=36
      DATA ((SELV(IS,ICOORD),ICOORD=1,3),IS=1,36)
     * / 1, 3, 2,    1, 4, 3,    1, 5, 4,    1, 6, 5,
     *   1, 7, 6,    1, 2, 7,    2, 3, 8,    3, 9, 8,
     *   3, 4, 9,    4,10, 9,    4, 5,10,    5,11,10,
     *   5, 6,11,    6,12,11,    6, 7,12,    7,13,12,
     *   7, 2,13,    2, 8,13,    8,15,14,    8, 9,15,
     *   9,16,15,    9,10,16,   10,17,16,   10,11,17,
     *  11,18,17,   11,12,18,   12,19,18,   12,13,19,
     *  13,14,19,   13, 8,14,   14,15,20,   15,16,20,
     *  16,17,20,   17,18,20,   18,19,20,   19,14,20/

      OELEND=6

C Set the points in the cavity where the acoustic properties
C  are sought, PINT. 
C : Let them be the points (0.000,0.000), (0.000,0.500), (0.250,0.250).
      NPI=4
      DATA ((PINT(IPI,ICOORD),ICOORD=1,3),IPI=1,4)
     *  /  0.500D0,     0.000D0,  0.0000D0,
     *     0.000D0,     0.000D0,  0.010D0,
     *     0.000D0,     0.000D0,  0.250D0,
     *     0.000D0,     0.000D0,  0.500D0/

C Set the points in the acoustic domain where the acoustic properties
C  are sought, PINT. 
C : Let them be the points (0.000,0.000), (0.000,0.500), (0.250,0.250).
      NPE=3
      DATA ((PEXT(IPE,ICOORD),ICOORD=1,3),IPE=1,3)
     *  /  0.000D0,     0.000D0,  1.000D0,
     *     0.000D0,     0.000D0,  2.000D0,
     *     0.000D0,     0.000D0,  10.00D0/

      
      DO 300 IS=1,NS
        QA(1)=VERTEX(SELV(IS,1),1)
        QA(2)=VERTEX(SELV(IS,1),2)
        QA(3)=VERTEX(SELV(IS,1),3)
        QB(1)=VERTEX(SELV(IS,2),1)
        QB(2)=VERTEX(SELV(IS,2),2)
        QB(3)=VERTEX(SELV(IS,2),3)
        QC(1)=VERTEX(SELV(IS,3),1)
        QC(2)=VERTEX(SELV(IS,3),2)
        QC(3)=VERTEX(SELV(IS,3),3)
        ELAREA(IS)=AREA(QA,QB,QC)
300   CONTINUE


C The number of points on the boundary is equal to the number of 
C  elements
      NSP=NS

C  TEST PROBLEM
C  ============
C  Properties of the acoustic medium. C the propagation velocity
C  and RHO the density of the acoustic medium. C>0, RHO>0
C  :Acoustic medium is air at 20 celcius and 1 atmosphere. 
C  [C in metres per second, RHO in kilograms per cubic metre.]
      C=344.0D0
      RHO=1.205D0

C  :Set acoustic frequency value (hertz) in FRVAL
      FREQ=50.0D0

C  : Set the wavenumber in KVAL
      K=TWO*PI*FREQ/C

C  :Set nature of the boundary condition by prescribing the values of
C   the boundary functions SALVAL and SBEVAL at the collocation points
C   :In this case a Dirichlet (phi-valued) boundary condition
      DO 240 ISP=OELEND+1,NSP
        SALPHA(ISP)=CZERO
        SBETA(ISP)=CONE
        IF (ISP.GT.30) THEN
          SF(ISP)=CONE
        ELSE
          SF(ISP)=CZERO
        END IF
240   CONTINUE


C  :Switch for particular solution
      LSOL=.TRUE.
C  :Switch on the validation of BERIM3
      LVALID=.TRUE.
C  :Set EGEOM
      EGEOM=1.0D-6

      OMEGA=2.0D0*PI*FREQ


      CALL BERIM3(K,
     *                 MAXNV,NV,VERTEX,
     *                 MAXNS,NS,SELV,OELEND,
     *                 MAXNPI,NPI,PINT,
     *                 MAXNPE,NPE,PEXT,
     *                 SALPHA,SBETA,SF,
     *                 LSOL,LVALID,EGEOM,
     *                 SPHI,SVEL,PIPHI,PEPHI, 
     *                 BIGMAT,INPVEC,SOLVEC,LO,LS,MS)



C Compute the sound pressure at the cavity surface points. Also compute
C  the velocity and intensity at the points for each type of boundary
C  condition and each related input function f and at each point.

            DO 690 ISP=1,NS
              PRESSURE=CIMAG*RHO*OMEGA*SPHI(ISP)
              SPRESS(ISP)=PRESSURE
              SINTY(ISP)=
     *         DBLE(CONJG(PRESSURE)*SVEL(ISP))/TWO
690         CONTINUE

         OPOWER=0.0D0
           DO 691 IS=1,OELEND
             OPOWER=OPOWER+SINTY(IS)*ELAREA(IS)
691        CONTINUE
         CPOWER=0.0D0
           DO 692 IS=OELEND+1,NS
             CPOWER=CPOWER+SINTY(IS)*ELAREA(IS)
692        CONTINUE

          DO 695 IPI=1,NPI
            IPRESS(IPI)=CIMAG*RHO*OMEGA*PIPHI(IPI)
695       CONTINUE

          DO 696 IPE=1,NPE
            EPRESS(IPE)=CIMAG*RHO*OMEGA*PEPHI(IPE)
696       CONTINUE

C Output the solutions
C  Open file for the output data
      OPEN(UNIT=20,FILE='BERIM3.OUT')

C  Formats for output
2800  FORMAT(1X,'Acoustic frequency = ',F8.2,'  Hz'/)
2810  FORMAT(1X,'Wavenumber = ',F8.2/)
2830  FORMAT(4X,'Acoustic properties at the boundary points'/)
2845  FORMAT(4X,'Sound pressure at the interior points',/)
2846  FORMAT(4X,'Sound pressure at the exterior points',/)
2850  FORMAT(5X,'index',7X,'Potential',19X,'Pressure',24X,
     * 'Velocity',17X,'Intensity'/)
2855  FORMAT(5X,'index',8X,'Potential',20X,'Pressure',13X,
     *'Magnitude',13X,'Phase'/)
2860  FORMAT(4X,I4,2X,E10.4,'+ ',E10.4,' i    ',
     * E10.4, '+ ',E10.4,' i    ',4X,
     * E10.4, '+ ',E10.4,7X,F10.4)
2910  FORMAT(4X,I4,2X,E10.4,'+ ',E10.4,' i',4X,
     * E10.4, '+ ',E10.4,' i    ',E10.4,' dB',7X,F10.4)

      WRITE(20,*) 'BERIM3: Computed solution to the acoustic properties'
      WRITE(20,*)
C   Output the acoustic frequency
        WRITE(20,*)
        WRITE(20,*)
        WRITE(20,*) 'Test problem '
        WRITE(20,*)
        WRITE(20,2800) FREQ
        WRITE(20,2810) K
        WRITE(20,*)
        WRITE(20,2830)
        WRITE(20,*)
        WRITE(20,2850)
        WRITE(20,*)
C      Loop(ISP) through the points on the boundary
        DO 2030 ISP=1,NSP
C       Output the sound pressure, velocity and intensity at each point
          WRITE(20,2860) ISP,DBLE(SPHI(ISP)),
     *     AIMAG(SPHI(ISP)),DBLE(SPRESS(ISP)),
     *     AIMAG(SPRESS(ISP)),DBLE(SVEL(ISP)),
     *     AIMAG(SVEL(ISP)),SINTY(ISP)
2030    CONTINUE
        WRITE(20,*)
        WRITE(20,*) 'Power measured on cavity surface = ',CPOWER
        WRITE(20,*) 'Power measured at opening = ',OPOWER
        WRITE(20,*)
        WRITE(20,2845)
        WRITE(20,2855)
C      Loop(IPI) through the points in the interior
        DO 2040 IPI=1,NPI
          PRESSURE=IPRESS(IPI)
C       Output the sound pressure, its magnitude(dB) and phase

          WRITE(20,2910) IPI,DBLE(PIPHI(IPI)),
     *     AIMAG(PIPHI(IPI)),
     *     DBLE(PRESSURE),AIMAG(PRESSURE),
     *     LOG10(ABS(PRESSURE)/PREREF)*20.0D0,
     *     ATAN2(AIMAG(PRESSURE),DBLE(PRESSURE))

2040    CONTINUE
        WRITE(20,*)

        WRITE(20,*)
        WRITE(20,2846)
        WRITE(20,2855)
C      Loop(IPE) through the points in the exterior
        DO 2050 IPE=1,NPE
          PRESSURE=EPRESS(IPE)
C       Output the sound pressure, its magnitude(dB) and phase

          WRITE(20,2910) IPE,DBLE(PEPHI(IPE)),
     *     AIMAG(PEPHI(IPE)),
     *     DBLE(PRESSURE),AIMAG(PRESSURE),
     *     LOG10(ABS(PRESSURE)/PREREF)*20.0D0,
     *     ATAN2(AIMAG(PRESSURE),DBLE(PRESSURE))

2050    CONTINUE
        WRITE(20,*)

      CLOSE(20)

      END
