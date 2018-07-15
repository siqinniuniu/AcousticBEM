C***************************************************************
C           Subroutine BERIM3 by Stephen Kirkup               
C***************************************************************
C
C  Copyright 2004- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/BERIM3.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************

C
C This subroutine computes the acoustics of an open cavity in three
C dimensions and of arbitrary shape. Since solutions are only computed
C for one harmonic at a time then this is equivalent to solving the 
C three-dimensional Helmholtz equation
C                  __ 2                2
C                  \/    {\phi}   +   k  {\phi}   =  0   
C
C for each wavenumber.
C
C The boundary (C) of the cavity and opening O are defined 
C (approximated) by a set of planar triangular elements. The domain of
C the equation is the cavity and the infinite domain exterior to the
C opening.
C
C The boundary condition is defined on the cavity surface and it may be
C Dirichlet, Robin or Neumann. It is assumed to have the following 
C general form
C
C           {\alpha}(q) {\phi}(q) + {\beta}(q) v(q) = f(q)
C    
C where {\phi}(q) is the solution at the point q on C, v(q) is the 
C derivative of {\phi} with respect to the outward normal to C at q and
C {\alpha}, {\beta} and f are complex-valued functions defined on C. The
C functions {\alpha} and {\beta} must be specified to define the nature
C of the boundary condition. Important examples are {\alpha}=1, 
C {\beta}=0 which is equivalent to a Dirichlet boundary condition and 
C {\alpha}=0, {\beta}=1 which is equivalent to a Neumann boundary 
C condition. The specification of f completes the definition of the 
C boundary condition.
C
C
C How to use the subroutine
C -------------------------
C
C The following diagram shows how the subroutine is to be used. 
C
C                                   ....................................
C                                   :                                  :
C                                   :                                  :
C      ----------------------       :     --------------------------   :
C      |                    |       :     |                        |   :
C      |   MAIN PROGRAM     |------->-----|      BERIM3            |   :
C      | (e.g. berim3_t.for)|       :     |                        |   :
C      |                    |       :     --------------------------   :
C      ----------------------       :                 |                :
C                                   :                 >                :
C                                   :                 |                :
C                                   :      ------------------------    :
C          Package ---------------->:      | subordinate routines |    :
C                                   :      ------------------------    :
C                                   :                                  :
C                                   :      (this file)                 :  
C                                   :..................................:
C                                            |                 |        \
C                                 _          >                 >         \
C                                            |                 |          \
C                                    ................   ................   ................
C                                    :   --------   :   :  --------    :   :     ------   :
C                                    :   | H3LC  |  :   :  | geom3D |  :   :    | CGLS |  :
C                                    :   --------   :   :  --------    :   :     ------   : 
C                                    : -------------:   : -------------:   : -------------:
C                                    : |subordinate|:   : |subordinate|:   : |subordinate|:
C                                    : | routines  |:   : | routines  |:   : | routines  |:
C                                    : -------------:   : -------------:   : -------------:
C                                    :              :   :              :   :              :
C                                    : (BERIM3.for) :   : (geom3d.for) :   : (cgls.for)   : 
C                                    :..............:   :..............:   :...............
C      
C
C The contents of the main program must be linked to  BERIM3, 
C  H3LC.FOR and CGLS.FOR.
C
C Method of solution
C ------------------
C 
C In the main program, the boundary and mouth of the cavity must be 
C  described as a set of elements. The elements are defined by three
C  indices (integers) which label a node or vertex on the boundary. 
C  The data structure VERTEX lists and enumerates the coordinates of 
C  the vertices, the data structure SELV defines each element of the
C  cavity by indicating the labels for the three nodes that are its
C  vertices and hence enumerates the cavity elements. Similarly OELV
C  defines and enumerates the elements over the opening.
C The cavity boundary solution points (the points on the boundary at which 
C  {\phi} (SPHI) and d {\phi}/dn (SVEL) are returned) are at the centres
C  of the elements. The boundary functions {\alpha} (SALPHA), {\beta} 
C  (SBETA) and f (SF) are also defined by their values at the centres
C  of the elements. 

C Normally a solution in the domain is required. By listing the 
C  coordinates of all the points interior to the cavity in PINT, the 
C  subroutine returns the value of {\phi} at these points in PIPHI. 
C  By listing the coordinates of all the points in the exterior, the
C  solution exterior to the opening can be obtained.
C
C
C Format and parameters of the subroutine
C ---------------------------------------
C
C The subroutine has the form
C
C      SUBROUTINE BERIM3(K,
C     *                 MAXNV,NV,VERTEX,
C     *                 MAXNSE,NSE,SELV,OELEND,
C     *                 MAXNPI,NPI,PINT,
C     *                 MAXNPE,NPE,PEXT,
C     *                 SALPHA,SBETA,SF,
C     *                 LSOL,LVALID,EGEOM,
C     *                 SPHI,SVEL,PIPHI,PEPHI, 
C     *                 AMAT,BMAT,LS,MS)

C The parameters to the subroutine
C ================================

C Wavenumber (input)
C real K: Must be positive.

C Geometry of the boundary S (input)
C integer MAXNV: The limit on the number of vertices of the polygon
C  that defines (approximates) S=C+O. MAXNV>=4.
C integer NV: The number of vertices on S=C+O. 4<=NV<=MAXNV.
C real VERTEX(MAXNV,3): The coordinates of the vertices. VERTEX(i,1),
C  VERTEX(i,2), VERTEX(i,3) are the x,y,z coordinates of the i-th 
C  vertex.
C integer MAXNSE: The limit on the number of elements describing C and O.
C  MAXNSE>=3.
C integer NSE: The number of elements describing C and O. 4<=NSE<=MAXNSE.
C integer SELV(MAXNSE,3): The indices of the three vertices defining
C  each element. The i-th element have vertices 
C  (VERTEX(SELV(i,1),1),VERTEX(SELV(i,1),2)),VERTEX(SELV(i,1),3)),
C  (VERTEX(SELV(i,2),1),VERTEX(SELV(i,2),2)),VERTEX(SELV(i,2),3)) and
C  (VERTEX(SELV(i,3),1),VERTEX(SELV(i,3),2)),VERTEX(SELV(i,3),3)).
C integer OELEND: The element where the opening ends. The elements
C  1..OLEND define the opening.


C Interior points at which the solution is to be observed (input)
C integer MAXNPI: Limit on the number of points interior to the
C  boundary. MAXNPI>=1.
C integer NPI: The number of interior points. 0<=NPI<=MAXNPI.
C real PINT(MAXNPI,3). The coordinates of the interior point.
C  PINT(i,1),PINT(i,2),PINT(i,3) are the x,y,z coordinates of the i-th
C  point.

C Exterior points at which the solution is to be observed (input)
C integer MAXNPE: Limit on the number of points interior to the
C  boundary. MAXNPE>=1.
C integer NPE: The number of interior points. 0<=NPE<=MAXNPE.
C real PEXT(MAXNPE,3). The coordinates of the interior point.
C  PEXT(i,1),PEXT(i,2),PEXT(i,3) are the x,y,z coordinates of the i-th
C  point.

C The boundary condition ({\alpha} phi + {\beta} v = f) (input)
C  defined only on the cavity boundary.
C complex SALPHA(MAXNSE): The values of {\alpha} at the
C  centres of the elements. Set SALPHA(i) for OLEND+1<=i<=NSE.
C complex SBETA(MAXNSE): The values of {\beta} at the 
C  centres of the elements. Set SBETA(i) for OLEND+1<=i<=NSE.
C complex SF(MAXNSE): The values of "f" at the centres
C  of the elements.  Set SF(i) for OLEND+1<=i<=NSE.

C Validation and control parameters (input) 
C logical LSOL: A switch to control whether the particular solution is
C  required.
C logical LVALID: A switch to enable the choice of checking of 
C  subroutine parameters.
C real EGEOM: The maximum absolute error in the parameters that
C  describe the geometry.

C Solution (output)
C complex SPHI(MAXNSE): The velocity potential ({\phi}) at the 
C  centres of the cavity boundary elements. 
C  Potentials on the opening SPHI(1..OLEND)
C  Potentials on the cavity SPHI(OLEND+1..NSE)
C complex SVEL(MAXNSE): The velocity (v or d{\phi}/dn where n is
C  the outward normal to the boundary) at the centres of the cavity
C  boundary elements.
C  Velocities on the opening SVEL(1..OLEND)
C  Velocities on the cavity SVEL(OLEND+1..NSE)
C complex PIPHI(MAXNPI): The velocity potential ({\phi}) at the 
C  interior points.
C complex PEPHI(MAXNPI): The velocity potential ({\phi}) at the 
C  exterior points.

C Working space
C complex BIGMAT(2*MAXNSE,2*MAXNSE)
C complex AMAT(MAXNSE,MAXNSE)
C complex BMAT(MAXNSE,MAXNSE)
C complex LS(MAXNPI,MAXNSE)
C complex MS(MAXNPI,MAXNSE)

C Notes on the geometric parameters
C ---------------------------------
C (1) Each of the vertices listed in VERTEX must be distinct points
C  with respect to EGEOM.
C (2) The boundary S=C+O must be complete and closed. 
C (3) The indices of the nodes listed in SELV must be such that they
C  are ordered counter-clockwise around the boundary, when viewed
C  from just outside the cavity boundary at each cavity element and
C  on the open side of each mouth element
C (4) The largest element must be no more than 10x the area of the
C  smallest element.

C Notes on the interior/exterior points 
C -------------------------------------
C (1) The points in PINT should lie within the cavity, as defined
C  by the parameters VERTEX and SELV. 
C (2) The points in PEXT should lie beyond the opening, as defined
C  by the parameters VERTEX and SELV(OELSTART..NSE). 

C Notes on the boundary condition
C -------------------------------
C (1) For each i=OELEND+1..NSE, it must not be the case that both of 
C  SALPHA(i)and SBETA(i) are zero

C External modules in external files
C ==================================
C subroutine H3LC: Returns the individual discrete Helmholtz integral
C  operators. (in file H3LC.FOR)
C subroutine CGLS: Solves a general linear system of equations. 
C  (in file CGLS.FOR)
C
C External modules provided in the package (this file)
C ====================================================
C subroutine GLT7: Returns the points and weights of the 7-point Gauss-
C  Legendre quadrature rule on the standard triangle.
C real function FNSQRT(X): real X : Returns the square root of X.
C complex function FNEXP(Z): complex Z : Returns the complex exponential
C   of X.


C The subroutine

      SUBROUTINE BERIM3(K,
     *                 MAXNV,NV,VERTEX,
     *                 MAXNSE,NSE,SELV,OELEND,
     *                 MAXNPI,NPI,PINT,
     *                 MAXNPE,NPE,PEXT,
     *                 SALPHA,SBETA,SF,
     *                 LSOL,LVALID,EGEOM,
     *                 SPHI,SVEL,PIPHI,PEPHI, 
     *                 BIGMAT,INPVEC,SOLVEC,LO,LS,MS)

      PARAMETER (MAXNQ=100)

C  Wavenumber
      REAL*8     K

C  Boundary geometry
C   Limit on the number of vertices on S=C+O
      INTEGER    MAXNV
C   The number of vertices on  on S=C+O
      INTEGER    NV
C   The coordinates of the vertices on S=C+O
      REAL*8     VERTEX(MAXNV,3)
C   Limit on number of elements 
      INTEGER    MAXNSE
C   The number of elements describing S=C+O
      INTEGER    NSE
C   The indices of the vertices describing each element on S=C+O
      INTEGER    SELV(MAXNSE,3)
C   The index of the FINAL element on the opening
      INTEGER    OELEND
      
C  Interior cavity points at which the solution is to be observed
C   Limit on the number of points interior to the boundary where 
C    solution is sought
      INTEGER    MAXNPI
C   The number of interior points
      INTEGER    NPI
C   Coordinates of the interior points
      REAL*8     PINT(MAXNPI,3)

C  Exterior points at which the solution is to be observed
C   Limit on the number of points interior to the opening where 
C    solution is sought
      INTEGER    MAXNPE
C   The number of interior points
      INTEGER    NPE
C   Coordinates of the interior points
      REAL*8     PEXT(MAXNPI,3)

C  The boundary condition is such that {\alpha} {\phi} + {\beta} v = f
C  where alpha, beta and f are complex valued functions over C.
C  These functions should only have values for indices OLEND+1..NSE
C  The functions are set values at the collocation points.
C   function alpha
      COMPLEX*16 SALPHA(MAXNSE)
C   function beta
      COMPLEX*16 SBETA(MAXNSE)
C   function f
      COMPLEX*16 SF(MAXNSE)

C  Validation and control parameters
      LOGICAL    LSOL
      LOGICAL    LVALID
      REAL*8     EGEOM

C  Solution 
C   function phi on S=C+O
      COMPLEX*16 SPHI(MAXNSE)
C   function vel on S=C+O
      COMPLEX*16 SVEL(MAXNSE)
C   interior solution
      COMPLEX*16 PIPHI(MAXNPI)
C   exterior solution
      COMPLEX*16 PEPHI(MAXNPE)

C  Working space
C   For BERIM3 routine
      COMPLEX*16 BIGMAT(2*MAXNSE,2*MAXNSE)
      COMPLEX*16 INPVEC(2*MAXNSE)
      COMPLEX*16 SOLVEC(2*MAXNSE)
      COMPLEX*16 LO(MAXNPE,MAXNSE)
      COMPLEX*16 LS(MAXNPI,MAXNSE)
      COMPLEX*16 MS(MAXNPI,MAXNSE)

c  External function
      REAL*8     DIST3
      REAL*8     AREA

C  Constants
C   Real scalars: 0, 1, 2, half, pi
      REAL*8 ZERO,ONE,TWO,THREE,HALF,THIRD,PI
C   Complex scalars: (0,0), (1,0), (0,1)
      COMPLEX*16 CZERO,CONE,CIMAG

C  Wavenumber in complex form
      COMPLEX*16 CK

C  Geometrical description of the boundary
C   Elements counter
      INTEGER    ISE,JSE
C   The points interior to the boundary where the solution is sought 
      INTEGER    IPI
C   Parameters for H3LC
      REAL*8     P(3),PA(3),PB(3),PC(3),QA(3),QB(3),QC(3),VECP(3)
      REAL*8     NORMP(3),NORMQ(3)
      LOGICAL    LPONEL


C  Quadrature rule information
C   [Note that in this program two quadrature rules are used: one for
C    the case when the point P lies on the element (LPONEL=.TRUE.) and
C    one for the case when P does not lie on the element. In general,
C    it is more efficient to define a larger set of quadrature rules
C    so that a particular rule can be selected for any given point P 
C    and element QA-QB. For example using more quadrature points when
C    the element is large, less when the element is small, more when
C    the element is close to P, less when it is far from P.]
C   Quadrature rule used when LPONEL=.TRUE.
C    Number of quadrature points
      INTEGER    NQON
C    x-Abscissae of the actual quadrature rule
      REAL*8     XQON(MAXNQ)
C    y-Abscissae of the actual quadrature rule
      REAL*8     YQON(MAXNQ)
C    Weights of the actual quadrature rule
      REAL*8     WQON(MAXNQ)
C   Quadrature rule used when LPONEL=.FALSE.
C    Number of quadrature points
      INTEGER    NQOFF
C    x-Abscissae of the actual quadrature rule
      REAL*8     XQOFF(MAXNQ)
C    y-Abscissae of the actual quadrature rule
      REAL*8     YQOFF(MAXNQ)
C    Weights of the actual quadrature rule
      REAL*8     WQOFF(MAXNQ)
C   Quadrature rule parameters for H3LC
C    Actual number of quadrature points
      INTEGER    NQ
C    x-Abscissae of the actual quadrature rule
      REAL*8     XQ(MAXNQ)
C    y-Abscissae of the actual quadrature rule
      REAL*8     YQ(MAXNQ)
C    Weights of the actual quadrature rule
      REAL*8     WQ(MAXNQ)
C   Counter through the quadrature points
      INTEGER    IQ

C  Validation and control parameters for subroutine H3LC
      LOGICAL    LVAL
      REAL*8     EK
      REAL*8     EQRULE
      LOGICAL    LFAIL1
      LOGICAL    LLK
      LOGICAL    LMK
      LOGICAL    LMKT
      LOGICAL    LNK

C  Parameters for subroutine H3LC. 
      COMPLEX*16 DISLK
      COMPLEX*16 DISMK
      COMPLEX*16 DISMKT
      COMPLEX*16 DISNK

C  Other variables
C   Error flag
      LOGICAL    LERROR
C   Failure flag
      LOGICAL    LFAIL
C   Accumulation of solution {\phi}
      COMPLEX*16 SUMPHI
C   Maximum,minimum sizes of elements
      REAL*8     SIZMAX,SIZMIN,SIZE
C   The `diameter' of the boundary or the maximum distance between any
C    three vertices
      REAL*8         DIAM
      REAL*8         SUMMK

      LOGICAL  NEUMANN

      COMPLEX*16 TEMP


    
C INITIALISATION
C --------------

C Set constants
      ZERO=0.0D0
      ONE=1.0D0
      TWO=2.0D0
      THREE=3.0D0
      HALF=ONE/TWO
      THIRD=ONE/THREE
      PI=4.0D0*ATAN(ONE)
      CZERO=CMPLX(ZERO,ZERO)
      CONE=CMPLX(ONE,ZERO)
      CIMAG=CMPLX(ZERO,ONE)

      IF (LVALID) THEN

C Validation of main paramters
        LERROR=.FALSE.
        IF (K.LT.ZERO) THEN
          WRITE(*,*) 'K = ',K
          WRITE(*,*) 'ERROR(BERIM3) - K must be positive'
          LERROR=.TRUE.
        END IF
        IF (MAXNV.LT.4) THEN
          WRITE(*,*) 'MAXNV = ',MAXNV
          WRITE(*,*) 'ERROR(BERIM3) - must have MAXNV>=4'
          LERROR=.TRUE.
        END IF
        IF (NV.LT.4.OR.NV.GT.MAXNV) THEN
          WRITE(*,*) 'NV = ',NV
          WRITE(*,*) 'ERROR(BERIM3) - must have 4<=NV<=MAXNV'
          LERROR=.TRUE.
        END IF
        IF (MAXNSE.LT.3) THEN
          WRITE(*,*) 'MAXNSE = ',MAXNSE
          WRITE(*,*) 'ERROR(BERIM3) - must have MAXNSE>=3'
          LERROR=.TRUE.
        END IF
        IF (NSE.LT.3.OR.NSE.GT.MAXNSE) THEN
          WRITE(*,*) 'NSE = ',NSE
          WRITE(*,*) 'ERROR(BERIM3) - must have 3<=NSE<=MAXNSE'
          LERROR=.TRUE.
        END IF
        IF (MAXNPI.LT.1) THEN
          WRITE(*,*) 'MAXNPI = ',MAXNPI
          WRITE(*,*) 'ERROR(BERIM3) - must have MAXNPI>=1'
          LERROR=.TRUE.
        END IF
        IF (MAXNPE.LT.1) THEN
          WRITE(*,*) 'MAXNPE = ',MAXNPE
          WRITE(*,*) 'ERROR(BERIM3) - must have MAXNPE>=1'
          LERROR=.TRUE.
        END IF
        IF (NPI.LT.0.OR.NPI.GT.MAXNPI) THEN
          WRITE(*,*) 'NPI = ',NPI
          WRITE(*,*) 'ERROR(BERIM3) - must have 0<=NPI<=MAXNPI'
          LERROR=.TRUE.
        END IF
        IF (NPE.LT.0.OR.NPE.GT.MAXNPE) THEN
          WRITE(*,*) 'NPE = ',NPE
          WRITE(*,*) 'ERROR(BERIM3) - must have 0<=NPE<=MAXNPE'
          LERROR=.TRUE.
        END IF
        IF (OELEND.LT.0.OR.OELEND.GT.NSE)THEN
          WRITE(*,*) 'OELEND = ',OELEND
          WRITE(*,*) 'ERROR(BERIM3) - must have 0<=OELEND<=NSE'
          LERROR=.TRUE.
        END IF
        IF (EGEOM.LE.ZERO) THEN
          WRITE(*,*) 'NPI = ',NPI
          WRITE(*,*) 'ERROR(BERIM3) - EGEOM must be positive'
          LERROR=.TRUE.
        END IF
        IF (LERROR) THEN
          LFAIL=.TRUE.
          WRITE(*,*)
          WRITE(*,*) 'Error(s) found in the main parameters of BERIM3'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF

        LERROR=.FALSE.
        DO 900 ISE=OELEND+1,NSE
          IF (ABS(SALPHA(ISE)).LT.EGEOM) THEN
            IF (ABS(SBETA(ISE)).LT.EGEOM) THEN
              LERROR=.TRUE.
            END IF
          END IF
900     CONTINUE
        IF (LERROR) THEN
          LFAIL=.TRUE.
          WRITE(*,*)
          WRITE(*,*) 'ERROR(BERIM3) - SALPHA and SBETA should not '
          WRITE(*,*) ' both be zero for any element on the cavity'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF

      NEUMANN=.TRUE.
      DO 910 ISE=OELEND+1,NSE
        IF (ABS(SALPHA(ISE)).GT.EGEOM) NEUMANN=.FALSE.
910   CONTINUE

C Find the diameter DIAM of the boundary
      DIAM=0.0
      DO 100 IV=1,NV-1
        PA(1)=VERTEX(IV,1)
        PA(2)=VERTEX(IV,2)
        PA(3)=VERTEX(IV,3)
        DO 110 JV=IV+1,NV
          PB(1)=VERTEX(JV,1)
          PB(2)=VERTEX(JV,2)
          PB(3)=VERTEX(JV,3)
          DIAM=MAX(DIAM,DIST3(PA,PB))
110     CONTINUE
100   CONTINUE

      IF (LVALID) THEN
        LERROR=.FALSE.
C Check that the boundary defined by SELV is complete and closed
        DO 120 ISE=1,NSE
120     CONTINUE


C Check that EGEOM is not too large
        IF (EGEOM.GT.DIAM/100.0D0) THEN
          WRITE(*,*) 'EGEOM = ',EGEOM
          WRITE(*,*) 'ERROR(BERIM3) - EGEOM is set too large'
          LERROR=.TRUE.
        END IF
        IF (LERROR) THEN
          LFAIL=.TRUE.
          WRITE(*,*)
          WRITE(*,*) 'Error in boundary geometry or EGEOM'
          WRITE(*,*) 'Execution terminated'
        END IF
      END IF                  

      IF (LVALID) THEN
C Check that the vertices are distinct with respect to EGEOM
        LERROR=.FALSE.
        DO 130 IV=1,NV-1
          PA(1)=VERTEX(IV,1)
          PA(2)=VERTEX(IV,2)
          PA(3)=VERTEX(IV,3)
          DO 140 JV=IV+1,NV
            PB(1)=VERTEX(JV,1)
            PB(2)=VERTEX(JV,2)
            PB(3)=VERTEX(JV,3)
            IF (ABS(PA(1)-PB(1)).LT.EGEOM) THEN
              IF (ABS(PA(2)-PB(2)).LT.EGEOM) THEN
                IF (ABS(PA(3)-PB(3)).LT.EGEOM) THEN
                  WRITE(*,*) 'Vertices ',IV,JV,' are not distinct'
                  LERROR=.TRUE.
                END IF
              END IF
            END IF
140       CONTINUE
130     CONTINUE
        IF (LERROR) THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR(BERIM3) - Vertices (see above) coincide'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF          

C Check that vertices on the opening are co-planar
      LERROR=.FALSE.
C  Find the normal to the first element
      PA(1)=VERTEX(SELV(1,1),1)
      PA(2)=VERTEX(SELV(1,1),2)
      PA(3)=VERTEX(SELV(1,1),3)
      PB(1)=VERTEX(SELV(1,2),1)
      PB(2)=VERTEX(SELV(1,2),2)
      PB(3)=VERTEX(SELV(1,2),3)
      PC(1)=VERTEX(SELV(1,3),1)
      PC(2)=VERTEX(SELV(1,3),2)
      PC(3)=VERTEX(SELV(1,3),3)
      CALL NORM3(PA,PB,PC,NORMP)
C  Find the normals to each of the other elements
      DO 125 ISE=2,OELEND
        QA(1)=VERTEX(SELV(ISE,1),1)
        QA(2)=VERTEX(SELV(ISE,1),2)
        QA(3)=VERTEX(SELV(ISE,1),3)
        QB(1)=VERTEX(SELV(ISE,2),1)
        QB(2)=VERTEX(SELV(ISE,2),2)
        QB(3)=VERTEX(SELV(ISE,2),3)
        QC(1)=VERTEX(SELV(ISE,3),1)
        QC(2)=VERTEX(SELV(ISE,3),2)
        QC(3)=VERTEX(SELV(ISE,3),3)
        CALL NORM3(QA,QB,QC,NORMQ)
        IF (DIST3(NORMP,NORMQ).GT.EGEOM) THEN
          LERROR=.TRUE.
        END IF
125   CONTINUE
      IF (LERROR) THEN
          WRITE(*,*) 'WARNING(RIM3) - Panels on the opening '
          WRITE(*,*) ' are not co-planar'
      END IF


C Check that the elements are not of disproportionate sizes
      IF (LVALID) THEN
        SIZMAX=ZERO
        SIZMIN=DIAM**2
        DO 150 ISE=1,NSE
          QA(1)=VERTEX(SELV(ISE,1),1)
          QA(2)=VERTEX(SELV(ISE,1),2)
          QA(3)=VERTEX(SELV(ISE,1),3)
          QB(1)=VERTEX(SELV(ISE,2),1)
          QB(2)=VERTEX(SELV(ISE,2),2)
          QB(3)=VERTEX(SELV(ISE,2),3)
          QC(1)=VERTEX(SELV(ISE,3),1)
          QC(2)=VERTEX(SELV(ISE,3),2)
          QC(3)=VERTEX(SELV(ISE,3),3)
          SIZE=AREA(QA,QB,QC)
          SIZMAX=MAX(SIZMAX,SIZE)
          SIZMIN=MIN(SIZMIN,SIZE)
150     CONTINUE
        IF (SIZMAX.GT.10.0D0*SIZMIN) THEN
          WRITE(*,*) 'WARNING(BERIM3) - Elements of disproportionate'
          WRITE(*,*) ' sizes'
        END IF
      END IF          
          

C Validation of the surface functions
      IF (LVALID.AND.LSOL) THEN
        LERROR=.FALSE.
        DO 170 ISE=OELEND+1,NSE
          IF (MAX(ABS(SALPHA(ISE)),ABS(SBETA(ISE))).LT.1.0D-6) 
     *     LERROR=.TRUE.
170     CONTINUE
        IF (LERROR) THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR(BERIM3) - at most one of SALPHA(i),SBETA(i)'
          WRITE(*,*) ' may be zero for all i'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF
        

C Set the wavenumber in complex form
      CK=CMPLX(K,ZERO)


C Set up validation and control parameters
C  Switch off the validation of H3LC
      LVAL=.FALSE.
C  Set EK
      EK=1.0D-6
C  Set EQRULE
      EQRULE=1.0D-6

C Set up the quadrature rule(s).
C  Set up quadrature rule for the case when P is not on the element.
C   Set up 8 point Gauss-Legendre rules
      CALL GLT7(MAXNQ,NQOFF,WQOFF,XQOFF,YQOFF)
C  Set up quadrature rule for the case when P is on the element.

C   Set up quadrature rule data. If LPONEL is false then use the standard
C    Gaussian quadrature rule above. If LPONEL is true then a
C    quadrature rule with 3 times as many points is used, this is made
C    up from three standard quadrature rules with the quadrature points
C    translated to the three triangles that each have the cetroid and two
C    of the original vertices as its vertices.
      NQON=3*NQOFF
      DO 330 IQ=1,NQOFF
        XQON(IQ)=XQOFF(IQ)*THIRD+YQOFF(IQ)
        YQON(IQ)=XQOFF(IQ)*THIRD
        WQON(IQ)=WQOFF(IQ)/THREE
        XQON(IQ+NQOFF)=XQOFF(IQ)*THIRD
        YQON(IQ+NQOFF)=XQOFF(IQ)*THIRD+YQOFF(IQ)
        WQON(IQ+NQOFF)=WQOFF(IQ)/THREE
        XQON(IQ+2*NQOFF)=THIRD*(ONE+TWO*XQOFF(IQ)-YQOFF(IQ))
        YQON(IQ+2*NQOFF)=THIRD*(ONE-XQOFF(IQ)+TWO*YQOFF(IQ))
        WQON(IQ+2*NQOFF)=WQOFF(IQ)/THREE
330   CONTINUE


C Validation that the surface is closed
      IF (LVALID) THEN
        PA(1)=VERTEX(SELV(1,1),1)
        PA(2)=VERTEX(SELV(1,1),2)
        PA(3)=VERTEX(SELV(1,1),3)
        PB(1)=VERTEX(SELV(1,2),1)
        PB(2)=VERTEX(SELV(1,2),2)
        PB(3)=VERTEX(SELV(1,2),3)
        PC(1)=VERTEX(SELV(1,3),1)
        PC(2)=VERTEX(SELV(1,3),2)
        PC(3)=VERTEX(SELV(1,3),3)
        P(1)=(PA(1)+PB(1)+PC(1))/THREE
        P(2)=(PA(2)+PB(2)+PC(2))/THREE
        P(3)=(PA(3)+PB(3)+PC(3))/THREE
        VECP(1)=0.0D0
        VECP(2)=0.0D0
        VECP(3)=1.0D0
        SUMMK=0.0D0
        DO 180 JSE=1,NSE
C  Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QA(3)=VERTEX(SELV(JSE,1),3)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
          QB(3)=VERTEX(SELV(JSE,2),3)
          QC(1)=VERTEX(SELV(JSE,3),1)
          QC(2)=VERTEX(SELV(JSE,3),2)
          QC(3)=VERTEX(SELV(JSE,3),3)
C     Set LPONEL
          LPONEL=(JSE.EQ.1)

C     Only the Mk operator is required. Set LMK true, 
C      LLK,LMKT,LNK false. 
          LLK=.FALSE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H3LC.
          CALL H3LC(CZERO,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQON,XQON,YQON,WQON,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

          
          SUMMK=SUMMK+DISMK
180     CONTINUE
        IF (ABS(SUMMK-0.5D0).LT.0.1) THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR(BERIM3) - in geometry'
          WRITE(*,*) ' The boundary (C+O) could be oriented wrongly'
          WRITE(*,*) '  On the outer boundary arrange panels'
     *     // 'in clockwise order'
          WRITE(*,*) '  If there are inner boundaries arrange the'
     *     // 'panels in anticlockwise order'
          STOP
        END IF
        IF (ABS(SUMMK+0.5D0).GT.0.1) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(BERIM3) - in geometry'
          WRITE(*,*) ' The boundary (C+O) panels may be arranged'
          WRITE(*,*) '  incorrectly'
        END IF
      END IF  


C Validation that the points in PINT are interior points
      IF (LVALID) THEN
        DO IPI=1,NPI
        P(1)=PINT(IPI,1)
        P(2)=PINT(IPI,2)
        P(3)=PINT(IPI,3)
        VECP(1)=0.0D0
        VECP(2)=0.0D0
        VECP(3)=1.0D0
        SUMMK=0.0D0
        DO 210 JSE=1,NSE
C  Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QA(3)=VERTEX(SELV(JSE,1),3)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
          QB(3)=VERTEX(SELV(JSE,2),3)
          QC(1)=VERTEX(SELV(JSE,3),1)
          QC(2)=VERTEX(SELV(JSE,3),2)
          QC(3)=VERTEX(SELV(JSE,3),3)
C     Set LPONEL
          LPONEL=.FALSE.

C     Only the Mk operator is required. Set  LMK true, 
C      LLK,LMKT,LNK false. 
          LLK=.FALSE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H3LC.
          CALL H3LC(CZERO,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQON,XQON,YQON,WQON,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

          
          SUMMK=SUMMK+DISMK
210     CONTINUE

        IF (ABS(SUMMK+1.0D0).GT.0.25) THEN
          WRITE(*,*) 
          WRITE(*,*) 'WARNING(IBEM2) - The observation point'
          WRITE(*,*) ' (',P(1),',',P(2),',',P(3),')'
          WRITE(*,*) ' may not be interior to the cavity'
        END IF
      END DO
      END IF


C  Compute the discrete Lk, Mk, Mkt and Nk matrices
C   Loop(ISE) through the points on the boundary (C+O)
      DO 510 ISE=1,NSE
C    Set P
        PA(1)=VERTEX(SELV(ISE,1),1)
        PA(2)=VERTEX(SELV(ISE,1),2)
        PA(3)=VERTEX(SELV(ISE,1),3)
        PB(1)=VERTEX(SELV(ISE,2),1)
        PB(2)=VERTEX(SELV(ISE,2),2)
        PB(3)=VERTEX(SELV(ISE,2),3)
        PC(1)=VERTEX(SELV(ISE,3),1)
        PC(2)=VERTEX(SELV(ISE,3),2)
        PC(3)=VERTEX(SELV(ISE,3),3)
        P(1)=(PA(1)+PB(1)+PC(1))/THREE
        P(2)=(PA(2)+PB(2)+PC(2))/THREE
        P(3)=(PA(3)+PB(3)+PC(3))/THREE
C    Set VECP to the normal on the boundary (C+O) of the element at P
        CALL NORM3(PA,PB,PC,VECP)
C    Loop(ISE) through the elements
        DO 520 JSE=1,NSE
C     Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QA(3)=VERTEX(SELV(JSE,1),3)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
          QB(3)=VERTEX(SELV(JSE,2),3)
          QC(1)=VERTEX(SELV(JSE,3),1)
          QC(2)=VERTEX(SELV(JSE,3),2)
          QC(3)=VERTEX(SELV(JSE,3),3)

C     Set LPONEL
          IF (ISE.EQ.JSE) THEN
            LPONEL=.TRUE.
          ELSE
            LPONEL=.FALSE.
          END IF

C     Select quadrature rule for H3LC
C   :  Select the quadrature rule XQON-WQON in the case when the
C   :   point p lies on the element, otherwise select XQOFF-WQOFF
C      [Note that the overall method would benefit from selecting from
C       a wider set of quadrature rules, and an appropriate method
C       of selection]
          IF (LPONEL) THEN
            NQ=NQON
            DO 600 IQ=1,NQ
              XQ(IQ)=XQON(IQ)
              YQ(IQ)=YQON(IQ)
              WQ(IQ)=WQON(IQ)
600         CONTINUE
          ELSE 
            NQ=NQOFF
            DO 610 IQ=1,NQ
              XQ(IQ)=XQOFF(IQ)
              YQ(IQ)=YQOFF(IQ)
              WQ(IQ)=WQOFF(IQ)
610         CONTINUE
          END IF

C     All operators are required
          LLK=.TRUE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H3LC.
          CALL H3LC(CK,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQ,XQ,YQ,WQ,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

          BIGMAT(ISE,NSE+JSE)=DISLK
          BIGMAT(ISE,JSE)=-DISMK
          BIGMAT(NSE+ISE,JSE)=CZERO
          BIGMAT(NSE+ISE,NSE+JSE)=CZERO
          IF (ISE.LE.OELEND.AND.JSE.LE.OELEND) THEN
            BIGMAT(NSE+ISE,NSE+JSE)=2*DISLK
          END IF
         
C    Close loop(JSE) 
520     CONTINUE

        BIGMAT(ISE,ISE)=BIGMAT(ISE,ISE)-HALF
        INPVEC(ISE)=CZERO
        IF (ISE.GT.OELEND) THEN
          BIGMAT(NSE+ISE,ISE)=SALPHA(ISE)
          BIGMAT(NSE+ISE,NSE+ISE)=SBETA(ISE)
          INPVEC(NSE+ISE)=SF(ISE)
        ELSE
          BIGMAT(NSE+ISE,ISE)=CONE
          INPVEC(NSE+ISE)=CZERO
        END IF
        
            
C   Close loop(ISE) 
510   CONTINUE

      IF (LSOL) THEN 
       IF (NEUMANN) THEN
         DO 710 ISE=1,NSE
           DO 720 JSE=1,OELEND
             TEMP=CZERO
             DO 730 KSE=1,OELEND
               TEMP=TEMP+BIGMAT(ISE,KSE)*BIGMAT(NSE+KSE,NSE+JSE)
730          CONTINUE
             BIGMAT(NSE+ISE,JSE)=-TEMP
720        CONTINUE
710      CONTINUE
         DO 740 ISE=1,NSE
           DO 750 JSE=1,OELEND
             BIGMAT(ISE,JSE)=BIGMAT(NSE+ISE,JSE)+BIGMAT(ISE,NSE+JSE)
750        CONTINUE
740      CONTINUE
         DO 755 ISE=1,NSE
           INPVEC(ISE)=CZERO
           DO 760 JSE=OELEND+1,NSE
             INPVEC(ISE)=INPVEC(ISE)-
     *        BIGMAT(ISE,NSE+JSE)*SF(JSE)/SBETA(JSE)
760        CONTINUE
755      CONTINUE
         CALL CLINSL(2*MAXNSE,NSE,BIGMAT,INPVEC,SOLVEC)
         DO 770 ISE=1,OELEND
           SPHI(ISE)=CZERO
             DO 771 JSE=1,OELEND
               SPHI(ISE)=SPHI(ISE)-BIGMAT(NSE+ISE,NSE+JSE)*SOLVEC(ISE)
771          CONTINUE
           SVEL(ISE)=SOLVEC(ISE)
770      CONTINUE
         DO 780 ISE=OELEND+1,NSE
           SPHI(ISE)=SOLVEC(ISE)
           SVEL(ISE)=SF(ISE)
780      CONTINUE

               
       ELSE
         CALL CLINSL(2*MAXNSE,2*NSE,BIGMAT,INPVEC,SOLVEC)
         DO 700 ISE=1,NSE
           SPHI(ISE)=SOLVEC(ISE)
           SVEL(ISE)=SOLVEC(NSE+ISE)
700      CONTINUE
       END IF
      END IF
        

C  SOLUTION IN THE CAVITY

C   Compute sound pressures at the selected cavity points.
C    Loop through the the points in the cavity
      DO 800 IPI=1,NPI
C    Set P
        P(1)=PINT(IPI,1)
        P(2)=PINT(IPI,2)
        P(3)=PINT(IPI,3)
C    Set VECP, this is arbitrary as the velocity/intensity at P
C     is not sought.
        VECP(1)=ONE
        VECP(2)=ZERO
        VECP(3)=ZERO

C    Initialise SUMPHI to zero
        SUMPHI=CZERO

C    Loop(ISE) through the elements
        DO 850 JSE=1,NSE
C     Compute the discrete Lk and Mk integral operators. 
            
C     Set QA and QB, the coordinates of the edges of the JSEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QA(3)=VERTEX(SELV(JSE,1),3)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
          QB(3)=VERTEX(SELV(JSE,2),3)
          QC(1)=VERTEX(SELV(JSE,3),1)
          QC(2)=VERTEX(SELV(JSE,3),2)
          QC(3)=VERTEX(SELV(JSE,3),3)

C     All the points do not lie on the boundary (C+O) hence LPONEL=.FALSE.
          LPONEL=.FALSE.              

C     Only Lk, Mk operators are required. Set LLK,LMK true, 
C      LMKT,LNK false. 
          LLK=.TRUE.
          LMK=.TRUE.
          LMKT=.FALSE.
          LNK=.FALSE.
                
C     Call H3LC.
          CALL H3LC(CK,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQOFF,XQOFF,YQOFF,WQOFF,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

C     Accumulate phi 
          IF (LSOL) SUMPHI=SUMPHI+DISLK*SVEL(JSE)-DISMK*SPHI(JSE)

          IF (.NOT.LSOL) THEN
            LS(IPI,JSE)=DISLK
            MS(IPI,JSE)=DISMK
          END IF

C      Close loop (JSE) through the elements
850     CONTINUE

        PIPHI(IPI)=SUMPHI

C     Close loop(IPI) through the cavity points
800   CONTINUE

C  SOLUTION IN THE EXTERIOR

C   Compute sound pressures at the selected exterior points.
C    Loop through the the points in the exterior region
      DO 940 IPE=1,NPE
C    Set P
        P(1)=PEXT(IPE,1)
        P(2)=PEXT(IPE,2)
        P(3)=PEXT(IPE,3)
C    Set VECP, this is arbitrary as the velocity/intensity at P
C     is not sought.
        VECP(1)=ONE
        VECP(2)=ZERO
        VECP(3)=ZERO

C    Initialise SUMPHI to zero
        SUMPHI=CZERO

C    Loop(ISE) through the elements
        DO 950 JSE=1,OELEND
C     Compute the discrete Lk and Mk integral operators. 
            
C     Set QA and QB, the coordinates of the edges of the ISEth element
          QA(1)=VERTEX(SELV(JSE,1),1)
          QA(2)=VERTEX(SELV(JSE,1),2)
          QA(3)=VERTEX(SELV(JSE,1),3)
          QB(1)=VERTEX(SELV(JSE,2),1)
          QB(2)=VERTEX(SELV(JSE,2),2)
          QB(3)=VERTEX(SELV(JSE,2),3)
          QC(1)=VERTEX(SELV(JSE,3),1)
          QC(2)=VERTEX(SELV(JSE,3),2)
          QC(3)=VERTEX(SELV(JSE,3),3)

C     All the points do not lie on the plate hence LPONEL=.FALSE.
          LPONEL=.FALSE.              

C     Only Lk, Mk operators are required. Set LLK,LMK true, 
C      LMKT,LNK false. 
          LLK=.TRUE.
          LMK=.FALSE.
          LMKT=.FALSE.
          LNK=.FALSE.
                

C     Call H3LC.
          CALL H3LC(CK,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQOFF,XQOFF,YQOFF,WQOFF,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

C     Accumulate phi 
          SUMPHI=SUMPHI-2.0D0*DISLK*SVEL(JSE)
          IF (LSOL) LO(IPE,JSE)=DISLK

C      Close loop (JSE) through the elements
950     CONTINUE

        PEPHI(IPE)=SUMPHI

C     Close loop(IPE) through the exterior points
940   CONTINUE



      END


C ----------------------------------------------------------------------

C Subordinate routines for BERIM3
C ==============================

C ----------------------------------------------------------------------
C           Subroutine GLT7.FOR by www.numerical-methods.com           |
C ----------------------------------------------------------------------
C
C Subroutine GLT7 assigns the weights and points of a 7 point Gaussian
C quadrature rule defined on the standard triangle.
C
C SUBROUTINE GLT7(MAXNQ, NQ, WQ, XQ, YQ)
C integer  maxnq: the maximimum number of weights/points
C integer     nq: the number of weights/points
C real        wq: the weights
C real        xq: the x-coordinates of the points
C real        yq: the y-coordinates of the points
C
C Source of the code: http://www.numerical-methods.com/fortran/GLT7.FOR
C Source of the user-guide: http://www.numerical-methods.com/fortran/
C  glt7.htm
C
C Licence: This is 'open source'; the software may be used and applied
C  within other systems as long as its provenance is appropriately
C  acknowledged. See the GNU Licence http://www.gnu.org/licenses/lgpl.txt
C  for more information or contact webmaster@numerical-methods.com


      SUBROUTINE GLT7(MAXNQ,NQ,WQ,XQ,YQ)
      INTEGER MAXNQ,NQ
      REAL*8 WQ(MAXNQ),XQ(MAXNQ),YQ(MAXNQ)
      NQ=7
      WQ(1)=0.225000000000000D0
      WQ(2)=0.125939180544827D0
      WQ(3)=0.125939180544827D0
      WQ(4)=0.125939180544827D0
      WQ(5)=0.132394152788506D0
      WQ(6)=0.132394152788506D0
      WQ(7)=0.132394152788506D0

      XQ(1)=0.333333333333333D0
      XQ(2)=0.797426985353087D0
      XQ(3)=0.101286507323456D0
      XQ(4)=0.101286507323456D0
      XQ(5)=0.470142064105115D0
      XQ(6)=0.470142064105115D0
      XQ(7)=0.059715871789770D0

      YQ(1)=0.333333333333333D0
      YQ(2)=0.101286507323456D0
      YQ(3)=0.797426985353087D0
      YQ(4)=0.101286507323456D0
      YQ(5)=0.470142064105115D0
      YQ(6)=0.059715871789770D0
      YQ(7)=0.470142064105115D0

      END

C Subroutines required for H3LC (not in file H3LC.FOR) 
C  Subroutine for returning the square root.
       REAL*8 FUNCTION FNSQRT(X)
       REAL*8 X
       FNSQRT=SQRT(X)
       END

C  Subroutine for returning the exponential.
       COMPLEX*16 FUNCTION FNEXP(Z)
       COMPLEX*16 Z
       FNEXP=EXP(Z)
       END

C ----------------------------------------------------------------------
                                                                               