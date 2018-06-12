C*************************************************************
C            Subroutine RIM3 by Stephen Kirkup                      
C*************************************************************
C
C  Copyright 2004- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/RIM3.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C
C This subroutine computes the solution to the three-dimensional 
C Helmholtz equation
C                  __ 2                2
C                  \/    {\phi}   +   k  {\phi}   =  0   
C
C in the domain exterior to a plate in an infinite rigid baffle.
C
C The plate (\Pi) is defined (approximated) by a set of planar 
C triangular elements. The domain of the equation is the half-space
C exterior to the plate.
C
C The plate condition may be Dirichlet, Robin or Neumann. It is 
C assumed to have the following general form
C
C           {\alpha}(q) {\phi}(q) + {\beta}(q) v(q) = f(q)
C    
C where {\phi}(q) is the solution at the point q on \Pi, v(q) is the 
C derivative of {\phi} with respect to the outward normal to \Pi at q
C and {\alpha}, {\beta} and f are complex-valued functions defined on 
C \Pi. The functions {\alpha} and {\beta} must be specified to define 
C the nature of the plate condition. Important examples are 
C {\alpha}=1, {\beta}=0 which is equivalent to a Dirichlet plate 
C condition and {\alpha}=0, {\beta}=1 which is equivalent to a 
C Neumann plate condition. The specification of f completes the 
C definition of the plate condition. On the baffle, the condition
C v=0 is assumed.
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
C      |   MAIN PROGRAM     |------->-----|      RIM3              |   :
C      | (e.g. rim3_t.for)  |       :     |                        |   :
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
C                                  /         |                 |
C                               |_           >                 >
C                              /             |                 |
C             ................       ................   ................  
C             :              :       :   --------   :   :  --------    : 
C             : (geom3d.for) :---<---:   | H3LC |   :   :  | CGLS |    : 
C             :              :       :   --------   :   :  --------    :  
C             :..............:       : -------------:   : -------------:  
C                                    : |subordinate|:   : |subordinate|: 
C                                    : | routines  |:   : | routines  |:  
C                                    : -------------:   : -------------: 
C                                    :              :   :              : 
C                                    : (h3lc.for)   :   : (cgls.for)   :
C                                    :..............:   :..............:
C                                    
C
C The contents of the main program must be linked to RIM3.FOR, H3LC.FOR
C and CGLS.FOR.
C
C Method of solution
C ------------------
C 
C In the main program, the plate must be described as a set of
C  elements. The elements are defined by three indices (integers) which
C  label a node or vertex on the plate. The data structure VERTEX 
C  lists and enumerates the coordinates of the vertices, the data 
C  structure SELV defines each element by indicating the labels for
C  the three nodes that are its vertices and hence enumerates the
C  elements.
C The plate solution points (the points on the plate at which 
C  {\phi} (PIPHI) and d {\phi}/dn (PIVEL) are returned) are at the centres
C  of the elements. The plate functions {\alpha} (PIALPHA), {\beta} 
C  (PIBETA) and f (PIF) are also defined by their values at the centres
C  of the elements.
C Normally a solution in the domain is required. By listing the 
C  coordinates of all the exterior points in PEXT, the subroutine
C  returns the value of {\phi} at these points in PEPHI.
C
C
C Format and parameters of the subroutine
C ---------------------------------------
C
C The subroutine has the form
C
C      SUBROUTINE RIM3(K,
C     *                 MAXNV,NV,VERTEX,MAXNPI,NPI,SELV,
C     *                 MAXNPE,NPE,PEXT,
C     *                 PIALPHA,PIBETA,PIF,
C     *                 LSOL,LVALID,EGEOM,
C     *                 PIPHI,PIVEL,PEPHI,
C     *                 WKSPC1,WKSPC2,WKSPC3,WKSPC4,WKSPC5,WKSPC6)

C The parameters to the subroutine
C ================================

C Wavenumber (input)
C real K: Must be positive.

C Geometry of the plate {\Pi} (input)
C integer MAXNV: The limit on the number of vertices of the triangles
C  that defines (approximates) {\Pi}. MAXNV>=3.
C integer NV: The number of vertices on {\Pi}. 3<=NV<=MAXNV.
C real VERTEX(MAXNV,3): The coordinates of the vertices. VERTEX(i,1),
C  VERTEX(i,2), VERTEX(i,3) are the x,y,z coordinates of the i-th 
C  vertex. The vertices must be co-planar.
C integer MAXNPI: The limit on the number of elements describing {\Pi}.
C  MAXNPI>=1.
C integer NPI: The number of elements describing {\Pi}. 1<=NPI<=MAXNPI.
C integer SELV(MAXNPI,3): The indices of the three vertices defining
C  each element. The i-th element have vertices 
C  (VERTEX(SELV(i,1),1),VERTEX(SELV(i,1),2)),VERTEX(SELV(i,1),3)),
C  (VERTEX(SELV(i,2),1),VERTEX(SELV(i,2),2)),VERTEX(SELV(i,2),3)) and
C  (VERTEX(SELV(i,3),1),VERTEX(SELV(i,3),2)),VERTEX(SELV(i,3),3)).

C Exterior points at which the solution is to be observed (input)
C integer MAXNPE: Limit on the number of points exterior to the
C  plate. MAXNPE>=1.
C integer NPE: The number of exterior points. 0<=NPE<=MAXNPE.
C real PEXT(MAXNPE,3). The coordinates of the exterior point.
C  PEXT(i,1),PEXT(i,2),PEXT(i,3) are the x,y,z coordinates of the i-th
C  point. 

C The plate condition ({\alpha} phi + {\beta} v = f) (input)
C complex PIALPHA(MAXNPI): The values of {\alpha} at the centres
C  of the elements.
C complex PIBETA(MAXNPI): The values of {\beta} at the centres
C  of the elements.
C complex PIF(MAXNPI): The values of f at the centres of the 
C  elements.

C Validation and control parameters (input) 
C logical LSOL: A switch to control whether the particular solution is
C  required.
C logical LVALID: A switch to enable the choice of checking of 
C  subroutine parameters.
C real EGEOM: The maximum absolute error in the parameters that
C  describe the geometry.

C Solution (output)
C complex PIPHI(MAXNPI): The velocity potential ({\phi}) at the 
C  centres of the plate elements.
C complex PIVEL(MAXNPI): The velocity (v or d{\phi}/dn where n is
C  the outward normal to the plate) at the centres of the plate 
C  elements.
C complex PEPHI(MAXNPE): The velocity potential ({\phi}) at the 
C  exterior points.

C  Working space 
C      COMPLEX*16 WKSPC1(MAXNPI,MAXNPI)
C      COMPLEX*16 WKSPC2(MAXNPE,MAXNPI)
C      COMPLEX*16 WKSPC3(MAXNPI,MAXNPI)
C      COMPLEX*16 WKSPC4(MAXNPI)
C      COMPLEX*16 WKSPC5(MAXNPI)
C      LOGICAL    WKSPC6(MAXNPI)

C Notes on the geometric parameters
C ---------------------------------
C (1) Each of the vertices listed in VERTEX must be distinct points
C  with respect to EGEOM.
C (2) The vertices in SELV must be co-planar.
C (3) The indices of the nodes listed in SELV must be such that they
C  are ordered counter-clockwise around the plate, when viewed
C  from above the plate.
C (4) The largest element must be no more than 10x the area of the
C  smallest element.

C Notes on the exterior points 
C ----------------------------
C (1) The points in PEXT should lie outside the plate, as defined
C  by the parameters VERTEX and SELV. Any point lying outside the 
C  plate and baffle.

C Notes on the plate condition
C -------------------------------
C (1) For each i=1..NPI, it must not be the case that both of PIALPHA(i)
C  and PIBETA(i) are zero

C External modules in external files
C ==================================
C subroutine H3LC: Returns the individual discrete Helmholtz integral
C  operators. (in file H3LC.FOR)
C subroutine CGLS: Solves a general linear system of equations. 
C  (in file CGLS.FOR)

C External modules provided in the package (this file)
C ====================================================
C subroutine GLT7: Returns the points and weights of the 7-point Gauss-
C  Legendre quadrature rule on the standard triangle.
C real function FNSQRT(X): real X : Returns the square root of X.
C complex function FNEXP(Z): complex Z : Returns the complex exponential
C  of Z.


C The subroutine

      SUBROUTINE RIM3(K,
     *                 MAXNV,NV,VERTEX,MAXNPI,NPI,SELV,
     *                 MAXNPE,NPE,PEXT,
     *                 PIALPHA,PIBETA,PIF,
     *                 LSOL,LVALID,EGEOM,
     *                 PIPHI,PIVEL,PEPHI,
     *                 WKSPC1,WKSPC2,WKSPC3,WKSPC4,WKSPC5,WKSPC6)
      PARAMETER (MAXNQ=100)

C  Wavenumber
      REAL*8     K

C  Boundary geometry
C   Limit on the number of vertices on {\Pi}
      INTEGER    MAXNV
C   The number of vertices on {\Pi}
      INTEGER    NV
C   The coordinates of the vertices on {\Pi}
      REAL*8     VERTEX(MAXNV,3)
C   Limit on the number of elements describing {\Pi}
      INTEGER    MAXNPI
C   The number of elements describing {\Pi}
      INTEGER    NPI
C   The indices of the vertices describing each element
      INTEGER    SELV(MAXNPI,3)
      
C  Exterior points at which the solution is to be observed
C   Limit on the number of points exterior to the plate where 
C    solution is sought
      INTEGER    MAXNPE
C   The number of exterior points
      INTEGER    NPE
C   Coordinates of the exterior points
      REAL*8     PEXT(MAXNPE,3)

C  The plate condition is such that {\alpha} {\phi} + {\beta} v = f
C  where alpha, beta and f are complex valued functions over {\Pi}.
C  The functions are set values at the collocation points.
C   function alpha
      COMPLEX*16 PIALPHA(MAXNPI)
C   function beta
      COMPLEX*16 PIBETA(MAXNPI)
C   function f
      COMPLEX*16 PIF(MAXNPI)


C  Validation and control parameters
      LOGICAL    LSOL
      LOGICAL    LVALID
      REAL*8     EGEOM

C  Solution 
C   function phi
      COMPLEX*16 PIPHI(MAXNPI)
C   function vel
      COMPLEX*16 PIVEL(MAXNPI)
C   domain solution
      COMPLEX*16 PEPHI(MAXNPE)

C  Working space 
      COMPLEX*16 WKSPC1(MAXNPI,MAXNPI)
      COMPLEX*16 WKSPC2(MAXNPE,MAXNPI)
      COMPLEX*16 WKSPC3(MAXNPI,MAXNPI)
      COMPLEX*16 WKSPC4(MAXNPI)
      COMPLEX*16 WKSPC5(MAXNPI)
      LOGICAL    WKSPC6(MAXNPI)

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

C  Geometrical description of the plate
C   Elements counter
      INTEGER    ISE,JSE
C   The points exterior to the plate where the solution is sought 
      INTEGER    IPE
C   Parameters for H3LC
      REAL*8     P(3),PA(3),PB(3),PC(3),QA(3),QB(3),QC(3),VECP(3)
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
C    Abscissae of the actual quadrature rule
      REAL*8     XQ(MAXNQ)
C    Abscissae of the actual quadrature rule
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
C   The `diameter' of the plate or the maximum distance between any
C    three vertices
      REAL*8     DIAM
     
      REAL*8     NORMP(3),NORMQ(3)

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

C Validation
C ==========

C Validation of parameters of RIM3
C ---------------------------------

      IF (LVALID) THEN

C Validation of main paramters
        LERROR=.FALSE.
        IF (K.LT.ZERO) THEN
          WRITE(*,*) 'K = ',K
          WRITE(*,*) 'ERROR(RIM3) - K must be positive'
          LERROR=.TRUE.
        END IF
        IF (MAXNV.LT.3) THEN
          WRITE(*,*) 'MAXNV = ',MAXNV
          WRITE(*,*) 'ERROR(RIM3) - must have MAXNV>=3'
          LERROR=.TRUE.
        END IF
        IF (NV.LT.3.OR.NV.GT.MAXNV) THEN
          WRITE(*,*) 'NV = ',NV
          WRITE(*,*) 'ERROR(RIM3) - must have 3<=NV<=MAXNV'
          LERROR=.TRUE.
        END IF
        IF (MAXNPI.LT.1) THEN
          WRITE(*,*) 'MAXNPI = ',MAXNPI
          WRITE(*,*) 'ERROR(RIM3) - must have MAXNPI>=1'
          LERROR=.TRUE.
        END IF
        IF (NPI.LT.1.OR.NPI.GT.MAXNPI) THEN
          WRITE(*,*) 'NPI = ',NPI
          WRITE(*,*) 'ERROR(RIM3) - must have 1<=NPI<=MAXNPI'
          LERROR=.TRUE.
        END IF
        IF (MAXNPE.LT.1) THEN
          WRITE(*,*) 'MAXNPE = ',MAXNPE
          WRITE(*,*) 'ERROR(RIM3) - must have MAXNPE>=1'
          LERROR=.TRUE.
        END IF
        IF (NPE.LT.0.OR.NPE.GT.MAXNPE) THEN
          WRITE(*,*) 'NPE = ',NPE
          WRITE(*,*) 'ERROR(RIM3) - must have 0<=NPE<=MAXNPE'
          LERROR=.TRUE.
        END IF
        IF (EGEOM.LE.ZERO) THEN
          WRITE(*,*) 'NPE = ',NPE
          WRITE(*,*) 'ERROR(RIM3) - EGEOM must be positive'
          LERROR=.TRUE.
        END IF
        IF (LERROR) THEN
          LFAIL=.TRUE.
          WRITE(*,*)
          WRITE(*,*) 'Error(s) found in the main parameters of RIM3'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF

C Find the diameter DIAM of the plate
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


C Check that EGEOM is not too large
        IF (EGEOM.GT.DIAM/100.0D0) THEN
          WRITE(*,*) 'EGEOM = ',EGEOM
          WRITE(*,*) 'ERROR(RIM3) - EGEOM is set too large'
          LERROR=.TRUE.
        END IF
        IF (LERROR) THEN
          LFAIL=.TRUE.
          WRITE(*,*)
          WRITE(*,*) 'Error in plate geometry or EGEOM'
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
          WRITE(*,*) 'ERROR(RIM3) - Vertices (see above) coincide'
          WRITE(*,*) 'Execution terminated'
          STOP
        END IF
      END IF          

C Check that vertices are co-planar
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
      DO 125 ISE=2,NPI
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
          WRITE(*,*) 'WARNING(RIM3) - Panels are not co-planar'
      END IF


C Check that the elements are not of disproportionate sizes
      IF (LVALID) THEN
        SIZMAX=ZERO
        SIZMIN=DIAM**2
        DO 150 ISE=1,NPI
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
          WRITE(*,*) 'WARNING(RIM3) - Elements of disproportionate'
          WRITE(*,*) ' sizes'
        END IF
      END IF          
          

C Validation of the plate functions
      IF (LVALID) THEN
        LERROR=.FALSE.
        DO 170 ISE=1,NPI
          IF (MAX(ABS(PIALPHA(ISE)),ABS(PIBETA(ISE))).LT.1.0D-6) 
     *     LERROR=.TRUE.
170     CONTINUE
        IF (LERROR) THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR(RIM3) - at most one of PIALPHA(i),PIBETA(i)'
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
C    Gaussian quadrature rule above. If LPONEL is true then then a
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


C  Compute the discrete Lk, Mk, Mkt and Nk matrices
C   Loop(ISE) through the points on the plate
      DO 510 ISE=1,NPI
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
C    Set VECP to the normal on the plate of the element at P
        CALL NORM3(PA,PB,PC,VECP)
C    Loop(ISE) through the elements
        DO 520 JSE=1,NPI
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

C   All operators are required
          LLK=.TRUE.
          LMK=.FALSE.
          LMKT=.FALSE.
          LNK=.FALSE.

C     Call H3LC.
          CALL H3LC(CK,P,VECP,QA,QB,QC,LPONEL,
     *     MAXNQ,NQ,XQ,YQ,WQ,
     *     LVAL,EK,EGEOM,EQRULE,LFAIL1,
     *     LLK,LMK,LMKT,LNK,DISLK,DISMK,DISMKT,DISNK)

          WKSPC1(ISE,JSE)=-2.0D0*DISLK
          WKSPC3(ISE,JSE)=CZERO
         
C    Close loop(JSE) 
520     CONTINUE

        WKSPC3(ISE,ISE)=CONE
        WKSPC5(ISE)=CZERO
            
C   Close loop(ISE) 
510   CONTINUE

      IF (LSOL) THEN
       CALL CGLS(MAXNPI,NPI,WKSPC3,WKSPC1,WKSPC5,PIALPHA,PIBETA,PIF,
     * PIPHI,PIVEL,LFAIL,WKSPC4,WKSPC6)        
      END IF  

C  SOLUTION IN THE DOMAIN

C   Compute sound pressures at the selected exterior points.
C    Loop through the the points in the exterior region
      DO 800 IPE=1,NPE
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
        DO 850 JSE=1,NPI
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
          SUMPHI=SUMPHI-2.0D0*DISLK*PIVEL(JSE)
          IF (LSOL) WKSPC2(IPE,JSE)=-2.0D0*DISLK

C      Close loop (JSE) through the elements
850     CONTINUE

        PEPHI(IPE)=SUMPHI

C     Close loop(IPE) through the exterior points
800   CONTINUE

      END


C ----------------------------------------------------------------------

C Subordinate routines for RIM3
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
