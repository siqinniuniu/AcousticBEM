C***************************************************************
C                       Hankel Function 2                              
C***************************************************************
C 
C  Copyright 1998- Stephen Kirkup
C  School of Computing Engineering and Physical Sciences
C  University of Central Lancashire - www.uclan.ac.uk 
C  smkirkup@uclan.ac.uk
C  http://www.researchgate.net/profile/Stephen_Kirkup
C
C  This open source code can be found at
C   www.boundary-element-method.com/fortran/FNHANK2.FOR
C
C  Issued under the GNU General Public License 2007, see gpl.txt
C
C  Part of the the author's open source BEM packages. 
C  All codes and manuals can be downloaded from 
C  www.boundary-element-method.com
C
C***************************************************************
C
      
      SUBROUTINE FNHANK(Z,H)
      COMPLEX*16 Z
      COMPLEX*16 H(0:1)
      REAL*8 BESSJ0,BESSJ1,BESSY0,BESSY1
      REAL*8 X,Y
      X=DREAL(Z)
      Y=DIMAG(Z)
      IF (Y.GT.1.0D-6) THEN
        WRITE(*,*) 'ERROR(FNHANK) - Im(Z) must be zero'
        STOP
      END IF
      H(0)=CMPLX(BESSJ0(X),BESSY0(X))
      H(1)=CMPLX(BESSJ1(X),BESSY1(X))
      END


      REAL*8 FUNCTION BESSJ0(X)
      REAL*8 X
      REAL*8 AX,XX,Z
      REAL*8 P1,P2,P3,P4,P5,Q1,Q2,Q3,Q4,Q5,R1,R2,R3,R4,R5,R6,
     *S1,S2,S3,S4,S5,S6,Y
      SAVE P1,P2,P3,P4,P5,Q1,Q2,Q3,Q4,Q5,R1,R2,R3,R4,R5,R6,S1,S2,S3,S4,
     *S5,S6
      DATA P1,P2,P3,P4,P5/1.D0,-.1098628627D-2,.2734510407D-4,
     *-.2073370639D-5,.2093887211D-6/, Q1,Q2,Q3,Q4,Q5/-.1562499995D-1,
     *.1430488765D-3,-.6911147651D-5,.7621095161D-6,-.934945152D-7/
      DATA R1,R2,R3,R4,R5,R6/57568490574.D0,-13362590354.D0,
     *651619640.7D0,-11214424.18D0,77392.33017D0,-184.9052456D0/,S1,S2,
     *S3,S4,S5,S6/57568490411.D0,1029532985.D0,9494680.718D0,
     *59272.64853D0,267.8532712D0,1.D0/
      IF(ABS(X).LT.8.)THEN
        Y=X**2
        BESSJ0=(R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6)))))/(S1+Y*(S2+Y*(S3+Y*
     *(S4+Y*(S5+Y*S6)))))
      ELSE
        AX=ABS(X)
        Z=8./AX
        Y=Z**2
        XX=AX-.785398164
        BESSJ0=SQRT(.636619772/AX)*(COS(XX)*(P1+Y*(P2+Y*(P3+Y*(P4+Y*
     *P5))))-Z*SIN(XX)*(Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)))))
      ENDIF
      RETURN
      END

      REAL*8 FUNCTION BESSJ1(X)
      REAL*8 X
      REAL*8 AX,XX,Z
      REAL*8 ONE
      REAL*8 P1,P2,P3,P4,P5,Q1,Q2,Q3,Q4,Q5,R1,R2,R3,R4,R5,R6,
     *S1,S2,S3,S4,S5,S6,Y
      SAVE P1,P2,P3,P4,P5,Q1,Q2,Q3,Q4,Q5,R1,R2,R3,R4,R5,R6,S1,S2,S3,S4,
     *S5,S6
      DATA R1,R2,R3,R4,R5,R6/72362614232.D0,-7895059235.D0,
     *242396853.1D0,-2972611.439D0,15704.48260D0,-30.16036606D0/,S1,S2,
     *S3,S4,S5,S6/144725228442.D0,2300535178.D0,18583304.74D0,
     *99447.43394D0,376.9991397D0,1.D0/
      DATA P1,P2,P3,P4,P5/1.D0,.183105D-2,-.3516396496D-4,
     *.2457520174D-5,-.240337019D-6/, Q1,Q2,Q3,Q4,Q5/.04687499995D0,
     *-.2002690873D-3,.8449199096D-5,-.88228987D-6,.105787412D-6/
      ONE=1.0D0
      IF(ABS(X).LT.8.)THEN
        Y=X**2
        BESSJ1=X*(R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6)))))/(S1+Y*(S2+Y*(S3+
     *Y*(S4+Y*(S5+Y*S6)))))
      ELSE
        AX=ABS(X)
        Z=8./AX
        Y=Z**2
        XX=AX-2.356194491
        BESSJ1=SQRT(.636619772/AX)*(COS(XX)*(P1+Y*(P2+Y*(P3+Y*(P4+Y*
     *P5))))-Z*SIN(XX)*(Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)))))*SIGN(ONE,X)
      ENDIF
      RETURN
      END

      REAL*8 FUNCTION BESSY0(X)
      REAL*8 X
      REAL*8 XX,Z,BESSJ0
      REAL*8 P1,P2,P3,P4,P5,Q1,Q2,Q3,Q4,Q5,R1,R2,R3,R4,R5,R6,
     *S1,S2,S3,S4,S5,S6,Y
      SAVE P1,P2,P3,P4,P5,Q1,Q2,Q3,Q4,Q5,R1,R2,R3,R4,R5,R6,S1,S2,S3,S4,
     *S5,S6
      DATA P1,P2,P3,P4,P5/1.D0,-.1098628627D-2,.2734510407D-4,
     *-.2073370639D-5,.2093887211D-6/, Q1,Q2,Q3,Q4,Q5/-.1562499995D-1,
     *.1430488765D-3,-.6911147651D-5,.7621095161D-6,-.934945152D-7/
      DATA R1,R2,R3,R4,R5,R6/-2957821389.D0,7062834065.D0,
     *-512359803.6D0,10879881.29D0,-86327.92757D0,228.4622733D0/,S1,S2,
     *S3,S4,S5,S6/40076544269.D0,745249964.8D0,7189466.438D0,
     *47447.26470D0,226.1030244D0,1.D0/
      IF(X.LT.8.)THEN
        Y=X**2
        BESSY0=(R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6)))))/(S1+Y*(S2+Y*(S3+Y*
     *(S4+Y*(S5+Y*S6)))))+.636619772*BESSJ0(X)*LOG(X)
      ELSE
        Z=8./X
        Y=Z**2
        XX=X-.785398164
        BESSY0=SQRT(.636619772/X)*(SIN(XX)*(P1+Y*(P2+Y*(P3+Y*(P4+Y*
     *P5))))+Z*COS(XX)*(Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)))))
      ENDIF
      RETURN
      END

      REAL*8 FUNCTION BESSY1(X)
      REAL*8 X
      REAL*8 XX,Z,BESSJ1
      REAL*8 P1,P2,P3,P4,P5,Q1,Q2,Q3,Q4,Q5,R1,R2,R3,R4,R5,R6,
     *S1,S2,S3,S4,S5,S6,S7,Y
      SAVE P1,P2,P3,P4,P5,Q1,Q2,Q3,Q4,Q5,R1,R2,R3,R4,R5,R6,S1,S2,S3,S4,
     *S5,S6,S7
      DATA P1,P2,P3,P4,P5/1.D0,.183105D-2,-.3516396496D-4,
     *.2457520174D-5,-.240337019D-6/, Q1,Q2,Q3,Q4,Q5/.04687499995D0,
     *-.2002690873D-3,.8449199096D-5,-.88228987D-6,.105787412D-6/
      DATA R1,R2,R3,R4,R5,R6/-.4900604943D13,.1275274390D13,
     *-.5153438139D11,.7349264551D9,-.4237922726D7,.8511937935D4/,S1,S2,
     *S3,S4,S5,S6,S7/.2499580570D14,.4244419664D12,.3733650367D10,
     *.2245904002D8,.1020426050D6,.3549632885D3,1.D0/
      IF(X.LT.8.)THEN
        Y=X**2
        BESSY1=X*(R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6)))))/(S1+Y*(S2+Y*(S3+
     *Y*(S4+Y*(S5+Y*(S6+Y*S7))))))+.636619772*(BESSJ1(X)*LOG(X)-1./X)
      ELSE
        Z=8./X
        Y=Z**2
        XX=X-2.356194491
        BESSY1=SQRT(.636619772/X)*(SIN(XX)*(P1+Y*(P2+Y*(P3+Y*(P4+Y*
     *P5))))+Z*COS(XX)*(Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)))))
      ENDIF
      RETURN
      END

                                                              