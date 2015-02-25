%this is a test function of the set of equations from the matlab function


%Up: 0.12  Down: 0.06 
%Left: 0.12 Right: 0.12
%c: 1.0 DU: (0.20, 0.0)  DU_p: [-0.  1.]
%V0: (0.0, -1.0)   D2U0: 2.0
%U0: 0.0036

function [DU, DU_p, V0, D2U0, U0] = enviroTest(c, x1, x2, x3, x4, vy, vx)

	dt = 0.001;

	dx = 0.2; dy = 0.2;
	aX = x1; bX = c; cX = x2; aY = x3; bY = c; cY = x4;
	xa = aX; xb = bX; xc = cX; ya = aY; yb = bY; yc = cY;
	xV = vx; yV = vy;


	DU_dy0 = (yV>=0)*(ya-yb)/(dy)+(yV<0)*(yb-yc)/(dy);
  	DU_dx0 = (xV>=0)*(xa-xb)/(dx)+(xV<0)*(xb-xc)/(dx);
	DU = [DU_dx0;DU_dy0];
	V0 = [xV;yV];
	DU_p = [-DU_dy0;DU_dx0];
	DU_p = DU_p/norm(DU_p+eps);
	D2U0 = (aX+cX-2*bX)./dx^2+(aY+cY-2*bY)./dy^2;

	U0 = c











