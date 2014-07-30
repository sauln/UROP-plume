
%%*****************************************%%
%%   Things that you might want to change  %%
n = 1; %number of robots

t0 = 0;					%start time
T_thresh = 2;			%time point to release robot
ts = 12;				%end time
dt = 0.001;
Dt = 0.002;				%visualization period

T_leader 	= 6000;		%time to start leader control
Dis_thresh 	= 2000;		%distance to activate leader control
U_s			= 3;
threshold 	= 0.1*U_s; %threshold of concentration detection

X0		= [12;24];
Xhat	= [11;25]; 

%%*****************************************%%



cdraw = 10;
ithrobot = 1;
mystorestore = zeros(0,2*n+1);




%%% These are for the control algorithm %%%%
k1 = 0.75;				
k2 = 0;    					%estimator gradient gain
k3 = 5;%10  				%adaptive control for estimator gradient gain
k4 = 5; 					%rotation gain--along the tangent direction
Xhatdot_max		= 10; 		%maximum velocity -- estimated value 
V0_robot_max	= 10; 		%robot velcity
v_compensate	= 1;% =0 	%velocity compensation   --- either 0 or 1, (on or off)
free_speed		= 6; 		%leader free speed for partroling
c_leader		= 20; 		%leader gain to drive the two leader close
c_r     		= 5; 		%robot gain to observed value
%%%





%%These things are for the environmental model %%%
%source:
x_s1 = 3; y_s1 = 13; 	
x_s2 = 3; y_s2 = 13;            
x0 = 0; x1 = 30;
y0 = 0; y1 = 20;
dx = 0.2; dy = 0.2;

U00 = zeros(n,1); 	%concentration at X0
Uhat = zeros(n,1);  	%concentration at Xhat

lx = 30; 		% width of box
ly = 20; 		% height of box
nx = length(x0:dx:x1);
ny = length(y0:dy:y1);
nx = 151;
ny = 101;
[x,y,U,V,Q,P] = mit18086_navierstokes1( lx, ly, nx, ny );

Len = sqrt(U.^2+V.^2+eps);
V1 = V./Len;
U1 = U./Len;

U_matrix = zeros(nx,ny);

%Velocity 
V_x_matrix = U(1:end-1,1:end-1);
V_y_matrix = V(1:end-1,1:end-1);

c0 = max(max(max(V_x_matrix)),max(max(V_y_matrix)));
V_x_matrix = 105*V_x_matrix/c0;
V_y_matrix = 105*V_y_matrix/c0;

%\partial U/\partial x
dU_dx = zeros(nx,ny);
dU_dy = zeros(nx,ny);
dU_dz = zeros(nx,ny);

%Nebula U
d2U = zeros(nx,ny);

U_matrix(round(x_s1/dx),round(y_s1/dy)) = U_s;
U_matrix(round(x_s2/dx),round(y_s2/dy)) = U_s;
U_last = U_matrix;



%these things are fro the kinematic model
l0 = 0.3;
d11 = 1;
m11 = 1;
d33 = 1;
m33 = 1;
l_kin = 0.2;
c01 = 500;
k11 = -d11/m11;
k21 = -d33/m33;
k31 = 1/m11;
k41 = 1/(m33*l_kin);
A_kin  = diag(k11,k21);
B_kin  = [k31, k31; -k41, k41];

zreal = [24; 5];
theta = 2*pi*rand(1,1);
x_sur = zreal-l0*[cos(theta); sin(theta)];
xytheta = [x_sur(1); x_sur(2); theta];
u_kin = [.5;.5] ;
%%


save constants l_kin xytheta u_kin l0 d11 m11 d33 m33 c01 k11 k21 k31 k41 A_kin B_kin U_last mystorestore d2U dU_dz dU_dy dU_dx c0 V_y_matrix V_x_matrix U_matrix V1 U1 Len Uhat U00 x y U V Q P n ts T_thresh t0 dt Dt T_leader Dis_thresh U_s threshold lx ly k1 k2 k3 k4 Xhatdot_max V0_robot_max v_compensate free_speed c_leader c_r x_s1 y_s1 x_s2 y_s2 x0 x1 y0 y1 dx dy nx ny X0 Xhat cdraw ithrobot




 
