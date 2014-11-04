%video generation
%clear all
%close all
load Xrobot;%single robot
load constants

disp 'going to make a video'
%needs: Xrobotx, y_s1, y_s2, x_s1, x_s2, y0, x1, x0, y1, dx, dy
%save Xrobot mytime Xrobotx Xroboty ithrobot dy dx x_s1 y_s1 x_s2 y_s2 V_x_matrix V_y_matrix U_s V1 U1 k1 k2 threshold x0 x1 y0 y1 cdraw t0 ts T_thresh k1 k2 dt



%load Xrobotmulti%multiple robot
[n1n2,n]=size(Xrobotx);
n = 1;

cdraw = 10
%Xrobot: mytime Xrobotx Xroboty ithrobot
%
%dy, dx, x_s1,y_s1,x_s2,y_s2,V_x_matrix,V_y_matrix,
%U_s,V1,U1,k1,k2,threshold,x0,x1,y0,y1,cdraw,k1,k2,T_thresh,dt


Dt=0.1;%visualization period


% A(1:length(t0:Dt:ts))=struct('cdata',[],'colormap',[]);

figure(3),hold on
%plot(y_s1,x_s1,'xr')
text(y_s1 ,x_s1+0.7,'source')
%hold off,
%plot(y_s2,x_s2,'xr')
%text(y_s2-1.5,x_s2+0.5,'S2')
axis equal
xlim([y0,y1])
ylim([x0,x1])
box on




nx=length(x0:dx:x1);
ny=length(y0:dy:y1);
[xv,yv] =meshgrid(y0:dy:y1,x0:dx:x1);
[xvdraw,yvdraw] =meshgrid(y0:cdraw*dy:y1,x0:cdraw*dx:x1);

figure(3),  quiver(xvdraw,yvdraw,-(V1(1:cdraw:end,1:cdraw:end)),-fliplr(U1(1:cdraw:end,1:cdraw:end)),.4,'y-')%[0,0,0.4]
%colormap
aaa=colormap(jet);
ccc=aaa(33:end,:);
bbb=[1 1 1;
    0.9 1 0.9;
    0.8 1 0.8;
    0.7 1 0.7;
    0.6 1 0.6;
    0.5 1 0.5];

ddd=[0.0833        0         0;
    0.1667         0         0;
    0.2500         0         0;
    0.3333         0         0;
    0.4167         0         0];
ccc=[bbb;ccc;ddd(end:-1:1,:)];   

axis equal
%axesValue=axis;
%State variable
U_matrix=zeros(nx,ny);

%Create auxiliary matrix
%\partial U/\partial x
dU_dx=zeros(nx,ny);
dU_dy=zeros(nx,ny);
dU_dz=zeros(nx,ny);
%Nebula U
d2U=zeros(nx,ny);


U_matrix(round(x_s1/dx),round(y_s1/dy))=U_s;
U_matrix(round(x_s2/dx),round(y_s2/dy))=U_s;



disp 'still making a video'

%new_robot_x=Xrobotx(1,:);
%new_robot_y=Xroboty(1,:);

figure(3),hold on
%plot(y_s1,x_s1,'xr')
h0=text(y1-5,x0+24,['t=',num2str(0),'s'],'FontSize',14);
%if n==1
%	h3=scatter(new_robot_y(1),new_robot_x(1),50,'ro');  
%else
%	h3=scatter(new_robot_y([1 n]),new_robot_x([1 n]),50,'ro');
%	h4=scatter(new_robot_y(2:n-1),new_robot_x(2:n-1),50,'bs');
%end
quiver(xvdraw,yvdraw,-(V1(1:cdraw:end,1:cdraw:end)),-fliplr(U1(1:cdraw:end,1:cdraw:end)),.4,'y-')
box on
axis equal
xlim([y0,y1])
ylim([x0,x1])
h1=pcolor(xv,yv,U_matrix);
set(h1,'edgecolor','none','facecolor','interp');
[abdc,h2]=contour(xv,yv,U_matrix,[threshold],'k-','LineWidth',1.5);
colormap(ccc)
% set(gca, 'nextplot', 'replacechildren');
caxis manual;          % allow subsequent plots to use the same color limits
colorbar
A(1:length(t0:Dt:ts))=struct('cdata',[],'colormap',[]);
A(1)=getframe;
myco=1;






for T=t0:Dt:ts
	T
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	set(gca,'YTick',flipud(get(gca,'YTick')));



	for t=0:dt:Dt-dt



		myco=myco+1;
		U_last=U_matrix;



		for i1=2:nx-1
			for i2=2:ny-1
				%update\partial U/\partial x and Nebula U
				dU_dx(i1,i2)=(V_x_matrix(i1,i2)>0)*(U_last(i1+1,i2)-U_last(i1,i2))/(dx)+(V_x_matrix(i1,i2)<0)*(U_last(i1,i2)-U_last(i1-1,i2))/(dx);
				dU_dy(i1,i2)=(V_y_matrix(i1,i2)>0)*(U_last(i1,i2+1)-U_last(i1,i2))/(dy)+(V_y_matrix(i1,i2)<0)*(U_last(i1,i2)-U_last(i1,i2-1))/(dy);
				%Nebula U
				d2U(i1,i2)=(U_last(i1+1,i2)+U_last(i1-1,i2)-2*U_last(i1,i2))./dx^2+(U_last(i1,i2+1)+U_last(i1,i2-1)-2*U_last(i1,i2))./dy^2;
				dU=V_x_matrix(i1,i2)*dU_dx(i1,i2)+V_y_matrix(i1,i2)*dU_dy(i1,i2)+k1*d2U(i1,i2)+k2*U_last(i1,i2);
				U_matrix(i1,i2)=U_last(i1,i2)+dU*dt;
			end
		end
		U_matrix(round(x_s1/dx),round(y_s1/dy))=U_s;
		U_matrix(round(x_s2/dx),round(y_s2/dy))=U_s;
	end




	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%last_robot_x=new_robot_x;
	%#last_robot_y=new_robot_y;
	%#new_robot_x = interp1(mytime,Xrobotx,T);
	%#new_robot_y = interp1(mytime,Xroboty,T);
	%#if (norm(new_robot_x-last_robot_x)+norm(new_robot_y-last_robot_y))>eps
	%#	plot([last_robot_y;new_robot_y],[last_robot_x;new_robot_x],'-', 'color',180*ones(1,3)/255);
	%#	if n==1
	%#	set(h3,'XData',new_robot_y(1),'YData',new_robot_x(1));
	%#	else
	%#	set(h3,'XData',new_robot_y([1 n]),'YData',new_robot_x([1 n]));
	%#	set(h4,'XData',new_robot_y(2:n-1),'YData',new_robot_x(2:n-1));
	%#	end
	%#end
	% contour(xv,yv,U_matrix,80,'k-','LineWidth',1.5);
	set(h1,'XData',xv,'YData',yv,'CData',U_matrix);
	set(h2,'XData',xv,'YData',yv,'ZData',U_matrix);
	set(h0,'string',['t=',num2str(T),'s']);
	refreshdata(h0)
	refreshdata(h1)
	refreshdata(h2)
	refreshdata(h3)
	if n>1
	refreshdata(h4)
	end
	%drawnow
	% haa=xlabel(['t=',num2str(T),'s']);
	% text(y1-4,x0+2,['t=',num2str(T),'s'])
	A(round((T-t0)/Dt+1)+1)=getframe;
	pause(0.01)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
movie2avi(A,'singlerobot','compression','none')

