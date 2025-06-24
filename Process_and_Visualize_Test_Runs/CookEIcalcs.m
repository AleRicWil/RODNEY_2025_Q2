function [EI, FdefRsquared, deflection, F, FitCoeffs] = CookEIcalcs(Fx, Fy, Zx, Zy, YawAngle, height)


% Squared location weighting:
F = sqrt(Fx.^2 + Fy.^2);
Wx = (Fx./F).^2;
Wy = (Fy./F).^2;
WT = Wx + Wy;
Wx = Wx./WT;
Wy = Wy./WT;
Z = Wx.*Zx + Wy.*Zy;      % Signal weighted location estimate.


YawAngle = YawAngle*pi/180;

deflection = -1*Z.*sin(YawAngle);

X = [ones(length(deflection),1), deflection(:)];

[FitCoeffs,~,~,~,stats] = regress(F(:),X);

% From Christian's IETC Paper:
Theta0 = pi/4 - atan2(Fy, Fx);
Fyp = F.*cos(pi/2 - Theta0);

% NOTE: This needs to be corrected based on a careful analysis of the
% angles.
% EI = FitCoeffs(2)*height^3/3 + 0.5*median(Fyp);

EI = FitCoeffs(2)*height^3/3;

FdefRsquared = stats(1);
Fc = F;
% 
% subplot(2,3,1)
% plot(Fx,'o')
% title('Fx')
% 
% subplot(2,3,2)
% plot(Fy,'o')
% title('Fy')
% 
% subplot(2,3,3)
% plot(atan2(Fy,Fx)*180/pi,'o')
% title('Theta')
% 
% subplot(2,3,4)
% plot(Zx,'o'), 
% hold on, 
% plot(Zy,'o')
% plot(Z,'LineWidth',2)
% title('Locations')
% hold off
% 
% % subplot(2,3,5)
% % plot(Z,'o')
% 
% subplot(2,3,6)
% plot(deflection, F,'o')
% title('Force/Def')
% 
% pause

end



