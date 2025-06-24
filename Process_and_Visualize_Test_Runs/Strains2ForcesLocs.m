function [Fx, Fy, F, Zx, Zy, Z] = Strains2ForcesLocs(K,D,C,StrainAx, StrainBx, StrainAy, StrainBy)
% Computes forces and locations from strain values.

% Unpack Calibration Values:
% For reference, these are input as:
% C = [cAx, cAy, cBx, cBy];
cAx = C(1);
cAy = C(2);
cBx = C(3);
cBy = C(4);

% K = [kAx, kAy, kBx, kBy];
kAx = K(1);
kAy = K(2);
kBx = K(3);
kBy = K(4);

% D = [dAx, dAy, dBx, dBy];
dAx = D(1);
dAy = D(2);
dBx = D(3);
dBy = D(4);

% CALCULATIONS
Fx = (kBx*(StrainAx - cAx) - kAx*(StrainBx - cBx))/(kAx*kBx*(dBx - dAx));
Fy = (kBy*(StrainAy - cAy) - kAy*(StrainBy - cBy))/(kAy*kBy*(dBy - dAy));
F = sqrt(Fx.^2 + Fy.^2);

Zx = (kBx*dBx*(StrainAx - cAx) - kAx*dAx*(StrainBx - cBx))./(kBx*(StrainAx - cAx) - kAx*(StrainBx - cBx));
Zy = (kBy*dBy*(StrainAy - cAy) - kAy*dAy*(StrainBy - cBy))./(kBy*(StrainAy - cAy) - kAy*(StrainBy - cBy));

% THIS METHOD DOES NOT WORK WELL:
Z = mean([Zx,Zy]');

% Squared location weighting:
Wx = (Fx./F).^2;        
Wy = (Fy./F).^2;
WT = Wx + Wy;
Wx = Wx./WT;
Wy = Wy./WT;
Z = Wx.*Zx + Wy.*Zy;      % Signal weighted location estimate.