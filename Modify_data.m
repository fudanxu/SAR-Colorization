
function data_modified = Modify_data(data)
% Modify_data.m
% This is used to correct the covariance matrix by pixel-wise processing.
% by Qian Song July 3 2017
% Input and Output are covariance matrix.

[row,col,~] = size(data);
data_modified = data;

r_13     = abs(data(:,:,4)+1i*data(:,:,5));
disp(length(find(r_13>1))/row/col)
r_13(r_13>1) = 1.0;
phi_13 = angle(data(:,:,4)+1i*data(:,:,5));

r_23     = abs(data(:,:,6)+1i*data(:,:,7));
disp(length(find(r_23>1))/row/col)
r_23(r_23>1) = 1.0;
phi_23 = angle(data(:,:,6)+1i*data(:,:,7));

r_12     = abs(data(:,:,8)+1i*data(:,:,9));
disp(length(find(r_12>1))/row/col)
r_12(r_12>1) = 1.0;
phi_12 = angle(data(:,:,8)+1i*data(:,:,9));

% mask = 1+2.*r_12.*cos(phi_12).*r_23.*cos(phi_23).*r_13.*cos(-phi_13) >= (r_13.^2+r_23.^2+r_12.^2);
mask = 1+2.*r_12.*cos(phi_12+phi_23 - phi_13).*r_23.*r_13 >= (r_13.^2+r_23.^2+r_12.^2);
R = (r_13.^2+r_23.^2+r_12.^2-1)./(2*r_12.*r_23.*r_13);
temp = sqrt((1-r_13.^2)./(r_23.^2+r_12.^2-2*r_13.*r_23.*r_12));
r_23(R>1) = r_23(R>1).*temp(R>1);
r_12(R>1) = r_12(R>1).*temp(R>1);
R(R>1) = 1.0;
% r_23(R>1) = r_23(R>1).*temp(R>1);
% r_12(R>1) = r_12(R>1).*temp(R>1);
temp_phi = acos(R) - phi_23 - phi_12 + phi_13;
phi_23 = phi_23 + temp_phi./2;
phi_12 = phi_12 + temp_phi./2;

data(:,:,6) = r_23.*cos(phi_23);
data(:,:,7) = r_23.*sin(phi_23);
data(:,:,8) = r_12.*cos(phi_12);
data(:,:,9) = r_12.*sin(phi_12);

for i = 1:row
    for j = 1:col
        if (mask(i,j)==0)
            data_modified(i,j,6) = data(i,j,6);
            data_modified(i,j,7) = data(i,j,7);
            data_modified(i,j,8) = data(i,j,8);
            data_modified(i,j,9) = data(i,j,9);
        end
    end
    disp([num2str(i/row*100),'%'])
end

