function output = setKernelData(input,location,range,value,kernel_type,width)
% output = set_kernel_data(input,location,range,value,kernel_type,width)
% Author: Joe
% Affiliation: XMU
% Create: 2022-12
% Updated Joe: 2023-11

if nargin<5    
   kernel_type = 'boll';  % uniform
end
if nargin<6
   width = [4,4,4];
end  
x0 = location(1);
y0 = location(2);
z0 = location(3);
size_data = size(input);
if strcmp(kernel_type , 'boll')
    if range<=4 && randi(5)>3
       value = -1.2-0.1*abs(randn(1));  
    end 
    for i =1:size_data(1)
        for j =1:size_data(2)
            for k =1:size_data(3)
                if (i-location(1))^2+(j-location(2))^2+(k-location(3))^2<range^2
                    noise = randn(1);
                    if abs(noise)>2
                        noise = 0;
                    end
                    input(i,j,k)=value+0.05*value*randn(1)*noise;
                end 
            end
        end
    end
elseif strcmp(kernel_type , 'ellipse')
    for i =1:size_data(1)
        for j =1:size_data(2)
            for k =1:size_data(3)
                if (i-location(1))^2/(width(1)*range)^2+(j-location(2))^2/(width(2)*range)^2+(k-location(3))^2/(width(3)*range)^2<1
                    noise = randn(1);
                    if abs(noise)>2
                        noise = 0; 
                    end
                    input(i,j,k)=value+0.05*value*randn(1)*noise;
                end 
            end
        end
    end
    
elseif strcmp(kernel_type , 'line_x')
%     size(input);
    temp = input(x0-range:x0+range,y0-width(2):y0,z0-width(3):z0);
    noise = 0.05*value*randn(size(temp));
    noise(noise>0.2*value)=0;
    if width(1)+width(2)==2
        noise = 4*noise;
    end
    input(x0-range:x0+range,y0-width(2):y0,z0-width(3):z0)=value+noise;
elseif strcmp(kernel_type , 'line_y')
    temp = input(x0-width(1),y0-range:y0+range,z0-width(3):z0);
    noise = 0.05*value*randn(size(temp));
    noise(noise>0.2*value)=0;
    if width(1)+width(2)==2
        noise = 4*noise;
    end
    input(x0-width(1),y0-range:y0+range,z0-width(3):z0)=value+noise;
elseif strcmp(kernel_type , 'line_z') 
    temp = input(x0-width(1):x0,y0-width(2):y0,z0-range:z0+range);
    noise = 0.05*value*randn(size(temp)); 
    noise(noise>0.2*value)=0;  
    if width(1)+width(2)==2
        noise = 4*noise; 
    end
    input(x0-width(1):x0,y0-width(2):y0,z0-range:z0+range)=value+noise;
    
elseif strcmp(kernel_type , 'line_rand')
    for i =1:size_data(1)
        for j =1:size_data(2)
            for k =max(1,z0-20):min(z0+20,size_data(3))
                if abs((i-x0)/width(1)-(j-y0)/width(2))<0.5 &&  abs((j-y0)/width(2)-(k-z0)/width(3))<0.4
                    noise = randn(1);
                    if abs(noise)>2
                        noise = 0;
                    end 
                    if abs(input(i,j,k))>0
                        input(i,j,k)=1/2*(value+0.05*value*randn(1)*noise+input(i,j,k));
                    else
                        input(i,j,k)=value+0.05*value*randn(1)*noise; 
                    end
                end 
            end
        end
    end
elseif strcmp(kernel_type , 'line_circle')
    value =3 + 0.1* abs(value);
    for i =1:size_data(1)
        for j =1:size_data(2)
            for k =1:size_data(3)
                if (i-x0)^2+(j-y0)^2+(k-z0)^2<(range+2)^2 && (i-x0)^2+(j-y0)^2+(k-z0)^2>(range-0.5)^2 &&k>z0
                    if (i-(x0+6))^2+(j-(y0+6))^2+(k-(z0+6))^2<(range+2)^2 && (i-(x0+6))^2+(j-(y0+6))^2+(k-(z0+6))^2>(range-0.5)^2 
                        noise = randn(1);  
                        if abs(noise)>2  
                            noise = 0; 
                        end
                        input(i,j,k)=value+0.05*value*randn(1)*noise;
                    end
                end 
            end
        end
    end
else
    disp('do nothing')       
end

output = input;
end