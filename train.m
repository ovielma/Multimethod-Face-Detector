% This script will perform the entire training sequence
% Before you run this code, make sure that you are in the right directory 
% and add the appropriate datapath manually (00_common, 00_common_data, 12_boosting),
% as hardcoding the datapath will require changing it everytime we
% pull changes each of us have made.
% Likewise for all data including training and test images.

% First step of the training procedure is to use rectangle filters on all 
% training images (refer to main_script in 12_boosting code example on TRACS).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  Rectangle filters
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% Test run of rectangle filter type 1. It looks like we will
% need to iterate through all training images with a for loop for training.

clear; close all;
photo = read_gray('04202d61.bmp');
figure(1); imshow(photo, []);
rec_filter = rectangle_filter1(1, 1);
result = imfilter(photo, rec_filter, 'same', 'symmetric');
figure(2); imshow(result, []);

rec_filter2 = rectangle_filter1(20, 10);
result2 = imfilter(photo, rec_filter2, 'same', 'symmetric');
figure(3); imshow(result2, []);
