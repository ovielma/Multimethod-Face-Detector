
% This script will perform the entire training sequence
% Before you run this code, make sure that you are in the right directory 

% set directory path for code and data files
clc;
clear;
directories;

% training faces list
training_faces_path = [training_directory, 'training_faces'];
training_faces_list = dir(training_faces_path);
training_faces_list = remove_directories_from_dir_list(training_faces_list);

% training nonfaces list
training_nonfaces_path = [training_directory, 'training_nonfaces'];
training_nonfaces_list = dir(training_nonfaces_path);
training_nonfaces_list = remove_directories_from_dir_list(training_nonfaces_list);

% get sizes of lists
num_faces = size(training_faces_list, 1);
num_nonfaces = size(training_nonfaces_list, 1);

%%
% First step of the training procedure is to use rectangle filters with adaboost
% on all training images

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  Train Using Adaboost & Rectangle filters
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% Generate a set of cropped faces measuring 60x60 and roughly the same number
% of non-faces to use for training. By default each training image measures 100x100
% for our particular data set.

% Calculate integrals for all training samples. 

% The integral image is used as a quick and effective way of calculating the
% sum of values (pixel values) in a given image ? or a rectangular subset of 
% a grid (the given image). It can also, or is mainly, used for calculating 
% the average intensity within a given image. If one wants to use the integral
% image, it is normally a wise idea to make sure the image is in greyscale first.

% Create cell to store cropped faces from training set
cropFaces = cell(3047,1);
% Create cell to store face integrals calculated for training samples
faceIntegrals = cell(3047, 1);

% iterate through all number of faces to crop and generate integral images.
for i = 1:num_faces-1
    
    face2Crop = getfield(training_faces_list(i),'name');
    photo = read_gray(face2Crop);
    % crop face images with respect to the center
    centroid = (size(photo)/2)/2;
    trainingpatch = imcrop(photo, [centroid 59 59]);
    cropFaces{i} = trainingpatch;
    % calculate integral image from cropped faces
    A = cropFaces{i,1};
    B = integral_image(A);
    figure(i); imshow(B, []);
    faceIntegrals{i, 1} = B;
  
end

%%

%cell array for NonFaces
cropNonFaces = cell(130,20);
%cell array for integral images Non Face
integralNonFaces = cell(130,20);
% number of patches from each non-face image
numOfPatches = 20;


for i = 1:num_nonfaces-1
    
    nonFace2Crop = getfield(training_nonfaces_list(i),'name');
    photo = read_gray(nonFace2Crop);
    
    [h w]= size(photo);
    L = 60;
    
    for j = 1:numOfPatches
        
        cropNonFaces{i,j} = photo(randi(h-L+1)+(0:L-1),randi(w-L+1)+(0:L-1));
        integralNonFaces{i,j} = integral_image(cropNonFaces{i,j});
    end
            
end

%looking at patches of non face images
%imshow(cropNonFaces{100,17},[0 255]);
