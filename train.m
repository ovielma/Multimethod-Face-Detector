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
% First step of the training procedure is to use rectangle filters on all 
% training images (refer to main_script in 12_boosting code example on TRACS)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  Train Using Adaboost & Rectangle filters
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% For each training_face and training_nonface image, we need to crop
% to 60x60 with respect to the centroid of the training image. By default
% each training image measures 100x100
%%
cropFaces = {};


for i = 1:num_faces-1
    
    face2Crop = getfield(training_faces_list(i),'name');
    photo = read_gray(face2Crop);
    centroid = (size(photo)/2)/2;
    trainingpatch = imcrop(photo, [centroid 59 59]);
    cropFaces{i} = trainingpatch;
  
end
%%


cropNonFaces = {};

for i = 1:3 %change to num_nonfaces-1
    
    nonFace2Crop = getfield(training_nonfaces_list(i),'name');
    photo = read_gray(nonFace2Crop);
    
    [h w]= size(photo);
    L = 60;

    % Crop 
    crop1 = photo(randi(h-L+1)+(0:L-1),randi(w-L+1)+(0:L-1));
    crop2 = photo(randi(h-L+1)+(0:L-1),randi(w-L+1)+(0:L-1));
    crop3 = photo(randi(h-L+1)+(0:L-1),randi(w-L+1)+(0:L-1));
    crop4 = photo(randi(h-L+1)+(0:L-1),randi(w-L+1)+(0:L-1));
    crop5 = photo(randi(h-L+1)+(0:L-1),randi(w-L+1)+(0:L-1));
    
    %figure(1); imshow(crop1,[0 255]);
   
    cropNonFaces{i,1} = crop1;
    cropNonFaces{i,2} = crop2;
    cropNonFaces{i,3} = crop3;
    cropNonFaces{i,4} = crop4;
    cropNonFaces{i,5} = crop5;
    
     
end



