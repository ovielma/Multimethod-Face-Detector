
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
faceIntegrals = cell(1,3047);

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
    %figure(i); imshow(B, []);
    faceIntegrals{1, i} = B;
  
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

%%
%transform integral NonFaces to 1 x 2600
NonFacesintegral = {};
for i =1:num_nonfaces-1
  for j =1:20
      
    NonFacesintegral{end+1} = integralNonFaces{i,j};  
  
  end
    
end

%%
face_horizontal =60;
face_vertical = 60;

%generate 1000 random classifiers  
number = 1000;
weak_classifiers = cell(1,number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_horizontal, face_vertical);
end

%%

example_number = (num_faces-1) + (numOfPatches * (num_nonfaces-1));
labels = zeros(example_number, 1);
labels (1:num_faces-1) = 1;
labels((num_faces):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);

%convert cell array to matrix
faceIntegralArray = zeros(face_vertical,face_horizontal,num_faces-1);
for i = 1: num_faces-1
    faceIntegralArray(:,:,i) = cell2mat(faceIntegrals(i));
end
examples (:, :, 1:num_faces-1) = faceIntegralArray;

%convert cell array to matrix
NonfaceIntegralArray = zeros(face_vertical,face_horizontal,(numOfPatches*130));
for i = 1: num_nonfaces-1
    NonfaceIntegralArray(:,:,i) = cell2mat(NonFacesintegral(i));
end

examples(:, :, num_faces:example_number) = NonfaceIntegralArray;
classifier_number = numel(weak_classifiers);
responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example)
end

%%

boosted_classifier = AdaBoost(responses, labels, 15);
%%
% load a photograph
photo = read_gray('DSC01181.JPG');

% rotate the photograph to make faces more upright (we 
% are cheating a bit, to save time compared to searching
% over multiple rotations).
photo2 = imresize(photo, [60 60]);
figure(1); imshow(photo2, []);

% w1 and w2 are the locations of the faces, according to me.
% Used just for bookkeeping.
%w1 = photo2(40:87, 75:113);
%w2 = photo2(100:130, 47:71);

%%
tic; result = apply_classifier_aux(photo2, boosted_classifier, weak_classifiers, [60 60]); toc
figure(2); imshow(result, []);
figure(3); imshow(max((result > 4) * 255, photo2 * 0.5), [])

%%

tic; [result, boxes] = boosted_detector_demo(photo2, 1, boosted_classifier, weak_classifiers, [60, 60], 2); toc
figure(2); imshow(result, []);



