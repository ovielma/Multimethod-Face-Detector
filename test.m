% test.m tests face detector using skin detection and classifier cascades
% using a classifier trained with AdaBoost and bootstrapping



%%
clear;
directories;

testing_nonfaces_path = [training_directory, 'test_nonfaces'];
testing_nonfaces_list = dir(testing_nonfaces_path);
testing_nonfaces_list = remove_directories_from_dir_list(testing_nonfaces_list);

num_testing_nonfaces = size(testing_nonfaces_list, 1);


load intergrals;
load training;
load boosted15;
load classifiers1550;


%TESTING FOR NONFACES IMAGES
%%

threshold = 5;

predicted = 0;
miss = 0;

for i =1:num_testing_nonfaces
    
    face2Test = getfield(testing_nonfaces_list(i),'name');
    photoT = read_gray(face2Test);
    
    %result = apply_classifier_aux(photoT, boosted_classifier, weak_classifiers, [60 60]);
    result = boosted_multiscale_search(photoT, 3, boosted_classifier, weak_classifiers, ...
                              [60, 60]);
    %figure(i); imshow(result);
    class = max(max(result));
   
    if class <= threshold
        predicted = predicted +1;
    else
        miss = miss +1;
    end
  
end

nonFaceAcc = (predicted/num_testing_nonfaces) * 100;

%%
%TESTING CROPPED FACES IMAGES
testing_cropped_faces_path = [training_directory, 'test_cropped_faces'];
testing_cropped_faces_list = dir(testing_cropped_faces_path);
testing_cropped_faces_list = remove_directories_from_dir_list(testing_cropped_faces_list);

num_testing_croppedfaces = size(testing_cropped_faces_list, 1);



%TESTING FOR CROPPED FACES
%%
predicted = 0;
miss = 0;

for i =1:num_testing_croppedfaces
    
    face2Test = getfield(testing_cropped_faces_list(i),'name');
    photoT = read_gray(face2Test);
    centroid = (size(photoT)/2)/2;
    photoT = imcrop(photoT, [centroid 59 59]);
   
    result = apply_classifier_aux(photoT, boosted_classifier, weak_classifiers, [60 60]);
    class = result(31,31);
    
    
    if class > threshold 
        predicted = predicted +1;
    else
        miss = miss +1;
    end
  
end

croppedFaceAcc = (predicted/num_testing_croppedfaces) * 100;


%%
%TESTING CROPPED FACES IMAGES
testing_faces_path = [training_directory, 'test_face_photos'];
testing_faces_list = dir(testing_faces_path);
testing_faces_list = remove_directories_from_dir_list(testing_faces_list);

% size of testing test face list
num_testing_faces = size(testing_faces_list, 1);


%TESTING FOR FACES
%%
predicted = 0;
miss = 0;
total = 0;
%%
for i =1:20
    
    face2Test = getfield(testing_faces_list(i),'name');
    photoT = read_gray(face2Test);
    result = apply_classifier_aux(photoT, boosted_classifier, weak_classifiers, [60 60]);
    class = max(max(result));
 
    
    if class > threshold
        predicted = predicted +1;
    else
        miss = miss +1;
    end
  
end

FaceAcc = (predicted/20) * 100;


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  Skin Detection
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read histograms
negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

%%

for i =1:1 %num_testing_faces-1
    test_img = getfield(testing_faces_list(i),'name');
    test_img = double(imread(test_img));
    %test_img = double(imread('DSC04810.JPG'));
    
    % check if image is rgb and run skin detector if true
    if(size(test_img, 3) == 3)
        result_on_skin = detect_skin(test_img, positive_histogram,  negative_histogram);
        %figure (i); imshow(result_on_skin > .6, []);
        % run classifier after skin detection
        result = apply_classifier_aux(result_on_skin, boosted_classifier, weak_classifiers, [60 60]);
        figure(i); imshow(result, []);
        %[result, boxes] = boosted_detector_demo(test_img, 1, boosted_classifier, weak_classifiers, [60, 60], 2);
        %figure(i); imshow(result, []);
        
    end
      
end
