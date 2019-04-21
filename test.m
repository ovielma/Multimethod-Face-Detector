% test.m tests face detector using skin detection and classifier cascades
% using a classifier trained with AdaBoost and bootstrapping



%%

directories;

testing_nonfaces_path = [training_directory, 'test_nonfaces'];
testing_nonfaces_list = dir(testing_nonfaces_path);
testing_nonfaces_list = remove_directories_from_dir_list(testing_nonfaces_list);


load intergrals;
load training;
load classifiers1550
load boosted15;


%TESTING FOR NONFACES IMAGES
%%
predicted = 0;
miss = 0;

%%
for i =1:36
    
    face2Test = getfield(testing_nonfaces_list(i),'name');
    photo = read_gray(face2Test);
    photoT = imresize(photo, [60 60]);
    result = apply_classifier_aux(photoT, boosted_classifier, weak_classifiers, [60 60]);
    class = max(result(:));
    if class <= 0
        predicted = predicted +1;
    else
        miss = miss +1;
    end
  
end


nonFaceAcc = (predicted/37) * 100;

%%
%TESTING CROPPED FACES IMAGES
testing_cropped_faces_path = [training_directory, 'test_cropped_faces'];
testing_cropped_faces_list = dir(testing_cropped_faces_path);
testing_cropped_faces_list = remove_directories_from_dir_list(testing_cropped_faces_list);



%TESTING FOR CROPPED FACES
%%
predicted = 0;
miss = 0;

%%
for i =1:770
    
    face2Test = getfield(testing_cropped_faces_list(i),'name');
    photo = read_gray(face2Test);
    photoT = imresize(photo, [60 60]);
    result = apply_classifier_aux(photoT, boosted_classifier, weak_classifiers, [60 60]);
    class = max(result(:));
    if class > 0
        predicted = predicted +1;
    else
        miss = miss +1;
    end
  
end

croppedFaceAcc = (predicted/770) * 100;


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  Skin Detection
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read histograms
clear;
negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

%%

testImg = double(imread('clintonAD2505_468x448.jpg'));

result = detect_skin(testImg, positive_histogram,  negative_histogram);
figure (2); imshow(result > .6, []);
