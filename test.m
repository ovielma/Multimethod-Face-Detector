% test.m tests face detector using skin detection and classifier cascades
% using a classifier trained with AdaBoost and bootstrapping



%%
clear;
directories;

testing_nonfaces_path = [training_directory, 'test_nonfaces'];
testing_nonfaces_list = dir(testing_nonfaces_path);
testing_nonfaces_list = remove_directories_from_dir_list(testing_nonfaces_list);


load intergrals;
load training;
load boosted15;
load classifiers1550;


%TESTING FOR NONFACES IMAGES
%%
predicted = 0;
miss = 0;
total = 0;
%%
for i =1:1
    
    face2Test = getfield(testing_nonfaces_list(i),'name');
    photoT = read_gray(face2Test);
    
    result = apply_classifier_aux(photoT, boosted_classifier, weak_classifiers, [60 60]);
    class = max(max(result));
    total = total + class;
   
    if class <= 0
        predicted = predicted +1;
    else
        miss = miss +1;
    end
  
end
total = total / 36;

nonFaceAcc = (predicted/36) * 100;

%%
%TESTING CROPPED FACES IMAGES
testing_cropped_faces_path = [training_directory, 'test_cropped_faces'];
testing_cropped_faces_list = dir(testing_cropped_faces_path);
testing_cropped_faces_list = remove_directories_from_dir_list(testing_cropped_faces_list);



%TESTING FOR CROPPED FACES
%%
predicted = 0;
miss = 0;
total = 0;
%%
for i =1:770
    
    face2Test = getfield(testing_cropped_faces_list(i),'name');
    photoT = read_gray(face2Test);
    centroid = (size(photoT)/2)/2;
    photoT = imcrop(photoT, [centroid 59 59]);
   
    result = apply_classifier_aux(photoT, boosted_classifier, weak_classifiers, [60 60]);
    class = result(31,31);
    total = total + class;
    disp(class);
    
    if class > 0 
        predicted = predicted +1;
    else
        miss = miss +1;
    end
  
end

total = total / 770;
croppedFaceAcc = (predicted/770) * 100;


%%
%TESTING CROPPED FACES IMAGES
testing_faces_path = [training_directory, 'test_face_photos'];
testing_faces_list = dir(testing_faces_path);
testing_faces_list = remove_directories_from_dir_list(testing_faces_list);



%TESTING FOR FACES
%%
predicted = 0;
miss = 0;
total = 0;
%%
for i =1:20
    
    face2Test = getfield(testing_faces_list(i),'name');
    face2Test = double(imread(face2Test));
    photoT = imresize(face2Test, [60 60]);
    result = apply_classifier_aux(photoT, boosted_classifier, weak_classifiers, [60 60]);
    class = result(31,31);
    total = total + class;
 
    
    if class > -4
        predicted = predicted +1;
    else
        miss = miss +1;
    end
  
end

total = total / 20;
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
clear;
negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

%%

testImg = double(imread('clintonAD2505_468x448.jpg'));

result = detect_skin(testImg, positive_histogram,  negative_histogram);
figure (2); imshow(result > .6, []);
