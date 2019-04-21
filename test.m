% test.m tests face detector using skin detection and classifier cascades
% using a classifier trained with AdaBoost and bootstrapping


%TESTING CROPPED IMAGES
testing_faces_path = [training_directory, 'training_nonfaces'];
testing_faces_list = dir(testing_faces_path);
testing_faces_list = remove_directories_from_dir_list(testing_faces_list);


%%

directories;

testing_nonfaces_path = [training_directory, 'test_nonfaces'];
testing_nonfaces_list = dir(testing_nonfaces_path);
testing_nonfaces_list = remove_directories_from_dir_list(testing_nonfaces_list);


load preprocessData
load classifiers1150
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


%% 






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





