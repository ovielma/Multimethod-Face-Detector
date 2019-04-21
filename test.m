% test.m tests face detector using skin detection and classifier cascades
% using a classifier trained with AdaBoost and bootstrapping

% load all data with code below
% load preprocessData
% load classifiers1200
% load boosted15

%TESTING CROPPED IMAGES
testing_faces_path = [training_directory, 'training_nonfaces'];
testing_faces_list = dir(testing_faces_path);
testing_faces_list = remove_directories_from_dir_list(testing_faces_list);


%%
predicted = 0;
miss = 0;

%%
for i =1:130
    
    face2Test = getfield(testing_faces_list(i),'name');
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





