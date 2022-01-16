from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
import glob
from sklearn.cluster import KMeans
from random import shuffle
import time


#This is training Data
path_aiplanes_train = "Images/airplanes_train/**/*.jpg"
path_faces_train = "Images/faces_train/**/*.jpg"
path_cars_train = "Images/cars_train/**/*.jpg"
path_bikes_train = "Images/bikes_train/**/*.jpg"
airplanes_train = glob.glob(path_aiplanes_train, recursive=True)
bikes_train = glob.glob(path_bikes_train, recursive=True)
faces_train = glob.glob(path_faces_train, recursive=True)
cars_train = glob.glob(path_cars_train, recursive=True)
training_list = airplanes_train + bikes_train + cars_train + faces_train


#This is Testing Data
path_aiplanes_test = "Images/airplanes_test/**/*.jpg"
path_faces_test = "Images/faces_test/**/*.jpg"
path_cars_test = "Images/cars_test/**/*.jpg"
path_bikes_test = "Images/bikes_test/**/*.jpg"
airplanes_test = glob.glob(path_aiplanes_test, recursive=True)
bikes_test = glob.glob(path_bikes_test, recursive=True)
faces_test = glob.glob(path_faces_test, recursive=True)
cars_test = glob.glob(path_cars_test, recursive=True)
testing_list = airplanes_test + cars_test + bikes_test + faces_test
shuffle(testing_list) #This is to shuffle testing data


if __name__ == "__main__":
    start_time = time.time()
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    features_train = []
    model.summary()


    #Creating Clusters
    for path in training_list:
        image_matrix = image.load_img(path, target_size=(224, 224))
        image_matrix = image.img_to_array(image_matrix)
        image_matrix = np.expand_dims(image_matrix, axis=0)
        image_matrix = preprocess_input(image_matrix)
        features_train.append(np.array(model.predict(image_matrix)).flatten())
    features_train = np.array(features_train)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(features_train)


    #Following is the Supervising of algrithm regarding the classes which correspond to different types of images given
    #Following classes are such that the no. of images that are firstly given are of class_airplane, then class_bike and so on...
    class_airplane = np.unique(kmeans.labels_[:30], return_counts=True)[0][np.unique(kmeans.labels_[:30], return_counts=True)[1] == np.max(np.unique(kmeans.labels_[:30], return_counts=True)[1])]
    class_bike = np.unique(kmeans.labels_[30:60], return_counts=True)[0][np.unique(kmeans.labels_[30:60], return_counts=True)[1] == np.max(np.unique(kmeans.labels_[30:60], return_counts=True)[1])]
    class_car = np.unique(kmeans.labels_[60:90], return_counts=True)[0][np.unique(kmeans.labels_[60:90], return_counts=True)[1] == np.max(np.unique(kmeans.labels_[60:90], return_counts=True)[1])]
    class_face = np.unique(kmeans.labels_[90:120], return_counts=True)[0][np.unique(kmeans.labels_[90:120], return_counts=True)[1] == np.max(np.unique(kmeans.labels_[90:120], return_counts=True)[1])]


    #Predicting the image
    positive_predictions = 0
    negative_predictions = 0
    for path in testing_list:
        image_matrix = image.load_img(path, target_size=(224, 224))
        image_matrix = image.img_to_array(image_matrix)
        image_matrix = np.expand_dims(image_matrix, axis=0)
        image_matrix = preprocess_input(image_matrix)
        prediction = kmeans.predict(np.array(model.predict(image_matrix)).flatten().reshape(1, -1))
        if prediction == class_airplane:
            if path.find("airplanes") != -1: #Finding the filename to consist airplanes as the algorithm has detected it as airplane
                positive_predictions += 1
            else:
                negative_predictions += 1
        elif prediction == class_car:
            if path.find("cars") != -1:  # Finding the filename to consist cars as the algorithm has detected it as airplane
                positive_predictions += 1
            else:
                negative_predictions += 1
        elif prediction == class_face:
            if path.find("faces") != -1:  # Finding the filename to consist faces as the algorithm has detected it as airplane
                positive_predictions += 1
            else:
                negative_predictions += 1
        elif prediction == class_bike:
            if path.find("bikes") != -1:  # Finding the filename to consist bikes as the algorithm has detected it as airplane
                positive_predictions += 1
            else:
                negative_predictions += 1
    print("Positive Predictions Made by the Algorithm:  ", positive_predictions)
    print("Negative Predictions Made by the Algorithm:  ", negative_predictions)
    print("Prediction Accuracy is:  ", (positive_predictions/(positive_predictions+negative_predictions))*100, "%")
    print("Time Required to Execute programme:  ", time.time() - start_time, " Seconds.")
