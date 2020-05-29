# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 01:33:25 2019

@author: Nep2
"""

#Importing packages for training phase of the network
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
K.set_image_dim_ordering('tf') # tensorflow ordering -> (170[Images], 96, 96, 3)
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#Import the necessary packages for Testing images
from keras.models import load_model, Model
from keras.utils import plot_model

#Nessesary libraries in the system
import sys  
from imutils import paths
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")# set the matplotlib backend so figures can be saved in the background
import numpy as np
import random
import cv2
import os

#For Graphical User Interface libraries
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtWidgets import QFileDialog

#Accessing Python GUI Files 
from uiComponents.guiDesign import Ui_MainWindow

class MainForm(QMainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.userInterface = Ui_MainWindow()
        self.userInterface.setupUi(self) 
        
        #Training ui
        self.userInterface.pushButtonStopTraining.setEnabled(False)
        #Testing ui
        self.userInterface.pushButtonPredictImageFile.setEnabled(False) 
        self.userInterface.spinBoxFMNum.setEnabled(False)
        self.userInterface.spinBoxLayerNum.setEnabled(False)
        self.userInterface.pushButtonShowFM.setEnabled(False)
        self.userInterface.pushButtonShowFMs.setEnabled(False)
        self.userInterface.pushButtonShowModSum.setEnabled(False)
        self.userInterface.pushButtonShowFandB.setEnabled(False)
        
        #Testing Button Events
        self.userInterface.pushButtonLoadImageFile.clicked.connect(self.browse_image) 
        self.userInterface.pushButtonPredictImageFile.clicked.connect(self.predict_image) 
        self.userInterface.pushButtonShowModSum.clicked.connect(self.showModelSummary)
        self.userInterface.pushButtonShowFMs.clicked.connect(self.showFMs)
        self.userInterface.pushButtonShowFM.clicked.connect(self.showFM)
        self.userInterface.pushButtonShowFandB.clicked.connect(self.showFilAndBias)
        
        #Training Button Events
        self.userInterface.pushButtonLoadFileDirectoryOfImages.clicked.connect(self.browse_folder)
        self.userInterface.pushButtonStartTraining.clicked.connect(self.start_training)
        self.userInterface.pushButtonStopTraining.clicked.connect(self.stop_training)
        
        #Declaring global variables
        self.model = None
        self.layerNum = None
        self.testImage = None
        self.featureMaps = []
        self.noOfEpochs = 0
        self.batchSize = 0
        self.initialearnLRate = 0.0
        self.noOfTrainedImages = 0
        self.fileNameImageFile = ""
        self.fileNameDirectory = ""
        self.layerNum = 0
        self.mapNum = 0
        self.actualGrade = ""
 
    def stop_training(self):
        sys.exit(1)
        
    def browse_folder(self):
        self.fileNameDirectory = QFileDialog.getExistingDirectory(self, "Pick a folder: ")
        if self.fileNameDirectory:
            self.userInterface.textEditFileDirectoryOfImages.setText(self.fileNameDirectory)  
            QMessageBox.information(self, "Information: ", "Loaded an directory done successully!!") 
        else:
            QMessageBox.information(self, "Information: ", "Loaded an directory done unsuccessully!!") 

    def browse_image(self):
        self.fileNameImageFile, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select an image file: ", "", "Image Files (*.jpg *jpeg)") # Ask for image file
       
        if self.fileNameImageFile: 
            pixmap = QtGui.QPixmap(self.fileNameImageFile) 
            pixmap = pixmap.scaled(self.userInterface.labelImageContainer.height()+500, self.userInterface.labelImageContainer.width()+500, QtCore.Qt.KeepAspectRatio) 
            self.userInterface.labelImageContainer.setPixmap(pixmap) 
            self.userInterface.labelImageContainer.setAlignment(QtCore.Qt.AlignCenter) 
            self.userInterface.labelGetFilenameLocation.setText(self.fileNameImageFile)
            self.userInterface.pushButtonPredictImageFile.setEnabled(True) 
            QMessageBox.information(self, "Information: ", "Loaded an image file done successully!!") 
        else:
            QMessageBox.warning(self, "Warning: ", "Loaded an image file done unsuccessully!!")
            
            
    def predict_image(self):
        def classifyAbacaFiberImage(filename):
            self.testImage = cv2.imread(filename)
            self.testImage = cv2.resize(self.testImage, (112, 112))            
            self.testImage = self.testImage.astype("float") / 255.0
            self.testImage = img_to_array(self.testImage)
            self.testImage = np.expand_dims(self.testImage, axis=0)
            
            print("Loading the network...")
            self.model = load_model("trainedModel/abacaFiberBestModel.model")
            lb = pickle.loads(open("trainedModel/abacafiberlb.pickle", "rb").read())
            
            plot_model(self.model, 'modelSummary/modelSummary.png', show_layer_names=True, show_shapes=True)          
            
            # Classify the image 
            print("Classifying the Image load...")
            probasPredict = self.model.predict(self.testImage)[0]
            
#            print(probasPredict)
            
            return lb.classes_, probasPredict
    
        if not self.fileNameImageFile:
            QMessageBox.warning(self, "Warning: ", "Prediction of abaca fiber image done unsuccessully!!\nPlease load an abaca fiber image file.") 
        else:
            QMessageBox.information(self, "Information: ", "Loaded the model, this process takes a minute!! Please wait...") 
            self.userInterface.pushButtonPredictImageFile.setEnabled(False)
            labelGrades, percentages = classifyAbacaFiberImage(filename = self.fileNameImageFile) 
            self.userInterface.labelGradePrediction1.setText(labelGrades[0]+": ")
            self.userInterface.labelGradePercentage1.setText("{:.2f}%".format(percentages[0] * 100))
            self.userInterface.labelGradePrediction2.setText(labelGrades[1]+": ")
            self.userInterface.labelGradePercentage2.setText("{:.2f}%".format(percentages[1] * 100))
            self.userInterface.labelGradePrediction3.setText(labelGrades[2]+": ")
            self.userInterface.labelGradePercentage3.setText("{:.2f}%".format(percentages[2] * 100))
            self.userInterface.labelGradePrediction4.setText(labelGrades[3]+": ")
            self.userInterface.labelGradePercentage4.setText("{:.2f}%".format(percentages[3] * 100))
            self.userInterface.labelGradePrediction5.setText(labelGrades[4]+": ")
            self.userInterface.labelGradePercentage5.setText("{:.2f}%".format(percentages[4] * 100))
            self.userInterface.labelGradePrediction6.setText(labelGrades[5]+": ")
            self.userInterface.labelGradePercentage6.setText("{:.2f}%".format(percentages[5] * 100))
            self.userInterface.labelGradePrediction7.setText(labelGrades[6]+": ")
            self.userInterface.labelGradePercentage7.setText("{:.2f}%".format(percentages[6] * 100))
            
            sepImageFile = os.path.normpath(self.fileNameImageFile)
            self.actualGrade = sepImageFile.split(os.path.sep)  
            self.userInterface.labelGradeActual.setText(self.actualGrade[-2]) 
            
            self.userInterface.spinBoxLayerNum.setEnabled(True)
            self.userInterface.pushButtonShowFMs.setEnabled(True)
            self.userInterface.pushButtonShowModSum.setEnabled(True)
            
            self.userInterface.pushButtonShowFandB.setEnabled(True) #show filters button
           
            QMessageBox.information(self, "Information: ", "Prediction of abaca fiber image done successully!!") 
        
    def showModelSummary(self):
        outputImg = QtGui.QPixmap("modelSummary/modelSummary.png")  
        self.userInterface.labelImageContainer.setScaledContents(False)
        self.userInterface.labelImageContainer.setPixmap(outputImg)
            
    def showFMs(self): 
        def drawFMs(model):
            self.featuremaps = model.predict(self.testImage)
            self.featuremaps = np.squeeze(self.featuremaps[int(self.layerNum)]) #must know
            self.featuremaps = np.moveaxis(self.featuremaps, 2, 0) #must know
            
            figDime = int(np.ceil(np.sqrt(self.featuremaps.shape[0]))) #must know

            figure = plt.figure(figsize=(40, 40))
            for i in range(self.featuremaps.shape[0]):
                ax = figure.add_subplot(figDime, figDime, i+1)
                ax.imshow(self.featuremaps[i], cmap='viridis')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
            
            plt.savefig("featureMaps/featuremaps@Layer{}".format(self.layerNum) + '.png') 
                        
        self.layerNum = self.userInterface.spinBoxLayerNum.value()
        boolShowFMs = True
        
        if(boolShowFMs):
            layers = np.arange(27)
            outputs = [self.model.layers[i].output for i in layers]
            model = Model(inputs=self.model.inputs, outputs=outputs)
            
          
           
            drawFMs(model)
            boolShowFMs = False
        else:
            if(int(self.layerNum) < 22):
                drawFMs()
            else:
                print(model.output.shape)
        
        outputImg = QtGui.QPixmap("featureMaps/featuremaps@Layer{}".format(self.layerNum) + '.png')  
        self.userInterface.labelImageContainer.setScaledContents(True)
        self.userInterface.labelImageContainer.setPixmap(outputImg) 
        
        self.userInterface.spinBoxFMNum.setEnabled(True)
        self.userInterface.pushButtonShowFM.setEnabled(True) 
        QMessageBox.information(self, "Information: ", "Generating a feature maps in a model at layer "+str(self.layerNum)+" \n with a dimension of "
                                +str(self.featuremaps.shape)+" done successfully !!")
            
        self.userInterface.spinBoxFMNum.setEnabled(True)
        self.userInterface.pushButtonShowFM.setEnabled(True)
        
    def showFM(self):  
        self.mapNum = self.userInterface.spinBoxFMNum.value()
        figDim = 1
        fig = plt.figure(figsize=(40, 40))  
        ax = fig.add_subplot(figDim, figDim, 1)
        ax.imshow(self.featuremaps[int(self.mapNum)-1], cmap='viridis')
        plt.savefig("featureMaps/specificFeaturemap@Layer{}".format(self.layerNum) + 'FeaturemapNumber{}'.format(self.mapNum)+ '.png')
        specificFeatureMap = QtGui.QPixmap("featureMaps/specificFeaturemap@Layer{}".format(self.layerNum) + 'FeaturemapNumber{}'.format(self.mapNum)+ '.png')
        
        self.userInterface.labelImageContainer.setScaledContents(True)
        self.userInterface.labelImageContainer.setPixmap(specificFeatureMap)
        QMessageBox.information(self, "Information: ", "Generating a spefic feature map number "+str(self.mapNum)+" in a model at layer "+str(self.layerNum)+
                                " with a dimension of "+str(self.featuremaps.shape)+" done successfully !!") 
         
    def showFilAndBias(self):
        self.layerNum = self.userInterface.spinBoxLayerNum.value()
        self.mapNum = self.userInterface.spinBoxFMNum.value()
        
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            print(i, layer.name, layer.output.shape)
        
        if('conv' in self.model.layers[int(self.layerNum)].name):
            convFilters, convBiases = self.model.layers[int(self.layerNum)].get_weights()
            if(int(self.layerNum) == 0):
                print("Specific filter at layer number: {}".format(int(self.layerNum))+" filter number {}".format(int(self.mapNum)))
                channelFilter = convFilters[:, :, :, self.mapNum]
                for j in range(3):
                    print(channelFilter[:, :, j])
            else:
                print("Specific filter at layer number: {}".format(int(self.layerNum))+" filter number {}".format(int(self.mapNum)))
                print(convFilters[:, :, :, self.mapNum])
        
            print("Specific bias at layer number: {}".format(int(self.layerNum))+" bias number {}".format(int(self.mapNum)))
            print(convBiases[self.mapNum])
        else:
            QMessageBox.warning(self, "Warning: ", "Specified layer does not convolutional layer!!")         
       
    def start_training(self):
        def smallVggModel(width, height, nChannels, abacaGrades):
            genModel = Sequential()
            inputShape = (height, 
                          width, 
                          nChannels)
            
            genModel.add(Conv2D(32, (3, 3),     
                                padding='same', 
                                activation='relu',
                                input_shape=inputShape)) 
            genModel.add(BatchNormalization())
            genModel.add(MaxPooling2D(pool_size=(3, 3)))
            genModel.add(Dropout(0.25))
        
            genModel.add(Conv2D(64, (3, 3), 
                         padding='same', 
                         activation='relu'))
            genModel.add(BatchNormalization())
            genModel.add(Conv2D(64, (3, 3),
                         padding='same', 
                         activation='relu'))
            genModel.add(BatchNormalization())
            genModel.add(MaxPooling2D(pool_size=(2, 2)))
            genModel.add(Dropout(0.25))
        
            genModel.add(Conv2D(128, (3, 3), 
                         padding='same', 
                         activation='relu'))
            genModel.add(BatchNormalization())
            genModel.add(Conv2D(128, (3, 3), 
                         padding='same', 
                         activation='relu'))
            genModel.add(BatchNormalization())
            genModel.add(MaxPooling2D(pool_size=(2, 2)))
            genModel.add(Dropout(0.25))
        
            genModel.add(Flatten())
            genModel.add(Dense(1024))
            genModel.add(Activation("relu")) 
            genModel.add(BatchNormalization())
        
            genModel.add(Dense(abacaGrades))
            genModel.add(Activation("softmax"))
        
            return genModel

            #Implentation of VGGNet-16 CNN Architecture
            """
            # initialize the model along with the input shape to be
            # "channels last" and the channels dimension itself
            model = Sequential()
            
            model.add(Conv2D(64, (3, 3), padding="same",
                input_shape=inputShape))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(Conv2D(64, (3,3), padding="same"))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            #model.add(Dropout(0.25))
            
            model.add(Conv2D(128, (3, 3), padding="same"))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(Conv2D(128, (3,3), padding="same"))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            #model.add(Dropout(0.25))

            model.add(Conv2D(256, (3, 3), padding="same"))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(Conv2D(256, (3,3), padding="same"))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            #model.add(Dropout(0.25))
            
            model.add(Conv2D(512, (3, 3), padding="same"))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(Conv2D(512, (3,3), padding="same"))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            #model.add(Dropout(0.25))
            
            model.add(Conv2D(512, (3, 3), padding="same"))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(Conv2D(512, (3,3), padding="same"))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            #model.add(Dropout(0.25))
            
            model.add(Flatten())
            model.add(Dense(4096)) # Fully Connected Layer
            model.add(Activation("relu")) # Rectified Function
            # model.add(BatchNormalization()) # Batch Normalization
            model.add(Dense(4096)) # Fully Connected Layer
            model.add(Activation("relu")) # Rectified Function
            #model.add(BatchNormalization()) # Batch Normalization

            # softmax classifier
            model.add(Dense(abacaGrades))
            model.add(Activation("softmax"))

            # return the constructed network architecture
            return model
            """
        
        def appendTextBox(trainStatus): 
            self.userInterface.textEdit.append('%s' % trainStatus) 
            QApplication.processEvents()
        
        def trainCNN(fileDir):
            
            self.noOfEpochs = self.userInterface.textEditNoOfEpochs.toPlainText()
            self.batchSize = self.userInterface.textEditNoOfBatchSize.toPlainText()
            self.randomSeed = self.userInterface.textNoOfRandomSeed.toPlainText()
            self.initialLearningRate = self.userInterface.textEditIniatialLearningRate.toPlainText()
            self.imgDime = (112, 112, 3)
            
            appendTextBox('Training Parameters') 
            appendTextBox('Number of Epochs: %s' % self.noOfEpochs) 
            appendTextBox('Batch Size: %s' % self.batchSize) 
            appendTextBox('Random Seed: %s' % self.randomSeed) 
            appendTextBox('Initial Learning Rate: %s' % self.initialLearningRate) 
            appendTextBox('Image Dimension: %s' % str(self.imgDime))  
            
            self.userInterface.textEditNoOfEpochs.setEnabled(False)
            self.userInterface.textEditNoOfBatchSize.setEnabled(False)
            self.userInterface.textNoOfRandomSeed.setEnabled(False)
            
            data = []
            labels = []
            
            print("Loading dataset...")
            appendTextBox('Loading abaca fiber images...')  
            imageDirs = sorted(list(paths.list_images(fileDir)))
            random.seed(int(self.randomSeed))
            random.shuffle(imageDirs) 
        
            global noOfTrainedImages
            
            for imageDir in imageDirs:
                image = cv2.imread(imageDir)
                image = cv2.resize(image, (self.imgDime[1], self.imgDime[0]))
                image = img_to_array(image)
                data.append(image) 
                self.noOfTrainedImages+=1;
                appendTextBox(imageDir) 
             
                label = imageDir.split(os.path.sep)[-2] 
                labels.append(label) 
                
            data = np.array(data, dtype="float") / 255.0
            labels = np.array(labels)
            self.userInterface.textNoOfImages.setText(str(self.noOfTrainedImages))

            lb = LabelBinarizer()
            labels = lb.fit_transform(labels)
          
            (trainX, testX, trainY, testY) = train_test_split(data,
                labels, test_size=0.0, random_state=self.randomSeed)
            
            dataAug = ImageDataGenerator(rotation_range=25, 
                                         width_shift_range=0.1,
                                         height_shift_range=0.1, 
                                         shear_range=0.2, 
                                         zoom_range=0.2,
                                         horizontal_flip=True, 
                                         fill_mode="nearest")

            appendTextBox('Compiling the model...') 
            model = smallVggModel(width=self.imgDime[0], 
                                  height=self.imgDime[1],
                                  nChannels=self.imgDime[2], 
                                  abacaGrades=len(lb.classes_))
            
            trainingOptimizer = Adam(lr=float(self.initialLearningRate), 
                           decay=float(self.initialLearningRate)/int(self.noOfEpochs))
            
            model.compile(loss="categorical_crossentropy", optimizer=trainingOptimizer,
                metrics=["accuracy"])
            
            checkpointer = ModelCheckpoint(filepath="trainedModel/abacaFiberBestModel.model", 
                                           monitor = 'loss',
                                           verbose=1, 
                                           save_best_only=True) 
            
            appendTextBox('Training the network...') 
            appendTextBox('Wait for a moment may take time to responds...') 
            trainingHistory = model.fit_generator(dataAug.flow(trainX, 
                                                               trainY, 
                                                               batch_size=int(self.batchSize)),
                                                  validation_data=None,
                                                  steps_per_epoch=len(trainX) // int(self.batchSize),
                                                  epochs=int(self.noOfEpochs), 
                                                  callbacks=[checkpointer], 
                                                  verbose=1)
            
            
            appendTextBox('Saving labels...') 
            f = open("trainedModel/abacafiberlb.pickle", "wb")
            f.write(pickle.dumps(lb))   
            f.close()
            
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, self.noOfEpochs), trainingHistory.history["loss"], label="trainLoss")
            plt.plot(np.arange(0, self.noOfEpochs), trainingHistory.history["acc"], label="trainAcc")      
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="upper right")
            plt.savefig("trainedModel/TrainPlotAcc&Loss.png")  
            
            self.userInterface.textEditNoOfEpochs.setEnabled(True)
            self.userInterface.textEditNoOfBatchSize.setEnabled(True)
            self.userInterface.textNoOfRandomSeed.setEnabled(True)
                
                
        if Path(self.fileNameDirectory).rglob('*\*.jpg') and self.fileNameDirectory:
            self.userInterface.pushButtonStopTraining.setEnabled(True)
            self.userInterface.textEdit.setEnabled(True)
            appendTextBox('Using tensorflow backend')
            trainCNN(self.fileNameDirectory)
            appendTextBox('Training the network done...')
            QMessageBox.information(self, "Information: ", "Training the CNN network done!!") 
        else:
            QMessageBox.warning(self, "Warning: ", "File directory of the images to train contains nothing or empty!!") 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainForm()
    window.show()
    sys.exit(app.exec_())



