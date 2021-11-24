from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pickle
import numpy as np
from keras.models import load_model
from tensorflow import Graph
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Create your views here.
model_graph = Graph()

model=load_model('models\plant_disease_detection_MobileNet.h5')

# path='models/'
# filename = 'densenet.pkl'
# model = pickle.load(open(path+filename, 'rb'))

def index(request):
    context={'a':1}
    return render(request,'index.html',context)

img_height, img_width=224,224

def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x=x/255
    x=x.reshape(img_height, img_width,3)
    x = np.expand_dims(x, axis=0)
    # with model_graph.as_default():
    #     with tf_session.as_default():
    predi=model.predict(x)
    print(predi)
    classes_x=np.argmax(predi)
    
    print(classes_x)
    classes=['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',  'Blueberry_healthy', 'Cherry(including_sour)___Powdery_mildew','Cherry_(including_sour)__healthy','Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)__Common_rust', 'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)__healthy',  'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)',  'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight' , 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot',  'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite','Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']
    MaxPosition=np.argmax(predi)  
    prediction_label=prediction_label=classes[MaxPosition]
    print(prediction_label)

    context={'filePathName':filePathName,'predictedLabel':prediction_label}
    return render(request,'index.html',context)