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
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_set = valid_datagen.flow_from_directory('models/valid',
                                            target_size=(224, 224),
                                            batch_size=50,
                                            class_mode='categorical')
model=load_model('models\plant_disease_detection_MobileNet.h5')
def get_class_string_from_index(index):
   for class_string, class_index in valid_set.class_indices.items():
      if class_index == index:
         return class_string
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
    # classes=['Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Corn_(maize)__Northern_Leaf_Blight', 'Soybean_healthy', 'Apple_Cedar_apple_rust', 'Tomato_Leaf_Mold', 'Tomato_Late_blight', 'Cherry(including_sour)__Powdery_mildew', 'Grape_Esca(Black_Measles)', 'Squash__Powdery_mildew', 'Apple_healthy', 'Tomato_Early_blight', 'Orange_Haunglongbing(Citrus_greening)', 'Corn_(maize)__healthy', 'Potato_healthy', 'Peach_Bacterial_spot', 'Tomato_Target_Spot', 'Tomato_Bacterial_spot', 'Potato_Late_blight', 'Blueberry_healthy', 'Pepper,_bell_Bacterial_spot', 'Apple_Black_rot', 'Pepper,_bell_healthy', 'Apple_Apple_scab', 'Peach_healthy', 'Strawberry_healthy', 'Tomato_Septoria_leaf_spot', 'Tomato_Tomato_mosaic_virus', 'Potato_Early_blight', 'Cherry(including_sour)__healthy', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Strawberry_Leaf_scorch', 'Tomato_healthy', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Grape_healthy', 'Raspberry_healthy', 'Corn(maize)__Common_rust', 'Grape___Black_rot']
    # MaxPosition=np.argmax(predi)  
    prediction_label=get_class_string_from_index(classes_x)
    print(prediction_label)

    context={'filePathName':filePathName,'predictedLabel':prediction_label}
    return render(request,'index.html',context)