import Model
from CubeOCR import CubeProcess

def SealRecog(imgpath,model):
    category,img = model.predict(imgpath)
    if category =='正方形':
        result = CubeProcess(img)
    elif category =='椭圆形':
        pass
    else :
        pass
    return result

