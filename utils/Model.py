from . import network
import cv2
import time
import torch
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import transforms




LabelDict = {'0':'圆形','1':'椭圆','2':'正方形'}
SPPDict1 = r'ModelDict/c3,c96,c256,c384,c384-256,s2,l1280,ac100SPP.pkl'
SPPDict2 = r'ModelDict/c3,c32,c64,c128,c128-192,s2,l960,ac100SPP.pkl'
SPPDict3 = r'ModelDict/c3,c32,c64,c128-192,s2,960,ac100SPP.pkl'
class Model():
    def __init__(self,choice = 2):
        if choice == 0:
            model = network.SPP_Net1()
            Dict = SPPDict1
        elif choice == 1:
            model = network.SPP_Net2()
            Dict = SPPDict2
        else:
            model = network.SPP_Net3()
            Dict = SPPDict3
        if self.Device()=="cuda":
            model.load_state_dict(torch.load(Dict))
        else :
            model.load_state_dict(torch.load(Dict, map_location=torch.device('cpu')))
        # print("Model loaded!")
        self.model = model
    def Device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    def predict(self,ImgPath):
        Tensor,img = self.__img2tensor(ImgPath)
        with torch.no_grad():
            pred = self.model(Tensor)
        result = self.__arg2label(pred.argmax())
        # print(result)
        return result,img
    def __img2tensor(self,ImgPath):
        OImg = cv2.imread(ImgPath) #origin图片
        Transfm = Compose([transforms.ToTensor()])
        Img = Transfm(OImg)
        Img = Img.unsqueeze(0)
        return Img,OImg
    def __arg2label(self,arg):
        return LabelDict[str(arg.item())]

