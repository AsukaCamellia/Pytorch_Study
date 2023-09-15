import torch
import numpy as np
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {'1':0,'100':1}

class RMBdataset(Dataset):

    def __init__(self,data_dir,transform = None):
        self.label_name = {'1':0,'100':1}
        self.data_info = self.get_img_info(self,data_dir)
        self.transform = transform

    def __getitem__(self, index):
        img_path,label = self.data_info[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.data_info)
    
    @staticmethod
    def get_img_info(self,data_dir):
        data_info = list()
        for root,dirs,_ in os.walk(data_dir):

            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root,sub_dir))
                img_names = list(filter(lambda x:x.endswith('.jpg'),img_names))

                for i in range(len(img_names)):
                    img_name = img_names[i]
                    img_path = os.path.join(root,sub_dir,img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((img_path,int(label)))

        return data_info
    

