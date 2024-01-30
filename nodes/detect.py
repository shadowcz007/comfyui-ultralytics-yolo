import os,sys
import folder_paths
from PIL import Image
import numpy as np

import torch


# print('#######s',os.path.join(__file__,'../'))

sys.path.append(os.path.join(__file__,'../../'))
                
from ultralytics import YOLO,settings

# Update a setting
settings.update({'weights_dir':os.path.join(folder_paths.models_dir,'ultralytics')})



def get_files_with_extension(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_name = os.path.splitext(file)[0]
                file_list.append(file_name)
    return file_list


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def createMask(image,x,y,w,h):
    mask = Image.new("L", image.size)
    pixels = mask.load()
    # 遍历指定区域的像素，将其设置为黑色（0 表示黑色）
    for i in range(int(x), int(x + w)):
        for j in range(int(y), int(y + h)):
            pixels[i, j] = 255
    # mask.save("mask.png")
    return mask

class detectNode:
  
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "model":(get_files_with_extension(os.path.join(folder_paths.models_dir,'ultralytics'),'.pt'),),
                             },

                 "optional":{ 
                    "target_label": ("STRING",{"forceInput": True,"default": "","multiline": False,"dynamicPrompts": False}), 
                }

                }
    
    RETURN_TYPES = ("MASK","STRING","_GRID",)
    RETURN_NAMES = ("masks","labels","grids",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/YOLOv8"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,True,True,)
  
    def run(self,image,model,target_label=""):
        # print(model)
        target_labels=target_label.split('\n')
        target_labels=[t.strip() for t in target_labels if t.strip()!='']

        model = YOLO(model+'.pt')  # load an official model. model(image, conf=confidence, device=device)

        image=tensor2pil(image)
        image=image.convert('RGB')
        images=[image]

        # Run batched inference on a list of images
        results = model(images)  # return a list of Results objects
        
        masks=[]
        names=[]
        grids=[]
        # Process results list
        for i in range(len(results)):
            result=results[i]
            img=images[i]
            boxes = result.boxes  # Boxes object for bbox outputs
            bb=boxes.xyxy.cpu().numpy()
       
            for j in range(len(bb)):
                name=result.names[boxes[j].cls.item()]
                # 判断是否是目标label
                is_target=True
                if len(target_labels)>0:
                    is_target=False
                    for t in target_labels:
                        if t==name:
                            is_target=True

                if is_target==True:
                    b=bb[j]
                    x,y,xw,yh=b
                    w=xw-x
                    h=yh-y
                    mask=createMask(img,x,y,w,h)
                    mask=pil2tensor(mask)
                    masks.append(mask)

                    names.append(name)

                    grids.append((x,y,w,h))

        if len(masks)==0:
            # 创建一个黑色图
            mask = Image.new("L", image.size)
            mask=pil2tensor(mask)
            masks.append(mask)
            grids.append((0,0,mask.width,mask.height))
            names.append(['-'])
            # masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Probs object for classification outputs
            # print(result)

        return (masks,names,grids,)