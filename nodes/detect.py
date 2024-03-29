import os,sys
import folder_paths
from PIL import Image
import numpy as np

import torch

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# print('#######s',os.path.join(__file__,'../'))

sys.path.append(os.path.join(__file__,'../../'))
                
from ultralytics import YOLO,settings,YOLOWorld
# from ultralytics import YOLOWorld

# Update a setting
settings.update({'weights_dir':os.path.join(folder_paths.models_dir,'ultralytics')})


# def get_files_with_extension(directory, extension):
#     file_list = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(extension):
#                 file_name = os.path.splitext(file)[0]
#                 file_list.append(file_name)
#     return file_list
def get_files_with_extension(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                file_name = os.path.relpath(file_path, directory)
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
            "confidence":("FLOAT", {"default": 0.1,
                                                                "min": 0.0,
                                                                "max": 1,
                                                                "step":0.01,
                                                                "display": "number"}),
            "model":(get_files_with_extension(os.path.join(folder_paths.models_dir,'ultralytics'),'.pt'),),
            "type":(["YOLO-World","YOLOv8"],),
                             },
                 "optional":{ 
                    "target_label": ("STRING",{"forceInput": True,"default": "","multiline": False,"dynamicPrompts": False}), 
                    "debug":(["on","off"],),
                }

                }
    
    RETURN_TYPES = ("MASK","STRING","_GRID","IMAGE",)
    RETURN_NAMES = ("masks","labels","grids","image",)

    FUNCTION = "run"

    CATEGORY = "♾️Mixlab/Mask"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,True,True,True,)
  
    def run(self,image,confidence,model,type="YOLO-World",target_label="",debug="on"):
        # print(model)
        target_labels=target_label.split('\n')
        target_labels=[t.strip() for t in target_labels if t.strip()!='']

        if type=="YOLO-World":
            model = YOLOWorld(model+'.pt')
        else:
            model = YOLO(model+'.pt')  # load an official model. model(image, conf=confidence, device=device)

        image=tensor2pil(image)
        image=image.convert('RGB')
        images=[image]

        # Run batched inference on a list of images
        results = model(images)  # return a list of Results objects
        
        masks=[]
        names=[]
        grids=[]
        images_debug=[]
        # Process results list
        for i in range(len(results)):
            result=results[i]
            img=images[i]
            boxes = result.boxes  # Boxes object for bbox outputs
            bb=boxes.xyxy.cpu().numpy()
            confs=boxes.conf.cpu().numpy()

            # Plot results image
            if debug=='on':
                im_bgr = result.plot()  # BGR-order numpy array
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                # im = result.plot(pil=True)
                images_debug.append(pil2tensor(im_rgb))
       
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
                    conf=confs[j]
                    if debug=='on':
                        print('#confidence',name,conf)
                    if conf >= confidence:
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
            grids.append((0,0,image.size[0],image.size[1]))
            names.append(['-'])
            # masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Probs object for classification outputs
            # print(result)
        del model
        return (masks,names,grids,images_debug,)
    

