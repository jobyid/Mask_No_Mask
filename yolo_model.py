import torch
import cv2
import time 

# Model
y_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def crop_person(tensor_array, img):
    i = -1
    for box in tensor_array: 
        for b in box:
            i += 1 
            if b[-1:] == 0.0:
                #print("person")
                x1 = int(b[0:1].item())
                y1 = int(b[1:2].item())
                x2  = int(b[2:3].item())
                y2 = int(b[3:4].item())
                #print(x1,y1,w,h)
                c_img = img[y1:y2, x1:x2]
                name = "results/Crops/Person" + str(time.time()) + ".png"
                cv2.imwrite(name, c_img)

def run_yolo(video=True, file_path="", frames=0):
    if video:
        vid = cv2.VideoCapture(file_path)
        frame_count = 0 
        while True:
            frame_count += 1
            ok, frame = vid.read()
            if ok: 
                if frame_count % frames == 0:
                    res = y_model(frame)
                    cords = res.xyxy
                    crop_person(cords, frame)
            else: 
                break 
    else: 
        img = cv2.imread(file_path)
        res = y_model(img)
        cords = res.xyxy
        crop_person(cords, img)