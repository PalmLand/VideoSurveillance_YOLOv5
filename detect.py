import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        
        #####################################################################################
        # results
        G_6,G_5,G_4,G_3,G_2,G_1 = 0,0,0,0,0,0
        LY_6,LY_5,LY_4,LY_3,LY_2,LY_1 = 0,0,0,0,0,0
        R_6,R_5,R_4,R_3,R_2,R_1 = 0,0,0,0,0,0
        RY_6,RY_5,RY_4,RY_3,RY_2,RY_1 = 0,0,0,0,0,0
        #####################################################################################
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            ################################################################################
            ## Lines图中从下到上依次是1-x
            #Line1
            cv2.line(im0, (590,1400), (1020,1400), (255, 255, 0), 5) 
            cv2.line(im0, (1020,1400), (1145,1400), (255, 0, 255), 5)
            cv2.line(im0, (1145,1400), (1860,1400), (0, 0, 255), 5)
            cv2.line(im0, (1860,1400), (2030,1400), (0, 255, 255), 5)

            #Line2
            cv2.line(im0, (640,1300), (1020,1300), (255, 255, 0), 5) 
            cv2.line(im0, (1020,1300), (1130,1300), (255, 0, 255), 5)
            cv2.line(im0, (1130,1300), (1765,1300), (0, 0, 255), 5)
            cv2.line(im0, (1765,1300), (1910,1300), (0, 255, 255), 5)

            #Line3
            cv2.line(im0, (700,1200), (1005,1200), (255, 255, 0), 5) 
            cv2.line(im0, (1005,1200), (1110,1200), (255, 0, 255), 5)
            cv2.line(im0, (1110,1200), (1660,1200), (0, 0, 255), 5)
            cv2.line(im0, (1660,1200), (1800,1200), (0, 255, 255), 5)

            #Line4
            cv2.line(im0, (750,1100), (1010,1100), (255, 255, 0), 5) 
            cv2.line(im0, (1010,1100), (1100,1100), (255, 0, 255), 5)
            cv2.line(im0, (1100,1100), (1555,1100), (0, 0, 255), 5)
            cv2.line(im0, (1555,1100), (1680,1100), (0, 255, 255), 5)

            #Line5
            cv2.line(im0, (800,1000), (1005,1000), (255, 255, 0), 5) 
            cv2.line(im0, (1005,1000), (1090,1000), (255, 0, 255), 5)
            cv2.line(im0, (1090,1000), (1450,1000), (0, 0, 255), 5)
            cv2.line(im0, (1450,1000), (1570,1000), (0, 255, 255), 5)

            #Line6
            cv2.line(im0, (855,900), (1005,900), (255, 255, 0), 5) 
            cv2.line(im0, (1005,900), (1080,900), (255, 0, 255), 5)
            cv2.line(im0, (1080,900), (1345,900), (0, 0, 255), 5)
            cv2.line(im0, (1345,900), (1450,900), (0, 255, 255), 5)

            #Line7
            cv2.line(im0, (900,800), (1000,800), (255, 255, 0), 5) 
            cv2.line(im0, (1000,800), (1060,800), (255, 0, 255), 5) 
            cv2.line(im0, (1060,800), (1250,800), (0, 0, 255), 5)
            cv2.line(im0, (1250,800), (1350,800), (0, 255, 255), 5)

            #Row
            cv2.line(im0, (590,1400), (900,800), (255, 255, 0), 5) 
            cv2.line(im0, (1020,1400), (1000,800), (255, 0, 255), 5) 
            cv2.line(im0, (1145,1400), (1060,800), (0, 0, 255), 5)
            cv2.line(im0, (1860,1400), (1250,800), (0, 255, 255), 5)
            cv2.line(im0, (2030,1400), (1350,800), (0, 255, 255), 5)
            
            '''
                             x1,y1-------x2,y1
                             |               |
                             |    PERSON     |
            a1_x_y,b1_x_y----|---------------|----a2_x_y,b1_x_y
            |                |               |                |
            |                |               |                |
            |                x1,y2-------x2,y2                |
            |                                                 |
            |                                                 |
            a1_x_y,b2_x_y-------------------------a2_x_y,b2_x_y
            ,where x=G,LY,R,RY, y=6,5,4,3,2,1
            '''     
            a1_G_6,a2_G_6,b1_G_6,b2_G_6 = 900,1000,800,900
            a1_LY_6,a2_LY_6,b1_LY_6,b2_LY_6 = 1005,1060,800,900
            a1_R_6,a2_R_6,b1_R_6,b2_R_6 = 1080,1250,800,900
            a1_RY_6,a2_RY_6,b1_RY_6,b2_RY_6 = 1300,1380,800,900
            
            a1_G_5,a2_G_5,b1_G_5,b2_G_5 = 855,1005,900,1000
            a1_LY_5,a2_LY_5,b1_LY_5,b2_LY_5 = 1005,1080,900,1000
            a1_R_5,a2_R_5,b1_R_5,b2_R_5 = 1090,1345,900,1000
            a1_RY_5,a2_RY_5,b1_RY_5,b2_RY_5 = 1450,1500,900,1000

            a1_G_4,a2_G_4,b1_G_4,b2_G_4 = 800,1005,1000,1100
            a1_LY_4,a2_LY_4,b1_LY_4,b2_LY_4 = 1010,1090,1000,1100
            a1_R_4,a2_R_4,b1_R_4,b2_R_4 = 1100,1450,1000,1100
            a1_RY_4,a2_RY_4,b1_RY_4,b2_RY_4 = 1500,1570,1000,1100

            a1_G_3,a2_G_3,b1_G_3,b2_G_3 = 720,1005,1100,1200
            a1_LY_3,a2_LY_3,b1_LY_3,b2_LY_3 = 1005,1110,1100,1200
            a1_R_3,a2_R_3,b1_R_3,b2_R_3 = 1110,1555,1100,1200
            a1_RY_3,a2_RY_3,b1_RY_3,b2_RY_3 = 1660,1750,1100,1200

            a1_G_2,a2_G_2,b1_G_2,b2_G_2 = 700,1005,1200,1300
            a1_LY_2,a2_LY_2,b1_LY_2,b2_LY_2 = 1020,1110,1200,1300
            a1_R_2,a2_R_2,b1_R_2,b2_R_2 = 1130,1660,1200,1300
            a1_RY_2,a2_RY_2,b1_RY_2,b2_RY_2 = 1765,1860,1200,1300

            a1_G_1,a2_G_1,b1_G_1,b2_G_1 = 640,1020,1300,1400
            a1_LY_1,a2_LY_1,b1_LY_1,b2_LY_1 = 1020,1130,1300,1400
            a1_R_1,a2_R_1,b1_R_1,b2_R_1 = 1145,1765,1300,1400
            a1_RY_1,a2_RY_1,b1_RY_1,b2_RY_1 = 1765,1910,1300,1400    

            ## Text
            font = cv2.FONT_HERSHEY_SIMPLEX
            im0 = cv2.putText(im0,"G",(200,140),font,1,(255,255,255),5)
            im0 = cv2.putText(im0,"LY",(250,140),font,1,(255,255,255),5)
            im0 = cv2.putText(im0,"R",(300,140),font,1,(255,255,255),5)
            im0 = cv2.putText(im0,"RY",(350,140),font,1,(255,255,255),5)

            im0 = cv2.putText(im0,"6 |",(100,200),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,"5 |",(100,250),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,"4 |",(100,300),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,"3 |",(100,350),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,"2 |",(100,400),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,"1 |",(100,450),font,1.5,(255,255,255),3)            
            #####################################################################################
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # print('det shape =',det.shape)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        
                        #####################################################################################
                        if names[int(cls)] == 'person':
                            print(label)
                        #####################################################################################
                        
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            
                            #####################################################################################
                            ## Count 
                            x1 = int(xyxy[0])
                            y1 = int(xyxy[1])
                            x2 = int(xyxy[2])
                            y2 = int(xyxy[3])
                            print('\nx1 =',x1)
                            print('y1 =',y1)
                            print('x2 =',x2)
                            print('y2 =',y2)
                            if x1<=a2_G_6 and x2>=a1_G_6 and y2<=b2_G_6 and y2>=b1_G_6:
                                G_6 += 1
                            if x1<=a2_G_5 and x2>=a1_G_5 and y2<=b2_G_5 and y2>=b1_G_5:
                                G_5 += 1
                            if x1<=a2_G_4 and x2>=a1_G_4 and y2<=b2_G_4 and y2>=b1_G_4:
                                G_4 += 1
                            if x1<=a2_G_3 and x2>=a1_G_3 and y2<=b2_G_3 and y2>=b1_G_3:
                                G_3 += 1
                            if x1<=a2_G_2 and x2>=a1_G_2 and y2<=b2_G_2 and y2>=b1_G_2:
                                G_2 += 1
                            if x1<=a2_G_1 and x2>=a1_G_1 and y2<=b2_G_1 and y2>=b1_G_1:
                                G_1 += 1 
                                
                            if x1<=a2_LY_6 and x2>=a1_LY_6 and y2<=b2_LY_6 and y2>=b1_LY_6:
                                LY_6 += 1
                            if x1<=a2_LY_5 and x2>=a1_LY_5 and y2<=b2_LY_5 and y2>=b1_LY_5:
                                LY_5 += 1
                            if x1<=a2_LY_4 and x2>=a1_LY_4 and y2<=b2_LY_4 and y2>=b1_LY_4:
                                LY_4 += 1
                            if x1<=a2_LY_3 and x2>=a1_LY_3 and y2<=b2_LY_3 and y2>=b1_LY_3:
                                LY_3 += 1
                            if x1<=a2_LY_2 and x2>=a1_LY_2 and y2<=b2_LY_2 and y2>=b1_LY_2:
                                LY_2 += 1
                            if x1<=a2_LY_1 and x2>=a1_LY_1 and y2<=b2_LY_1 and y2>=b1_LY_1:
                                LY_1 += 1           
                                
                            if x1<=a2_R_6 and x2>=a1_R_6 and y2<=b2_R_6 and y2>=b1_R_6:
                                R_6 += 1
                            if x1<=a2_R_5 and x2>=a1_R_5 and y2<=b2_R_5 and y2>=b1_R_5:
                                R_5 += 1
                            if x1<=a2_R_4 and x2>=a1_R_4 and y2<=b2_R_4 and y2>=b1_R_4:
                                R_4 += 1
                            if x1<=a2_R_3 and x2>=a1_R_3 and y2<=b2_R_3 and y2>=b1_R_3:
                                R_3 += 1
                            if x1<=a2_R_2 and x2>=a1_R_2 and y2<=b2_R_2 and y2>=b1_R_2:
                                R_2 += 1
                            if x1<=a2_R_1 and x2>=a1_R_1 and y2<=b2_R_1 and y2>=b1_R_1:
                                R_1 += 1              

                            if x1<=a2_RY_6 and x2>=a1_RY_6 and y2<=b2_RY_6 and y2>=b1_RY_6:
                                RY_6 += 1
                            if x1<=a2_RY_5 and x2>=a1_RY_5 and y2<=b2_RY_5 and y2>=b1_RY_5:
                                RY_5 += 1
                            if x1<=a2_RY_4 and x2>=a1_RY_4 and y2<=b2_RY_4 and y2>=b1_RY_4:
                                RY_4 += 1
                            if x1<=a2_RY_3 and x2>=a1_RY_3 and y2<=b2_RY_3 and y2>=b1_RY_3:
                                RY_3 += 1
                            if x1<=a2_RY_2 and x2>=a1_RY_2 and y2<=b2_RY_2 and y2>=b1_RY_2:
                                RY_2 += 1
                            if x1<=a2_RY_1 and x2>=a1_RY_1 and y2<=b2_RY_1 and y2>=b1_RY_1:
                                RY_1 += 1                                           
                            #####################################################################################
                            
            #####################################################################################            
            #print(G_6,G_5,G_4,G_3,G_2,G_1)                        
            #print(LY_6,LY_5,LY_4,LY_3,LY_2,LY_1)
            #print(R_6,R_5,R_4,R_3,R_2,R_1)
            #print(RY_6,RY_5,RY_4,RY_3,RY_2,RY_1)
            
            ## Sum
            count = G_6+G_5+G_4+G_3+G_2+G_1+LY_6+LY_5+LY_4+LY_3+LY_2+LY_1+R_6+R_5+R_4+R_3+R_2+R_1+RY_6+RY_5+RY_4+RY_3+RY_2+RY_1
            im0 = cv2.putText(im0,'Person: ',(100,600),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(count),(280,600),font,1.5,(255,255,255),3)
                
            
            # 6
            im0 = cv2.putText(im0,str(G_6),(200,200),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(LY_6),(250,200),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(R_6),(300,200),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(RY_6),(350,200),font,1.5,(255,255,255),3)
            # 5
            im0 = cv2.putText(im0,str(G_5),(200,250),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(LY_5),(250,250),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(R_5),(300,250),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(RY_5),(350,250),font,1.5,(255,255,255),3)

            # 4
            im0 = cv2.putText(im0,str(G_4),(200,300),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(LY_4),(250,300),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(R_4),(300,300),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(RY_4),(350,300),font,1.5,(255,255,255),3)

            # 3
            im0 = cv2.putText(im0,str(G_3),(200,350),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(LY_3),(250,350),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(R_3),(300,350),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(RY_3),(350,350),font,1.5,(255,255,255),3)

            # 2
            im0 = cv2.putText(im0,str(G_2),(200,400),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(LY_2),(250,400),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(R_2),(300,400),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(RY_2),(350,400),font,1.5,(255,255,255),3)

            # 1
            im0 = cv2.putText(im0,str(G_1),(200,450),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(LY_1),(250,450),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(R_1),(300,450),font,1.5,(255,255,255),3)
            im0 = cv2.putText(im0,str(RY_1),(350,450),font,1.5,(255,255,255),3)
            #######################################+#########################################
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
