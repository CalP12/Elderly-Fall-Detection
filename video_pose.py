# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2
import numpy as np
import argparse
import imutils
import time
parser = argparse.ArgumentParser(
        description='This script is used to demonstrate OpenPose human pose estimation network '
                    'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using OpenCV. '
                    'The sample and model are simplified and could be used for a single person on the frame.')
#parser.add_argument('--input', help='Path to input image.')
parser.add_argument('--proto', help='Path to .prototxt' ,default='pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt')
parser.add_argument('--model', help='Path to .caffemodel', default='pose/mpi/pose_iter_160000.caffemodel')
parser.add_argument('--dataset', help='Specify what kind of model was trained. '
                                      'It could be (COCO, MPI) depends on dataset.' ,default='MPI')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

if args.dataset == 'COCO':
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
elif args.dataset=='MPI':
    #assert(args.dataset == 'MPI')
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                   "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                   ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                   ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
else:
    
    BODY_PARTS ={"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,"LElbow":6,"LWrist":7,"MidHip":8,"RHip":9,"RKnee":10,"RAnkle":11,"LHip":12,"LKnee":13,"LAnkle":14,"REye":15,"LEye":16,"REar":17,"LEar":18,"LBigToe":19,"LSmallToe":20,"LHeel":21,"RBigToe":22,"RSmallToe":23,"RHeel":24,"Background":25}

    POSE_PAIRS =[ ["Neck","MidHip"],   ["Neck","RShoulder"],   ["Neck","LShoulder"],   ["RShoulder","RElbow"],   ["RElbow","RWrist"],   ["LShoulder","LElbow"],   ["LElbow","LWrist"],   ["MidHip","RHip"],   ["RHip","RKnee"],  ["RKnee","RAnkle"], ["MidHip","LHip"],  ["LHip","LKnee"], ["LKnee","LAnkle"],  ["Neck","Nose"],   ["Nose","REye"], ["REye","REar"],  ["Nose","LEye"], ["LEye","LEar"],   
["RShoulder","REar"],  ["LShoulder","LEar"],   ["LAnkle","LBigToe"],["LBigToe","LSmallToe"],["LAnkle","LHeel"], ["RAnkle","RBigToe"],["RBigToe","RSmallToe"],["RAnkle","RHeel"] ]

inWidth = args.width
inHeight = args.height
kwinName="Pose Estimation"



net = cv2.dnn.readNetFromCaffe(args.proto, args.model)

cap = cv2.VideoCapture('test2.mp4') 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
bll=np.zeros((frame_height,frame_width,3), dtype="uint8")

if (cap.isOpened()== False):
    print("Error opening video stream or file")

#out = cv2.VideoWriter('output/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
cframe=0
while cap.isOpened():
    ret,frame = cap.read()
    #frame = cv2.imread(args.input)
    if ret == True:
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp)
        start_t = time.time()
        out = net.forward()
        
        print("Time Taken ",time.time()-start_t)
        # print(inp.shape)
        #assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > args.thr else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]
            if points[idFrom] and points[idTo]:
                cv2.line(bll, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(bll, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
                cv2.ellipse(bll, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
                cv2.putText(bll, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255),2,cv2.LINE_AA)
                cv2.putText(bll, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255),1,cv2.LINE_AA)
        

        #cv2.imshow(kwinName, frame)
        cv2.imwrite('output/pics/'+str(cframe)+'.jpg',bll)
        cframe+=1
        bll=np.zeros((frame_height,frame_width,3), dtype="uint8")
        #if cv2.waitKey(25) & 0xFF == ord('q'):
            #break

    else:
        break
cap.release()    
cv2.destroyAllWindows()
#cv2.imwrite('C:/Users/crist/Desktop/Pose_Estimation/output/mpi_result_'+args.input,frame)