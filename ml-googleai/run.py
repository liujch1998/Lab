import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           '/m/0cmf2', '/m/0199g', '/m/015p6', '/m/019jd',
           '/m/04dr76w', '/m/01bjv', '/m/0k4j', '/m/01yrx', '/m/01mzpv',
           '/m/01xq0k1', '/m/0h8n5zk', '/m/0bt9lr', '/m/03k3r',
           '/m/04_sv', '/m/01g317', '/m/03fp41',
           '/m/07bgp', '/m/03m3pdh', '/m/07jdr', '/m/07c52')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def predict (net, token, f):
    im_file = 'googleai/challenge2018_test/' + token + '.jpg'
    im = cv2.imread(im_file)
    h, w, c = im.shape

    scores, boxes = im_detect(net, im)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    f.write('%s,' % token)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            bbox = dets[i, :4]  # xmin, ymin, xmax, ymax
            score = dets[i, -1]
            xmin = bbox[0] / w
            ymin = bbox[1] / h
            xmax = bbox[2] / w
            ymax = bbox[3] / h
            f.write('%s %f %f %f %f %f ' % (cls, score, xmin, ymin, xmax, ymax))
    f.write('\n')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

cfg.TEST.HAS_RPN = True  # Use RPN for proposals

args = parse_args()

prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                          NETS[args.demo_net][1])

if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
                   'fetch_faster_rcnn_models.sh?').format(caffemodel))

if args.cpu_mode:
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

print '\n\nLoaded network {:s}'.format(caffemodel)

n_images = 99999
images = []
for file in os.listdir('googleai/challenge2018_test'):
    if not file.endswith('.jpg'):
        continue
    token = file[:-4]
    images.append(token)
    if len(images) == n_images:
        break
if len(images) != n_images:
    print('CRITICAL: less than 99999 images loaded!')

with open('ans.csv', 'w') as f:
    f.write('ImageId,PredictionString\n')
    for i in range(n_images):
        token = images[i]
        predict(net, token, f)
        if i % 1000 == 0:
            print('Progress: %d%%' % (100 * i // n_images))
