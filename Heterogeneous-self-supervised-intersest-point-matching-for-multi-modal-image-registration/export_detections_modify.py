import numpy as np
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import experiment_modify
from superpoint.settings import EXPER_PATH

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        print(tuple(s*np.flip(c, 0)))
        cv2.circle(img, tuple(s*np.flip(c, 0)), radius, color, thickness=-1)
    return img

# 引入coco数据集对它进行标注

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)  #配置文件路径
    parser.add_argument('experiment_name', type=str) # 权重文件名
    parser.add_argument('--export_name', type=str, default=None) #导出名
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pred_only', action='store_true')
    args = parser.parse_args()

    experiment_name = args.experiment_name
    export_name = args.export_name if args.export_name else experiment_name
    batch_size = args.batch_size
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert 'eval_iter' in config

    output_dir = Path(EXPER_PATH, 'outputs/{}/'.format(export_name))
    if not output_dir.exists():
        os.makedirs(output_dir)
    checkpoint = Path(EXPER_PATH, experiment_name)
    if 'checkpoint' in config:
        checkpoint = Path(checkpoint, config['checkpoint'])

    config['model']['pred_batch_size'] = batch_size
    batch_size *= experiment_modify.get_num_gpus()
    print(batch_size)
    #import ipdb
    #ipdb.set_trace()

    with experiment_modify._init_graph(config, with_dataset=True) as (net, dataset1,dataset2): # net指模型，
        if net.trainable:
            net.load(str(checkpoint))
        test_set = dataset1.get_test_set()
        test_set2 = dataset2.get_test_set()

        for _ in tqdm(range(config.get('skip', 0))):
            next(test_set)
            next(test_set2)

        pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)
        i = 0
        while True:
            # Gather dataset
            data = []
            data2 =[]
            try:
                for _ in range(batch_size):
                    data.append(next(test_set))
                    data2.append(next(test_set2))
            except (StopIteration, dataset1.end_set):
                if not data:
                    break
                data += [data[-1] for _ in range(batch_size - len(data))]  # add dummy
                data2 += [data2[-1] for _ in range(batch_size - len(data2))]
            data = dict(zip(data[0], zip(*[d.values() for d in data])))
            data2 = dict(zip(data2[0], zip(*[d.values() for d in data2])))
#             import ipdb
#             ipdb.set_trace()
            #print(type(data))
#             print(data.keys())
            #print(type(data['image']))
            #print(type(data2))
#             print(data2.keys())
            #print(type(data2['image']))
            #print(data['image'])
            #data['image']=tf.squeeze(data['image'],0)
            #data1=np.array(data['image'])
            #print(type(data['image']))
            #print(data1.shape)
            #print(len(data2['image']))
            #data3 = np.array(data2['image'])
            #print(type(data2['image']))
            #print(data3.shape)
            # Predict
            if args.pred_only:
#                 import ipdb
#                 ipdb.set_trace()
                p = net.predict(data, data2 , keys='pred', batch=True)
                pred = {'points': [np.array(np.where(e)).T for e in p]}
#                 orignal_image=cv2.imread('black_pic.png')
#                 image =[]
#                 print(type(pred['points']))

#                 image.append(draw_keypoints(orignal_image * 255, np.array(pred['points'])/255.))
#                 output_name = 'visualize_image.png'
#                 plt.imsave(output_name, image)
            else:
                pred = net.predict(data,data2, keys='descriptors', batch=True)
#                 print(type(pred))
#                 print(pred.shape)
#                 pred = np.transpose(pred, [1,2,0])
#                 print(pred.shape)
               
            

            # Export
            d2l = lambda d: [dict(zip(d, e)) for e in zip(*d.values())]  # noqa: E731
            for p, d in zip(d2l(pred), d2l(data)):
                if not ('name' in d):
                    p.update(d)  # Can't get the data back from the filename --> dump
                filename = d['name'].decode('utf-8') if 'name' in d else str(i)
                filepath = Path(output_dir, '{}.npz'.format(filename))
                np.savez_compressed(filepath, **p)
                i += 1
                pbar.update(1)

            if config['eval_iter'] > 0 and i >= config['eval_iter']:
                break
