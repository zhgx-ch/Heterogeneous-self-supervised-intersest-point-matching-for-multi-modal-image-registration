import logging
import yaml
import os
import argparse
import numpy as np
from contextlib import contextmanager
from json import dumps as pprint

from superpoint.datasets import get_dataset
from superpoint.models import get_model
from superpoint.utils.stdout_capturing import capture_outputs
from superpoint.settings import EXPER_PATH


import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',    #%(asctime)s: 打印日志的时间
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)     #%(levelname)s: 打印日志级别名称
import tensorflow as tf  # noqa: E402                                    #%(message)s: 打印日志信息


def train(config, n_iter, output_dir, checkpoint_name='model.ckpt'):
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    with _init_graph(config) as net:
        try:
            net.train(n_iter, output_dir=output_dir,
                      validation_interval=config.get('validation_interval', 100),
                      save_interval=config.get('save_interval', None),
                      checkpoint_path=checkpoint_path,
                      keep_checkpoints=config.get('keep_checkpoints', 1))
        except KeyboardInterrupt:
            logging.info('Got Keyboard Interrupt, saving model and closing.')
        net.save(os.path.join(output_dir, checkpoint_name))


def evaluate(config, output_dir, n_iter=None):
    with _init_graph(config) as net:
        net.load(output_dir)
        results = net.evaluate(config.get('eval_set', 'test'), max_iterations=n_iter)
    return results


def predict(config, output_dir, n_iter):
    pred = []
    data = []
    data2= []
    with _init_graph(config, with_dataset=True) as (net, dataset1,dataset2):
        if net.trainable:
            net.load(output_dir)
        test_set = dataset.get_test_set()
        test_set2 = dataset2.get_test_set()
        for _ in range(n_iter):
            data.append(next(test_set))
            data2.append(next(test_set2))
            pred.append(net.predict(data[-1], keys='*'))
    return pred, data


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def get_num_gpus():
    return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))


@contextmanager
def _init_graph(config, with_dataset=False):
    set_seed(config.get('seed', int.from_bytes(os.urandom(4), byteorder='big')))
    n_gpus = get_num_gpus()   #获取gpu数量
    logging.info('Number of GPUs detected: {}'.format(n_gpus)) #打印信息

     #转到synthetic_shapes.py中_init_dataset_()函数，得到随机打乱的数据集
    dataset1 = get_dataset(config['data']['name1'])(**config['data']) 
    dataset2 = get_dataset(config['data']['name2'])(**config['data'])
    
    
    #转到magic_point.py中_model()函数，
    model = get_model(config['model']['name'])(
            data1={} if with_dataset else dataset1.get_tf_datasets(), data2={} if with_dataset else dataset2.get_tf_datasets() ,
             n_gpus=n_gpus, **config['model'])
    model.__enter__()
    if with_dataset:
        yield model, dataset1,dataset2

    else:
        #import pdb
        #pdb.set_trace()
        yield model
    model.__exit__() 
    #tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。


def _cli_train(config, output_dir, args):
    assert 'train_iter' in config

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False) #把生成python对象生成为yaml文档
    train(config, config['train_iter'], output_dir) 

    if args.eval:
        _cli_eval(config, output_dir, args)


def _cli_eval(config, output_dir, args):
    # Load model config from previous experiment
    with open(os.path.join(output_dir, 'config.yml'), 'r') as f:
        model_config = yaml.load(f)['model']
    model_config.update(config.get('model', {}))
    config['model'] = model_config

    results = evaluate(config, output_dir, n_iter=config.get('eval_iter'))

    # Print and export results
    logging.info('Evaluation results: \n{}'.format(
        pprint(results, indent=2, default=str)))
    with open(os.path.join(output_dir, 'eval.txt'), 'a') as f:
        f.write('Evaluation for {} dataset:\n'.format(config['data']['name']))
        for r, v in results.items():
            f.write('\t{}:\n\t\t{}\n'.format(r, v))
        f.write('\n')


# TODO
def _cli_pred(config, args):
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.set_defaults(func=_cli_train)    #设置默认函数

    # Evaluation command
    p_train = subparsers.add_parser('evaluate')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.set_defaults(func=_cli_eval)

    # Inference command
    p_train = subparsers.add_parser('predict')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.set_defaults(func=_cli_pred)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader) #yaml.load(input, Loader=yaml.FullLoader)
    output_dir = os.path.join('experiments_data', args.exper_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with capture_outputs(os.path.join(output_dir, 'log')):
        logging.info('Running command {}'.format(args.command.upper()))
        args.func(config, output_dir, args)
