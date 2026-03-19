import yaml
import argparse
import os

class Config:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def merge_dict(a, b):
    for k, v in b.items():
        if k in a and isinstance(a[k], dict):
            merge_dict(a[k], v)
        else:
            a[k] = v
    return a

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--exp', default=None)

    # 覆盖参数（论文常用）
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)

    parser.add_argument('--source', nargs='+')
    parser.add_argument('--target', nargs='+')
    parser.add_argument('--resume', default=None)

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # 加载实验配置
    if args.exp:
        exp_cfg = load_yaml(args.exp)
        cfg = merge_dict(cfg, exp_cfg)

    # 命令行覆盖（最高优先级）
    if args.batch_size:
        cfg['train']['batch_size'] = args.batch_size
    if args.lr:
        cfg['train']['lr'] = args.lr
    if args.epochs:
        cfg['train']['epochs'] = args.epochs

    if args.source:
        cfg['domain']['source'] = args.source
    if args.target:
        cfg['domain']['target'] = args.target
    
    cfg['resume'] = args.resume

    return Config(cfg)