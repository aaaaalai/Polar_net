import yaml
from importlib import import_module

##原来的##def get_cfg(args):
####if args.cfg.endswith('yaml'):
####cfg_dict = get_yaml(args.cfg)
####elif args.cfg.endswith('py'):
####cfg_dict = get_py(args.cfg)
####the priority of the config file is higher than argparse
#### for key, value in cfg_dict.items():
####   args.__setattr__(key, value)
###代码#return args

def get_cfg(args):
    if args.cfg.endswith('yaml'):
        cfg_dict = get_yaml(args.cfg)
    elif args.cfg.endswith('.py') or args.cfg.endswith('.pyc'):
        cfg_dict = get_py(args.cfg)
    else:
        # 如果是模块名（比如 Config.polarrcnn_tusimple_r18）
        try:
            mod = import_module(args.cfg)
            cfg_dict = {k: v for k, v in mod.__dict__.items() if not k.startswith('__')}
        except ModuleNotFoundError as e:
            raise ValueError(f"Cannot import config: {args.cfg}, error: {e}")

    # 配置文件优先级高于命令行参数
    for key, value in cfg_dict.items():
        setattr(args, key, value)
    return args

def get_yaml(filename):
    with open(filename) as f:
        cfg_dict = yaml.load(f.read(), Loader = yaml.FullLoader)
    return cfg_dict

def get_py(filename):
    filename = convert_filename_to_package(filename)
    mod = import_module(filename)
    mod_dict = mod.__dict__
    del_keys = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__', '__cached__', '__builtins__']
    for del_key in del_keys:
        mod_dict.pop(del_key)
    return mod_dict 


def convert_filename_to_package(filename):
    # 去除文件扩展名
    filename = filename.strip('.py')
    filename = filename.replace('/', '.')
    filename = filename.replace('\\', '.')
    while filename.startswith('.'):
        filename = filename[1:]
    return filename