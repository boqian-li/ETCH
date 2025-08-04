# options.py ---

import argparse
import os


class HierarchyArgmentParser():
    def __init__(self, flatten_args=['experiment', 'train', 'eval', 'test']):
        super(HierarchyArgmentParser, self).__init__()
        self.flatten_args = flatten_args
        self.parser = argparse.ArgumentParser()
        self.sub = self.parser.add_subparsers()
        self.parser_list = {}

    def add_parser(self, name):
        args = self.sub.add_parser(name)
        self.parser_list[name] = args
        return args

    def parse_args(self):
        opt_all, _ = self.parser.parse_known_args()
        for name, parser in self.parser_list.items():
            opt, _ = parser.parse_known_args()
            if name in self.flatten_args:
                for key, value in vars(opt).items():
                    setattr(opt_all, key, value)
            else:
                setattr(opt_all, name, opt)
        return opt_all


def dump_args(opt):
    args = {}
    for k, v in vars(opt).items():
        if isinstance(v, argparse.Namespace):
            args[k] = vars(v)
        else:
            args[k] = v
    return args




# def get_config(verbose=True, training=True):
#     config, unparsed = parser.parse_known_args()

#     args = vars(config)

#     if verbose:
#       print('------------ Options -------------')
#       for k, v in sorted(args.items()):
#           print('%s: %s' % (str(k), str(v)))
#       print('-------------- End ----------------')

#     if training:
#       create_dir(config.output_data_folder)
#       create_dir(os.path.join(config.output_data_folder, config.name))
#       file_name = os.path.join(config.output_data_folder, config.name, 'opt.txt')
#       with open(file_name, 'wt') as opt_file:
#           opt_file.write('------------ Options -------------\n')
#           for k, v in sorted(args.items()):
#               opt_file.write('%s: %s\n' % (str(k), str(v)))
#           opt_file.write('-------------- End ----------------\n')
#     return config, unparsed

# def print_config(opt):
#     args = vars(opt)
#     print('------------ Options -------------')
#     for k, v in sorted(args.items()):
#         print('%s: %s' % (str(k), str(v)))
#     print('-------------- End ----------------')


# def print_usage():
#     parser.print_usage()

# from parse import parse
# def parse_opt(opt):
#     ckpt_dir = os.path.dirname(opt.checkpoint_path)
#     opt_file = os.path.join(ckpt_dir, "opt.txt")
#     if os.path.exists(opt_file):
#       with open(opt_file,'r') as fp:
#         l = fp.readline()
#         while l:
#           if 'name' in l:
#             name = parse("name:{}",l)
#             if name is not None:
#               opt.name = name[0].strip()
#           elif 'dataset' in l:
#             dataset = parse("dataset:{}",l)
#             opt.dataset = dataset[0].strip()
#           elif 'network_model' in l:
#             network_model = parse("network_model:{}",l)
#             opt.network_model = network_model[0].strip()
#           elif 'input_dim' in l:
#             input_dim = parse("input_dim:{:d}",l)
#             opt.input_dim = input_dim[0]
#           elif 'output_dim' in l:
#             output_dim = parse("output_dim:{:d}",l)
#             opt.output_dim = output_dim[0]
#           elif 'normalize_input' in l:
#             normalize_input = parse("normalize_input:{}",l)
#             opt.normalize_input = normalize_input[0].strip() == "True"
#           # elif 'batch_size' in l:
#           #   batch_size = parse("batch_size:{:d}",l)
#           #   if batch_size is not None:
#           #     opt.batch_size = batch_size[0]
#           l = fp.readline()
#     else:
#       raise Exception("configuration (opt.txt) does not exist at checkpoint directory", ckpt_dir)
#     return opt


























# arg_lists = []
# parser = argparse.ArgumentParser()

# def add_argument_group(name):
#     arg = parser.add_argument_group(name)
#     arg_lists.append(arg)
#     return arg



# def get_config(verbose=True, training=True):
#     config, unparsed = parser.parse_known_args()

#     args = vars(config)

#     if verbose:
#       print('------------ Options -------------')
#       for k, v in sorted(args.items()):
#           print('%s: %s' % (str(k), str(v)))
#       print('-------------- End ----------------')

#     if training:
#       create_dir(config.output_data_folder)
#       create_dir(os.path.join(config.output_data_folder, config.name))
#       file_name = os.path.join(config.output_data_folder, config.name, 'opt.txt')
#       with open(file_name, 'wt') as opt_file:
#           opt_file.write('------------ Options -------------\n')
#           for k, v in sorted(args.items()):
#               opt_file.write('%s: %s\n' % (str(k), str(v)))
#           opt_file.write('-------------- End ----------------\n')
#     return config, unparsed

# def print_config(opt):
#     args = vars(opt)
#     print('------------ Options -------------')
#     for k, v in sorted(args.items()):
#         print('%s: %s' % (str(k), str(v)))
#     print('-------------- End ----------------')


# def print_usage():
#     parser.print_usage()

# from parse import parse
# def parse_opt(opt):
#     ckpt_dir = os.path.dirname(opt.checkpoint_path)
#     opt_file = os.path.join(ckpt_dir, "opt.txt")
#     if os.path.exists(opt_file):
#       with open(opt_file,'r') as fp:
#         l = fp.readline()
#         while l:
#           if 'name' in l:
#             name = parse("name:{}",l)
#             if name is not None:
#               opt.name = name[0].strip()
#           elif 'dataset' in l:
#             dataset = parse("dataset:{}",l)
#             opt.dataset = dataset[0].strip()
#           elif 'network_model' in l:
#             network_model = parse("network_model:{}",l)
#             opt.network_model = network_model[0].strip()
#           elif 'input_dim' in l:
#             input_dim = parse("input_dim:{:d}",l)
#             opt.input_dim = input_dim[0]
#           elif 'output_dim' in l:
#             output_dim = parse("output_dim:{:d}",l)
#             opt.output_dim = output_dim[0]
#           elif 'normalize_input' in l:
#             normalize_input = parse("normalize_input:{}",l)
#             opt.normalize_input = normalize_input[0].strip() == "True"
#           # elif 'batch_size' in l:
#           #   batch_size = parse("batch_size:{:d}",l)
#           #   if batch_size is not None:
#           #     opt.batch_size = batch_size[0]
#           l = fp.readline()
#     else:
#       raise Exception("configuration (opt.txt) does not exist at checkpoint directory", ckpt_dir)
#     return opt
