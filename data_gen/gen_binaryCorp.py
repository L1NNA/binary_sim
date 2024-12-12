import json
import gzip
import os
import random
import pickle
from collections import defaultdict
from tqdm import trange, tqdm
import re
import numpy as np
import torch
import glob

## oprs filter
oprs_filter = (
    'loc_', 'sub_', 'arg_', 'var_',
    'unk_', 'word_', 'off_', 'locret_',
    'flt_', 'dbl_', 'param_', 'local_'
)

def get_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

## process each instruction
def ins2seq(ins):
    tkns = []
    for tkn in ins.split(' '):
        ## check for address
        address_pattern = re.compile(r'\[.*?\]|0x[0-9A-Fa-f]+|\d+H|^0f|\b[a-zA-Z_]\w*:\b|\b[A-Fa-f0-9]{4,}\b')
        string_pattern = re.compile(r'["\'].*?["\']')
        if address_pattern.search(tkn):
            tkns.append('<addr>')
            continue
        if string_pattern.search(tkn):
            tkns.append('<str>')
            continue
        if tkn.startswith('byte'):
            tkns.append('<byte>')
            continue
        for p in re.split(r'[,+\-*\\\[\]:()\s@?$]', tkn.lower()):
            if len(p) > 0:
                if p.startswith(oprs_filter):
                    for opr in oprs_filter:
                        if p.startswith(opr):
                            tkns.append(opr[:-1])
                else:
                    tkns.extend([s for s in p.split('_') if len(s.strip()) > 0])
    return ' '.join(tkns)


def find_pattern(p, string):
    pattern = re.compile(p)
    pattern = pattern.search(string)
    if pattern is not None:
        return pattern.group()
    else:
        return ''

def process_func(opt,  func_name = None, file_name = None):
    if file_name is not None:
        f_name = file_name.split('/')[-1]
        opt_pattern = r'(-O0-|-O1-|-O2-|-O3-|-Os-)'
        opt_token = ' '+find_pattern(opt_pattern, file_name)[1:3]
    ID, code, raw_byte, _, graph = opt
    tkns = []
    code = [re.sub(' +', ' ', ' '.join(i.split(','))) for i in code]
    
    for block in code:
        ins = ins2seq(block)
        tkns.append(ins)
                    
    if func_name is None and file_name is None:
        return ' '.join(tkns)
    return ' '.join(tkns) + f'{opt_token.lower()}'
    



def generate_single(list_of_files, stage):
    
#     if stage != 'training':
#         for _file in tqdm(list_of_files, f'Parsing {stage} files'):
#             ## _file is a dict, with key being the function name, and value is a list of 5 element, with each element being a opt level
#             try:
#                 data = get_data(_file)
#                 for key, value in data.items():  ## loop through functions
#                     function_name = key
#                     code = process_func(value, func_name = function_name, file_name = _file)
#                     result = {'function': function_name, 'blocks': code, 'file': _file}
#                     yield result
                    
#             except Exception as e:
#                 print(_file, e)

#     else: 
    for _file in tqdm(list_of_files, f'Parsing {stage} files'):
        ## _file is a dict, with key being the function name, and value is a list of 5 element, with each element being a opt level
        try:
            data = get_data(_file)
            for key, value in data.items():  ## loop through functions
                function_name = key
                for opt in value:
                    ### no additional opt token during testing?
                    code = process_func(opt, func_name = function_name, file_name = _file)
                    result = {'function': function_name, 'blocks': code, 'file': _file}
                    yield result
                # if require_cfg:
                #     cfg = calls2cfg(function, data)
                #     result['cfg'] = cfg
                # yield result
        except Exception as e:
            print(_file, e)

    
def main():
    
    # generate training
    path_train = "datasets/BinaryCorp/small_train/**"
    training = glob.glob(path_train, recursive=True)
    training = [i for i in training if 'saved_index.pkl' in i]
    
    print("Generating training data...")
    with open('datasets/train_BinaryCorp.jsonl', 'a') as f:
        for js in generate_single(training, 'training'):
            f.write(json.dumps(js) + '\n')
    print("Finish training data generation")
    
#     # get testing files for cross-opt
#     path_opt = "datasets/BinaryCorp/small_test/**"
#     testing = glob.glob(path_opt, recursive=True)
#     O0 = [i for i in testing if 'O0' in i.split('-')]
#     O1 = [i for i in testing if 'O1' in i.split('-')]
#     O2 = [i for i in testing if 'O2' in i.split('-')]
#     O3 = [i for i in testing if 'O3' in i.split('-')]
#     Os = [i for i in testing if 'Os' in i.split('-')]
    
#     # get testing files for O0 optimization
#     print("Generating O0 data...")
#     with open('datasets/test_BinaryCorp_O0.jsonl', 'a') as f:
#         for js in generate_single(O0, 'O0'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish O0 data generation")
    
#     # get testing files for O1 optimization
#     print("Generating O1 data...")
#     with open('datasets/test_BinaryCorp_O1.jsonl', 'a') as f:
#         for js in generate_single(O1, 'O1'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish O1 data generation")
    
#     # get testing files for O2 optimization
#     print("Generating O2 data...")
#     with open('datasets/test_BinaryCorp_O2.jsonl', 'a') as f:
#         for js in generate_single(O2, 'O2'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish O2 data generation")
    
#     # get testing files for O3 optimization
#     print("Generating O3 data...")
#     with open('datasets/test_BinaryCorp_O3.jsonl', 'a') as f:
#         for js in generate_single(O3, 'O3'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish O3 data generation")
    
#     # get testing files for Os optimization
#     print("Generating Os data...")
#     with open('datasets/test_BinaryCorp_Os.jsonl', 'a') as f:
#         for js in generate_single(Os, 'Os'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish Os data generation")
    
        

if __name__ == '__main__':
    main()