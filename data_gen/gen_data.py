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
from varname import nameof

## oprs filter
oprs_filter = (
    'loc_', 'sub_', 'arg_', 'var_',
    'unk_', 'word_', 'off_', 'locret_',
    'flt_', 'dbl_', 'param_', 'local_'
)

def get_data(filename):
    with gzip.open(filename) as f:
        data = f.read()
    data = json.loads(data)
    return data

## process each instruction
def ins2seq(ins, mne_only=False):
    tkns = []
    for tkn in ([ins['mne']] + ins['oprs']) if not mne_only else [ins['mne']]:
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

### use include this if a CFG is required
def calls2cfg(f, data):
    Allblocks = [b for b in data['blocks'] if b['addr_f'] == f['addr_start']]
    blockName = [b['addr_start'] for b in Allblocks]
    matrix = np.zeros((len(Allblocks), len(Allblocks)))
    np.fill_diagonal(matrix, 1)
    for b in Allblocks:
        if b['calls'] != []:
            for c in b['calls']:
                if c in blockName:
                    matrix[blockName.index(b['addr_start'])][blockName.index(c)] = 1
    return matrix.astype(int).tolist()

def find_pattern(p, string):
    pattern = re.compile(p)
    pattern = pattern.search(string)
    if pattern is not None:
        return pattern.group()
    else:
        return ''


### process a function into a blocks of sequences
def func2blocks(func, data, file_name = None, add_token = True, stage = None):
    if file_name is not None and add_token == True and stage=='training':
        f_name = file_name.split('/')[-1]
        opt_pattern = r'(o0|o1|o2|o3|os|O0|O1|O2|O3|Os)'
        opt_token = find_pattern(opt_pattern, file_name)
        if opt_token != '':
            opt_token += ' '
        
        compiler_pattern = r'(gcc|clang|gcc32)'
        compiler_token = find_pattern(compiler_pattern, file_name)
        if compiler_token != '':
            compiler_token += ' '

        arch_pattern = r'(arm|mips|powerpc)'
        arch_token = find_pattern(arch_pattern, file_name)
        if arch_token == '':
            arch_token = 'x86'
        
        
        obf_pattern = r'(bcf|fla|sub|sub-fla-bcf)'
        obf_token = find_pattern(obf_pattern, file_name)
        
        if obf_token != '':
            arch_token += ' '
        # func_name = func['name']+' '
    
    blocks = []
    
    all_blocks = [b for b in data['blocks'] if b['addr_f'] == func['addr_start']]
    for block in all_blocks:
        block_ins = ' '.join([ins2seq(ins) for ins in block['ins']])
        if block_ins != "": # filter out empty block
            blocks.append(block_ins)
    if file_name is not None and add_token == True and stage=='training':
        blocks.append(f'{opt_token.lower()}{compiler_token.lower()}{arch_token.lower()}{obf_token.lower()}')
    return blocks

def generate_single(list_of_files, stage, require_cfg=False, min_blocks=3, add_compiler_token = True):

    for _file in tqdm(list_of_files, f'Parsing {stage} files'):
        try:
            data = get_data(_file)
            for function in data['functions']:
                if function['bbs_len'] <= min_blocks:
                    continue
                blocks = func2blocks(function, data, file_name = _file, add_token = add_compiler_token, stage = stage)
                if len(blocks) <= min_blocks: # check again
                    continue
                result = {'function': function['name'], 'blocks':blocks, 'file':_file}
                if require_cfg:
                    cfg = calls2cfg(function, data)
                    result['cfg'] = cfg
                yield result
        except Exception as e:
            print(_file, e)

    
def main():
    training, O0, O1, O2, O3, gcc, clang, obf_bcf, obf_fla, obf_all, obf_sub, obf_none, arm, x86_64, x86_32, mips, powerpc = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    
#     # generate training
#     path_train = 'datasets/firmware_images'
#     libs = [
#         'l-busybox-busybox_unstripped', 'l-sqlite-sqlite3',
#         'l-coreutils-coreutils', 'l-curl-curl',
#         'l-ImageMagick-magick', 'l-putty-puttygen'
#     ]
#     for lib in libs:
#         files = os.listdir(os.path.join(path_train, lib))
#         for file in files:
#             if 'merged' in file:
#                 training.append(os.path.join(path_train, lib,file))
                
#     ## add obfuscated code
#     path_train2 = 'libraries/data/obfuscation/imagemagick'
#     files = os.listdir(path_train2)
#     for file in files:
#         if 'merged' in file:
#             training.append(os.path.join(path_train2, file))

#     ## add clang code
#     path_train3 = 'libraries/data/compiler/imagemagick'
#     files = os.listdir(path_train3)
#     for file in files:
#         if 'merged' in file and 'clang' in file:
#             training.append(os.path.join(path_train3, file))
        
    
#     print("Generating training data...")
#     with open('datasets/train.jsonl', 'a') as f:
#         for js in generate_single(training, 'training'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish training data generation")
    
    
#     # get testing files for cross-opt
#     path_opt = 'libraries/data/optimization-amd64'
#     for lib in tqdm(['gmp','libtomcrypt','openssl']):
#         files = os.listdir(os.path.join(path_opt, lib))
#         for file in files:
#             if 'o0' in file and 'merged' in file:
#                 O0.append(os.path.join(path_opt, os.path.join(lib,file)))
#             if 'o1' in file and 'merged' in file:
#                 O1.append(os.path.join(path_opt, os.path.join(lib,file)))
#             if 'o2' in file and 'merged' in file:
#                 O2.append(os.path.join(path_opt, os.path.join(lib,file)))
#             if 'o3' in file and 'merged' in file:
#                 O3.append(os.path.join(path_opt, os.path.join(lib,file)))
    
#     # get testing files for O0 optimization
#     print("Generating O0 data...")
#     with open('datasets/test_o0.jsonl', 'a') as f:
#         for js in generate_single(O0, 'O0'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish O0 data generation")
    
#     # get testing files for O1 optimization
#     print("Generating O1 data...")
#     with open('datasets/test_o1.jsonl', 'a') as f:
#         for js in generate_single(O1, 'O1'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish O1 data generation")
    
#     # get testing files for O2 optimization
#     print("Generating O2 data...")
#     with open('datasets/test_o2.jsonl', 'a') as f:
#         for js in generate_single(O2, 'O2'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish O2 data generation")
    
#     # get testing files for O3 optimization
#     print("Generating O3 data...")
#     with open('datasets/test_o3.jsonl', 'a') as f:
#         for js in generate_single(O3, 'O3'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish O3 data generation")
    
#     # get testing files for cross-compiler
#     path_compiler = 'libraries/data/compiler'
#     for lib in ['gmp','libtomcrypt','openssl']:
#         files = os.listdir(os.path.join(path_compiler, lib))
#         for file in files:
#             if 'merged' in file and 'gcc' in file:
#                 gcc.append(os.path.join(path_compiler, lib, file))
#             if 'merged' in file and 'clang' in file:
#                 clang.append(os.path.join(path_compiler, lib, file))
#     print("Generating cross-compiler data...")
#     with open('datasets/test_gcc.jsonl', 'a') as f:
#         for js in generate_single(gcc, 'gcc'):
#             f.write(json.dumps(js) + '\n')
#     with open('datasets/test_clang.jsonl', 'a') as f:
#         for js in generate_single(clang, 'clang'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish cross-compiler data generation")
    
#     # get testing files for obfuscation
#     path_obfuscation = 'libraries/data/obfuscation'
#     for lib in ['gmp','libtomcrypt','openssl']:
#         files = os.listdir(os.path.join(path_obfuscation, lib))
#         for file in files:
#             if 'merged' in file and 'g-bcf.so' in file:
#                 obf_bcf.append(os.path.join(path_obfuscation, lib, file))
#             if 'merged' in file and 'g-fla.so' in file:
#                 obf_fla.append(os.path.join(path_obfuscation, lib, file))
#             if 'merged' in file and 'g-sub-fla-bcf.so' in file:
#                 obf_all.append(os.path.join(path_obfuscation, lib, file))
#             if 'merged' in file and 'g-sub.so' in file:
#                 obf_sub.append(os.path.join(path_obfuscation, lib, file))
#             if 'merged' in file and 'g.so' in file:
#                 obf_none.append(os.path.join(path_obfuscation, lib, file))
#     print("Generating cross-obf data...")
#     with open('datasets/test_obf_bcf.jsonl', 'a') as f:
#         for js in generate_single(obf_bcf, 'obf_bcf'):
#             f.write(json.dumps(js) + '\n')
#     with open('datasets/test_obf_fla.jsonl', 'a') as f:
#         for js in generate_single(obf_fla, 'obf_fla'):
#             f.write(json.dumps(js) + '\n')
#     with open('datasets/test_obf_all.jsonl', 'a') as f:
#         for js in generate_single(obf_all, 'obf_all'):
#             f.write(json.dumps(js) + '\n')
#     with open('datasets/test_obf_sub.jsonl', 'a') as f:
#         for js in generate_single(obf_sub, 'obf_sub'):
#             f.write(json.dumps(js) + '\n')
#     with open('datasets/test_obf_none.jsonl', 'a') as f:
#         for js in generate_single(obf_none, 'obf_none'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish cross-obf data generation")

#  # get testing files for cross-architecture
#     path_compiler = 'libraries/data/optimization-cross-arch/l-openssl-openssl'
#     files = os.listdir(path_compiler)
#     for file in files:
#         if 'merged' in file and '-gcc-' in file:
#             x86_64.append(os.path.join(path_compiler, file))
#         if 'merged' in file and 'arm' in file:
#             arm.append(os.path.join(path_compiler, file))
#         if 'merged' in file and 'gcc32' in file:
#             x86_32.append(os.path.join(path_compiler, file))
#         if 'merged' in file and 'mips' in file:
#             mips.append(os.path.join(path_compiler, file))
#         if 'merged' in file and 'powerpc' in file:
#             powerpc.append(os.path.join(path_compiler, file))
#     print("Generating cross-compiler data...")
#     with open('datasets/test_x86_64.jsonl', 'a') as f:
#         for js in generate_single(x86_64, 'x86_64'):
#             f.write(json.dumps(js) + '\n')
#     with open('datasets/test_arm.jsonl', 'a') as f:
#         for js in generate_single(arm, 'arm'):
#             f.write(json.dumps(js) + '\n')
#     with open('datasets/test_x86_32.jsonl', 'a') as f:
#         for js in generate_single(x86_32, 'x86_32'):
#             f.write(json.dumps(js) + '\n')
#     with open('datasets/test_mips.jsonl', 'a') as f:
#         for js in generate_single(mips, 'mips'):
#             f.write(json.dumps(js) + '\n')
#     with open('datasets/test_powerpc.jsonl', 'a') as f:
#         for js in generate_single(powerpc, 'powerpc'):
#             f.write(json.dumps(js) + '\n')
#     print("Finish cross-compiler data generation")

    ### generate training set with cfg for baselines
    train_cfg = []
#     path_train = 'datasets/firmware_images'
#     libs = [
#         'l-busybox-busybox_unstripped', 'l-sqlite-sqlite3',
#         'l-coreutils-coreutils', 'l-curl-curl',
#         'l-ImageMagick-magick', 'l-putty-puttygen'
#     ]
#     for lib in libs:
#         files = os.listdir(os.path.join(path_train, lib))
#         for file in files:
#             if 'merged' in file:
#                 train_cfg.append(os.path.join(path_train, lib,file))
                
#     ## add obfuscated code
#     path_train2 = 'libraries/data/obfuscation/imagemagick'
#     files = os.listdir(path_train2)
#     for file in files:
#         if 'merged' in file:
#             train_cfg.append(os.path.join(path_train2, file))

#     ## add clang code
#     path_train3 = 'libraries/data/compiler/imagemagick'
#     files = os.listdir(path_train3)
#     for file in files:
#         if 'merged' in file and 'clang' in file:
#             train_cfg.append(os.path.join(path_train3, file))
        
    
#     print("Generating training data...")
#     with open('datasets/train_cfg.jsonl', 'a') as f:
#         for js in generate_single(train_cfg, 'training', add_compiler_token = False, require_cfg=True):
#             f.write(json.dumps(js) + '\n')
#     print("Finish training data generation")
        

if __name__ == '__main__':
    main()