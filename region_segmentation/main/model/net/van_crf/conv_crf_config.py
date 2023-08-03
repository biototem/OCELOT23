from basic import config


crf_config = {
    'filter_size': 7,
    'blur': 4,
    'merge': True,
    'norm': 'none',
    'weight': 'vector',
    'unary_weight': 1,
    'weight_init': 0.2,
    'trainable': False,
    'convcomp': False,
    'logsoftmax': True,
    'softmax': True,
    # 这个决定最后输出是否归一化
    'final_softmax': True,
    'pos_feats': {'sdims': 3, 'compat': 3},
    'col_feats': {'sdims': 80, 'schan': 13, 'compat': 10, 'use_bias': False},
    'trainable_bias': True,
    'pyinn': False
}

class_nums = config['dataset.class_num']

device = config['train.device']
