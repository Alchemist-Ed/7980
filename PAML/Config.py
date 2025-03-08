config_db = {
    'input_dir': 'data_process/dbook',
    'output_dir': 'res/dbook',
    'dataset': 'dbook',
    #'use_cuda': True,
    'use_cuda': False,

    'gpu': '2',

    # user
    'num_location': 453,
    'num_fea_user': 1,
    'use_fea_user': 1,

    # item
    'num_publisher': 1789,
    'num_author': 10713,
    'num_fea_item': 2,
    'use_fea_item': 1,

    'embedding_dim': 32,

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    'dropout': 0.2,
    ### original value 0.2

    'local_update': 2,
    'outer_lr': 1e-4,
    'local_lr': 1e-3,

    # early stopping
    'patience':5,
    'min_delta':1e-3,

    'weight_decay': 3e-3,
    'batch_size': 64,  # for each batch, the number of tasks
    'num_epoch': 20,
    # 'num_epoch': 118,
   

    # option
    'social_num': 5,
    'implicit_num': 5,
    'use_coclick': False,# Based on CF, useless, do not change  
    # 'coclick_num': 20,
}

states = ["meta_training", "warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing"]

