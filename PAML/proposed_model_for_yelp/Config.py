# config_db = {
#     #modify for yelp
#     # 'input_dir': 'data_process/dbook',
#     # 'output_dir': 'res/dbook',
#     # 'dataset': 'dbook',
#     'input_dir': 'data_process/yelp',
#     'output_dir': 'res/yelp',
#     'dataset': 'yelp',
    
#     ## 'use_cuda': True,
#     'use_cuda': False,

#     'gpu': '2',
# # modify
#     # # user
#     # 'num_location': 453,
#     # 'num_fea_user': 1,
#     # 'use_fea_user': 1,

#     # # item
#     # 'num_publisher': 1789,
#     # 'num_author': 10713,
#     # 'num_fea_item': 2,
#     # 'use_fea_item': 1,
# # above are for dbook

#  ## User-related features in yelp
#     'num_fans': 691,                  # Number of unique fan levels in the dataset
#     'num_avgrating': 401,             # Number of unique average rating levels in the dataset
#     'num_fea_user': 2,                # Number of features for each user (fan and avg rating)
#     'use_fea_user': 1,                # Enable usage of user features

#     ## Item-related features
#     'num_stars': 396,                 # Number of unique star ratings in the dataset
#     'num_postalcode': 3362,           # Number of unique postal codes in the dataset
#     'num_fea_item': 2,                # Number of features for each item (stars and postal code)
#     'use_fea_item': 1, 

# # end modify

#     'embedding_dim': 32,

#     'first_fc_hidden_dim': 64,
#     'second_fc_hidden_dim': 64,
#     'dropout': 0.2,

#     'local_update': 1,
#     'lr': 1e-3,
    
#     'local_lr': 1e-3,

#     'weight_decay': 3e-3,
#     'batch_size': 64,  # for each batch, the number of tasks
#     'num_epoch': 170,
#     # 'num_epoch': 118,
   

#     # option
#     'social_num': 5,
#     'implicit_num': 5,
#     'use_coclick': False,# Based on CF, useless, do not change  
#     # 'coclick_num': 20,
# }


config_yelp = {
    # #modify for yelp
    # # 'input_dir': 'data_process/dbook',
    # # 'output_dir': 'res/dbook',
    # # 'dataset': 'dbook',
    'input_dir': 'data_process/yelp',
    'output_dir': 'res/yelp',
    'dataset': 'yelp',
    
    ## 'use_cuda': True,
    'use_cuda': False,

    'gpu': '2',
# # modify
#     # user
#     'num_location': 453,
#     'num_fea_user': 1,
#     'use_fea_user': 1,

#     # item
#     'num_publisher': 1789,
#     'num_author': 10713,
#     'num_fea_item': 2,
#     'use_fea_item': 1,
# # above are for dbook

 # User-related features in yelp
    'num_fans': 277,                  
    'num_avgrating': 366,             
    'num_fea_user': 2,                # Number of features for each user (fan and avg rating)
    'use_fea_user': 2,                

    ## Item-related features
    'num_stars': 9,                 
    'num_postalcode': 2770,           
    'num_fea_item': 2,                # Number of features for each item (stars and postal code)
    'use_fea_item': 2, 

# end modify

    'embedding_dim': 32,

    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,
    'dropout': 0.2,

    'local_update': 1,
    'lr': 1e-3,
    
    'local_lr': 1e-3,

    'weight_decay': 3e-3,
    'batch_size': 64,  # for each batch, the number of tasks
    # 'num_epoch': 170,
    'num_epoch': 20,
   

    # option
    'social_num': 5,
    'implicit_num': 5,
    'use_coclick': False,# Based on CF, useless, do not change  
    # 'coclick_num': 20,
}


states = ["meta_training", "warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing"]

