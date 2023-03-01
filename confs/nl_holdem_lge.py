{
    'env': 'NlHoldemEnvWithOpponent',
    'sample_batch_size': 50,
    'train_batch_size': 1000,
    'num_workers': 89,
    'num_envs_per_worker': 1,
    #'broadcast_interval': 5,
    #'max_sample_requests_in_flight_per_worker': 1,
    #'num_data_loader_buffers': 4,
    'num_gpus': 1,
    'gamma': 1,
    'entropy_coeff': 1.0,
    'vf_loss_coeff': 1e-3,
    'lr': 3e-4,
    'model':{
        'custom_model': 'NlHoldemNet',
        'max_seq_len': 20,
        'custom_options': {
        },
    },
    "env_config":{
        'custom_options': {
            'weight':"default",
            "cut":[[0,12],[13,25],[26,38],[39,51],[52,53],[53,54]],
            'epsilon': 0.15,
            'tracker_n': 1000,
            'conut_bb_rather_than_winrate': 2,
            'use_history': True,
            'use_cardnum': True,
            'history_len': 20,
        },
    }
}