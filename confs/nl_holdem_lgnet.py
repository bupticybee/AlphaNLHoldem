{
    'env': 'NlHoldemEnvWithOpponent',
    'sample_batch_size': 50,
    'train_batch_size': 1000,
    'num_workers': 90,
    'num_envs_per_worker': 1,
    #'broadcast_interval': 5,
    #'max_sample_requests_in_flight_per_worker': 1,
    #'num_data_loader_buffers': 4,
    'num_gpus': 1,
    'gamma': 1,
    'entropy_coeff': 1e-1,
    'lr': 3e-4,
    'model':{
        'custom_model': 'NlHoldemLgNet',
        'max_seq_len': 20,
        'custom_options': {
        },
    },
    "env_config":{
        'custom_options': {
            "rwd_ratio": 1.0
        },
    }
}