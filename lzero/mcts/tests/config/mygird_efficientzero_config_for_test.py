from easydict import EasyDict
import logging
import sys
logging.basicConfig(level=logging.ERROR)
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================

collector_env_num = 8
# 多卡
# https://github.com/opendilab/LightZero/issues/196

n_episode = 8

evaluator_env_num = 50
num_simulations = 40 #


update_per_collect = 50
batch_size = 20#256#100#16# 256

# 使用 efficientzero 那么减少max_env_step 试试
max_env_step =int(2e3)# int(6e5)# int(1e5) #max_env_step * num_simulations /num_unroll_steps =learner.train_iter=2000
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# 所有配置和mazegame_muzero_config_for_test.py 保持一致
# ==============================================================
grid_size = 10
frame_stack_num=1#3
mygrid_efficientzero_config = dict(
    exp_name=f'data_mz_ctree/mygrid_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_id='MyGrid-v1',
        continuous=False,
        obs_shape=(3, grid_size, grid_size),
        channel_last=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        # add 
        replay_path=None,
        save_replay_gif=False,
        replay_path_gif='./replay_gif',
    ),
    policy=dict(
        # 根据issue 这里增加sampled_algo , gumble_algo,use_ture_chance_label_in_chance_encoder,model_path
        sampled_algo=False,
        gumbel_algo=False,
         mcts_ctree=True,
        use_ture_chance_label_in_chance_encoder=False,
        # add moel_path here
        model_path="/kaggle/working/LightZero/data_mz_ctree/mygrid_efficientzero_ns40_upc50_rr0_seed0/ckpt/ckpt_best.pth.tar" ,
        # add end 20240423
        
        model=dict(
            observation_shape=(3*frame_stack_num,grid_size,grid_size),#16,#4,
            channel_last=False,
            image_channel=3,#1,
            action_space_size=grid_size*grid_size,#4,#2,
            model_type='conv',#'mlp', 
            #lstm_hidden_size=128,
            #latent_state_dim=128,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
            frame_stack_num=frame_stack_num,
            
            num_res_blocks=2,#2,
            num_channels=64,#32,
            # add 虽然默认是300 但是还是明确写出来好
            # # support_scale 和 categorical_distribution=True 搭配使用 categorical_distribution 默认是True
            # https://github.com/valkryhx/LightZero/blob/4d73183c5b3a40cba3a5a66bf792bb87016d92d2/lzero/policy/muzero.py#L52C13-L52C86
            support_scale=300,
            reward_support_size=300*2 + 1,
            value_support_size=300*2 + 1,
            # add 默认是True 但这里也明确写出来 
            categorical_distribution=True,
            
        ),
        cuda=True,

        num_unroll_steps=3,
        td_steps=5,
        
        # add
        gray_scale=False,
        transform2string=False,
        discount_factor=1,
        #add end

        env_type='not_board_games',
        action_type='varied_action_space',
        game_segment_length=5,#50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        #ssl_loss_weight=2,  # NOTE: default is 0.
        grad_clip_value=0.5,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(50),#int(50),
        
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
save_freq_dict={
    'learn': {
        'learner': {
            'hook': {
                'log_show_after_iter': 200,
                'save_ckpt_after_iter': 200,   # Set this to your desired frequency
                'save_ckpt_after_run': False,
            },
        },
    },
}


mygrid_efficientzero_config = EasyDict(mygrid_efficientzero_config )
# https://github.com/opendilab/LightZero/issues/196#issuecomment-2006133687
mygrid_efficientzero_config.policy.update(save_freq_dict)

main_config = mygrid_efficientzero_config

mygrid_efficientzero_create_config = dict(
    env=dict(
        type='mygrid_lightzero',
        import_names=['zoo.classic_control.grid.envs.grid_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',#'muzero',
        import_names=['lzero.policy.efficientzero'],#['lzero.policy.muzero'],
    ),
)
mygrid_efficientzero_create_config = EasyDict(mygrid_efficientzero_create_config)
create_config = mygrid_efficientzero_create_config
