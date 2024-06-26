from easydict import EasyDict
import logging
import sys
logging.basicConfig(level=logging.ERROR)
# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================

collector_env_num = 8
# 多卡 训练 修改gpu_num 和 multi_gpu=True, main函数中的启动方式
# https://github.com/opendilab/LightZero/issues/196
gpu_num = 2
n_episode = int(8*gpu_num)#8

evaluator_env_num = 100
num_simulations = 40


update_per_collect = 100
batch_size = 20*gpu_num#16# 256

# 使用 efficientzero 那么减少max_env_step 试试
max_env_step =int(4e4)# int(6e3)# int(1e5) #max_env_step * num_simulations /num_unroll_steps =learner.train_iter=2000
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# 所有配置和mazegame_muzero_config_for_test.py 保持一致
# ==============================================================
grid_size = 10
frame_stack_num=3#1#3
mygrid_efficientzero_config = dict(
    exp_name=f'data_mz_ctree/mygrid_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_ddp_{gpu_num}gpu_seed0',
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
    ),
    policy=dict(
        
        model=dict(
            observation_shape=(3*frame_stack_num,grid_size,grid_size),#16,#4,
            channel_last=False,
            image_channel=3,#1,
            action_space_size=grid_size*grid_size,#4,#2,
            #model_type='conv',#'mlp', 
            #lstm_hidden_size=128,
            #latent_state_dim=128,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
            frame_stack_num=frame_stack_num,
            num_res_blocks=8,#2,
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
        multi_gpu=True,
        num_unroll_steps=5,
        td_steps=5,

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
        eval_freq=int(2e2),#int(50),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

save_freq_dict={
    'learn': {
        'learner': {
            'hook': {
                'log_show_after_iter': 2000,
                'save_ckpt_after_iter': 2000,   # Set this to your desired frequency
                'save_ckpt_after_run': False,#True,
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
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
mygrid_efficientzero_create_config = EasyDict(mygrid_efficientzero_create_config)
create_config = mygrid_efficientzero_create_config

if __name__ == "__main__":
    # Users can use different train entry by specifying the entry_type.
    entry_type = "train_muzero"  # options={"train_muzero", "train_muzero_with_gym_env"}

    if entry_type == "train_muzero":
        from lzero.entry import train_muzero ,eval_muzero
    elif entry_type == "train_muzero_with_gym_env":
        """
        The ``train_muzero_with_gym_env`` entry means that the environment used in the training process is generated by wrapping the original gym environment with LightZeroEnvWrapper.
        Users can refer to lzero/envs/wrappers for more details.
        """
        from lzero.entry import train_muzero_with_gym_env as train_muzero
        from lzero.entry import eval_muzero_with_gym_env as eval_muzero
    if len(sys.argv)>=3 and sys.argv[1]=='eval' :
        #print(sys.argv)
        print(f"eval模式")
        res = eval_muzero(
            input_cfg=[main_config, create_config],
            seed= 0,
            model= None,
            print_seed_details= False,
            model_path =sys.argv[2],# '/kaggle/working/LightZero/data_mz_ctree/mymaze_muzero_ns25_upc100_rr0_seed0_240301_065201/ckpt/ckpt_best.pth.tar',
            num_episodes_each_seed= 1
            )
        print(res)
    
    elif len(sys.argv)>=3 and sys.argv[1]=='train' :
        """
        单机多卡训练的启动方式:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=2 ./LightZero/zoo/atari/config/atari_muzero_multigpu_ddp_config.py
        """
        from ding.utils import DDPContext
        from lzero.entry import train_muzero
        from lzero.config.utils import lz_to_ddp_config
        with DDPContext():
            main_config = lz_to_ddp_config(main_config)
            print(f"ddp单机多卡 带pretrained model ckpt的继续train模式")
            train_muzero([main_config, create_config], seed=0, model_path =sys.argv[2],max_env_step=max_env_step)
    else :
        """
        单机多卡训练的启动方式:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=2 ./LightZero/zoo/atari/config/atari_muzero_multigpu_ddp_config.py
        """
        from ding.utils import DDPContext
        from lzero.entry import train_muzero
        from lzero.config.utils import lz_to_ddp_config
        with DDPContext():
            main_config = lz_to_ddp_config(main_config)
            print(f"ddp单机多卡 从头开始的train模式")
            train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)

