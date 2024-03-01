from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 26
update_per_collect = 50
batch_size = 16#256
max_env_step = int(1e4)# int(1e5)
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

mymaze_muzero_config = dict(
    exp_name=f'data_mz_ctree/mymaze_muzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0',
    env=dict(
        env_id='MyMaze-v1',
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=[1,4,4],#4,
            action_space_size=4,#2,
            model_type='mlp', 
            lstm_hidden_size=128,
            latent_state_dim=128,
            self_supervised_learning_loss=True,  # NOTE: default is False.
            discrete_action_encoding_type='one_hot',
            norm_type='BN', 
        ),
        cuda=True,
        env_type='not_board_games',
        action_type='varied_action_space',
        game_segment_length=50,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        lr_piecewise_constant_decay=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e1),
        replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

mymaze_muzero_config = EasyDict(mymaze_muzero_config)
main_config = mymaze_muzero_config

mymaze_muzero_create_config = dict(
    env=dict(
        type='mymaze_lightzero',
        import_names=['zoo.classic_control.grid_world.envs.gridworld_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
mymaze_muzero_create_config = EasyDict(mymaze_muzero_create_config)
create_config = mymaze_muzero_create_config

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

    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
    #res = eval_muzero(
    #    input_cfg=[main_config, create_config],
    #    seed= 0,
    #    model= None,
    #    model_path = '/kaggle/working/LightZero/data_mz_ctree/cartpole_muzero_ns25_upc100_rr0_seed0_240229_114840/ckpt/ckpt_best.pth.tar',
    #    num_episodes_each_seed= 1,
    #    print_seed_details= False)
    #print(res)
