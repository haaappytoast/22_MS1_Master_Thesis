# COMMAND: python main.py config/humanoid_view.py --ckpt ckpt_temp --test
env_cls = "HumanoidView"  #  HumanoidViewTennis
env_params = dict(
    episode_length = 300,
    motion_file = "assets/retargeted/block_MIXAMO/cml@idleblock (1).npy",
    goal_embedding = False
)

training_params = dict(
    max_epochs = 10000,
    save_interval = 2000,
    terminate_reward = -1
)

discriminators = {
    "_/full": dict(
        parent_link = None,
    )
}