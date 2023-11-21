import os, time
import importlib
from collections import namedtuple

import env
from models import ACModel, Discriminator, ACModel_gembed

import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str,
    help="Configure file used for training. Please refer to files in `config` folder.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--test", action="store_true", default=False,
    help="Run visual evaluation.")
parser.add_argument("--seed", type=int, default=42,
    help="Random seed.")
parser.add_argument("--device", type=int, default=0,
    help="ID of the target GPU device for model running.")
parser.add_argument('--headless', action='store_true',
    help='Run headless without creating a viewer window')
parser.add_argument("--server", type=str, default=None,
    help="server docker name")
parser.add_argument("--resume", type=str, default=None,
    help="resume with existing checkpoint")

settings = parser.parse_args()

    
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)
torch.manual_seed(settings.seed)
torch.cuda.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


FPS = 30
FRAMESKIP = 2
CONTROL_MODE = "position"
HORIZON = 8
NUM_ENVS = 512
BATCH_SIZE = 256 #HORIZON*NUM_ENVS//16
OPT_EPOCHS = 5
ACTOR_LR = 5e-6
CRITIC_LR = 1e-4
GAMMA = 0.95
GAMMA_LAMBDA = GAMMA * 0.95

TRAINING_PARAMS = dict(
    max_epochs = 10000,
    save_interval = None,
    terminate_reward = -1

)

EVAL = False

def logger(obs, rews, info):
    buffer = dict(r=[])
    buffer_disc = {
    name: dict(fake=[], seq_len=[]) for name in env.discriminators.keys()  # Dict[str, DiscriminatorConfig]
    }

    has_goal_reward = env.rew_dim > 0
    if has_goal_reward:
        buffer["r"].append(rews)
    
    multi_critics = env.reward_weights is not None
    if multi_critics:
        rewards = torch.zeros(num_envs, len(env.discriminators)+env.rew_dim)                      # [num_envs X 8, reward 개수]
    else:
        rewards = torch.zeros(num_envs, len(env.discriminators))
    
    fakes = info["disc_obs"]
    disc_seq_len = info["disc_seq_len"]

    for name, fake in fakes.items():
        buffer_disc[name]["fake"].append(fake)
        buffer_disc[name]["seq_len"].append(disc_seq_len[name])

    with torch.no_grad():
        # 1. Reward related to discriminators
        disc_data_raw = []
        for name, data in buffer_disc.items():              # data: fake, real, seq_len
            disc = model.discriminators[name]   
            fake = torch.cat(data["fake"])                  # [N * HORIZON, 2/5, 56/49] / len(data["fake"]) = HORIZON
            seq_len = torch.cat(data["seq_len"])            # [N * HORIZON]
            end_frame = seq_len - 1
            disc_data_raw.append((name, disc, fake, end_frame))
        
        for name, disc, ob, seq_end_frame in disc_data_raw:
            r = (disc(ob, seq_end_frame).clamp_(-1, 1).mean(-1, keepdim=True)) # clamp shape: [num_envs X 8, 32: ensemble]   / r.shape: [num_envs X 8, 1]
            if rewards is None:
                rewards = r
            else:
                rewards[:, env.discriminators[name].id] = r.squeeze_(-1)    # id: 0 / 1

        # 2. Reward related to goal      
        if has_goal_reward:
            rewards_task = torch.cat(buffer["r"])                           # [num_envs X 8, 2] / buffer["r"]: [8, 512, 2]
            if rewards is None:
                rewards = rewards_task
            else:
                rewards[:, -rewards_task.size(-1):] = rewards_task          # 마지막 reward 들 (개수만큼)
        
        else:
            rewards_task = None
        
        print("Reward: {}".format("/".join(list(map("{:.4f}".format, rewards.mean(0).cpu().tolist()  )))))  # 모든 env에서의 mean값
        
    return rewards.sum(dim=1), rewards.mean(0)      # 각 env에 대한 reward 값


def test(env, model):
    #### for evaluation
    if EVAL:
        ep = np.load(config.discriminators["usermotion1/upper"]["motion_file"], allow_pickle=True).item()['rotation']['arr'].shape[0]   # 300
        np_reward = np.empty([ep, 3]) if env.sensor_ablation else np.empty([ep, 4]) 
        curr_ep = 0
        count = 0
        dir_name = settings.ckpt.split("/")[0] + "/eval"
        # Check if the directory already exists
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print(f"Directory {dir_name} created.")
        else:
            print(f"Directory {dir_name} already exists.")
        ####

    model.eval()
    env.reset()
    rewards_tot = torch.zeros((num_envs,), dtype=torch.float64)                   # 4개의 env
    counter = torch.zeros((num_envs,), dtype=torch.int32)
    while not env.request_quit:
        obs, info = env.reset_done()
        seq_len = info["ob_seq_lens"]
        actions = model.act(obs, seq_len-1)
        obs, rews, _, info = env.step(actions)                                 # apply_actions -> do_simulation -> refresh_tensors -> observe()


        reward, sep_reward = logger(obs, rews, info)

        if EVAL:
            if curr_ep < ep:
                np_reward[curr_ep] = sep_reward
                curr_ep += 1
        
        rewards_tot += reward
        counter += torch.where(info["terminate"] == False, 1, 0).cpu()  # counter 더해주기
        # terminate 되었다면
        terminated_env = (info["terminate"] == True).nonzero().view(-1).cpu()
        if EVAL:
            if curr_ep == ep or len(terminated_env):
                time.time()
                print("\n================" + str(curr_ep) + ": save reward_trial " + str(count), "================\n")
                np.save(dir_name + "/trial" + str(count) + "_avg_reward", np_reward[..., :curr_ep])
                #tot_return = np.array([rewards_tot[e] / counter[e].item(), rewards_tot[e]])
                #print(tot_return)
                #np.save(settings.ckpt+ "/trial" + str(count) + "_avg_reward+total_return", np_reward)
                curr_ep = 0
                count += 1
        
        # terminate 된 env가 있다면
        if len(terminated_env):
            for e in terminated_env:
                print("ENV {:d} / avg reward: {:.4f} / total return: {:.4f}".format(e, rewards_tot[e] / counter[e].item(), rewards_tot[e]))
            counter[terminated_env] = torch.zeros_like(terminated_env, dtype=torch.int32)   # counter는 0dmfh gownjdigka!

        # write txt file
        # while terminated_env < 300:

        #     np_reward = np.vstack([np_reward, sep_reward])
        #     print(np_reward)
        #     pass

def train(env, model, ckpt_dir, training_params, reward_coeff = None, resumed_optimizer=None):
    if ckpt_dir is not None:
        logger = SummaryWriter(ckpt_dir)
    else:
        logger = None

    optimizer = torch.optim.Adam([
        {"params": model.actor.parameters(), "lr": ACTOR_LR},
        {"params": model.critic.parameters(), "lr": CRITIC_LR}
    ])
    ac_parameters = list(model.actor.parameters()) + list(model.critic.parameters())
    disc_optimizer = {name: torch.optim.Adam(disc.parameters(), 1e-5) for name, disc in model.discriminators.items()}

    # load_state_dict of optimizers
    if resumed_optimizer is not None:
        # optimize model optim
        optimizer.load_state_dict(resumed_optimizer['model_optim'])

        for name, _ in model.discriminators.items():
            assert(name in resumed_optimizer.keys()), "Optimizer keys should be matched"
            disc_optimizer[name].load_state_dict(resumed_optimizer[name])

    buffer = dict(
        s=[], a=[], v=[], lp=[], v_=[], not_done=[], terminate=[],
        ob_seq_len=[]
    )
    multi_critics = env.reward_weights is not None
    if multi_critics:
        buffer["reward_weights"] = []
    has_goal_reward = env.rew_dim > 0
    if has_goal_reward:
        buffer["r"] = []

    # each for discriminator
    buffer_disc = {
        name: dict(fake=[], real=[], seq_len=[]) for name in env.discriminators.keys()
    }
    # each for discriminator
    real_losses, fake_losses = {n:[] for n in buffer_disc.keys()}, {n:[] for n in buffer_disc.keys()}
    
    epoch = 0
    model.eval()
    env.reset()
    tic = time.time()
    while not env.request_quit:
        with torch.no_grad():
            obs, info = env.reset_done()    # reset_envs -> reset_goals -> 다음 obs observe
            seq_len = info["ob_seq_lens"]   # for each environment, how many sequences character observed
            reward_weights = info["reward_weights"]
            actions, values, log_probs = model.act(obs, seq_len-1, stochastic=True)
            obs_, rews, dones, info = env.step(actions)     # NEXT OBS!!! (apply_actions -> do_simulation -> reward -> termination_check -> overtime_check -> observe)
            log_probs = log_probs.sum(-1, keepdim=True)
            not_done = (~dones).unsqueeze_(-1)
            terminate = info["terminate"]

            # Discriminator observations
            fakes = info["disc_obs"]
            reals = info["disc_obs_expert"]
            disc_seq_len = info["disc_seq_len"]

            values_ = model.evaluate(obs_, seq_len)     # get next values from obs of next step

        buffer["s"].append(obs)
        buffer["a"].append(actions)
        buffer["v"].append(values)
        buffer["lp"].append(log_probs)
        buffer["v_"].append(values_)
        buffer["not_done"].append(not_done)
        buffer["terminate"].append(terminate)
        buffer["ob_seq_len"].append(seq_len)
        if has_goal_reward:
            buffer["r"].append(rews)
        if multi_critics:
            buffer["reward_weights"].append(reward_weights)
        for name, fake in fakes.items():
            buffer_disc[name]["fake"].append(fake)
            buffer_disc[name]["real"].append(reals[name])
            buffer_disc[name]["seq_len"].append(disc_seq_len[name])

        if len(buffer["s"]) == HORIZON:
            with torch.no_grad():
                disc_data_training = []
                disc_data_raw = []
                for name, data in buffer_disc.items():
                    disc = model.discriminators[name]
                    fake = torch.cat(data["fake"])
                    real = torch.cat(data["real"])
                    seq_len = torch.cat(data["seq_len"])
                    end_frame = seq_len - 1
                    disc_data_raw.append((name, disc, fake, end_frame))

                    length = torch.arange(fake.size(1), 
                        dtype=end_frame.dtype, device=end_frame.device)
                    mask = length.unsqueeze_(0) <= end_frame.unsqueeze(1)
                    disc.ob_normalizer.update(fake[mask])
                    disc.ob_normalizer.update(real[mask])

                    ob = disc.ob_normalizer(fake)
                    ref = disc.ob_normalizer(real)
                    disc_data_training.append((name, disc, ref, ob, end_frame))

            model.train()
            n_samples = 0
            for name, disc, ref, ob, seq_end_frame_ in disc_data_training:
                real_loss = real_losses[name]
                fake_loss = fake_losses[name]
                opt = disc_optimizer[name]
                if len(ref) != n_samples:                        
                    n_samples = len(ref)                        # n_samples = 4096 (PPO replay buffer size)
                    idx = torch.randperm(n_samples) 
                for batch in range(n_samples//BATCH_SIZE):      # BATCH_SIZE = 256 (PPO batch size)
                    sample = idx[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                    r = ref[sample]                             # [4096, 3, 56] -> [256, 3, 56]
                    f = ob[sample]                              # [4096, 3, 56] -> [256, 3, 56]
                    
                    seq_end_frame = seq_end_frame_[sample]

                    score_r = disc(r, seq_end_frame, normalize=False)
                    score_f = disc(f, seq_end_frame, normalize=False)
                
                    loss_r = torch.nn.functional.relu(1-score_r).mean()     # reference motions
                    loss_f = torch.nn.functional.relu(1+score_f).mean()     # simulated motions

                    with torch.no_grad():
                        alpha = torch.rand(r.size(0), dtype=r.dtype, device=r.device)
                        alpha = alpha.view(-1, *([1]*(r.ndim-1)))
                        interp = alpha*r+(1-alpha)*f
                    interp.requires_grad = True
                    with torch.backends.cudnn.flags(enabled=False):
                        score_interp = disc(interp, seq_end_frame, normalize=False)
                    grad = torch.autograd.grad(
                        score_interp, interp, torch.ones_like(score_interp),
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gp = grad.reshape(grad.size(0), -1).norm(2, dim=1).sub(1).square().mean()
                    l = loss_f + loss_r + 10*gp
                    l.backward()
                    opt.step()
                    opt.zero_grad()

                    real_loss.append(score_r.mean().item())
                    fake_loss.append(score_f.mean().item())


            model.eval()
            with torch.no_grad():
                terminate = torch.cat(buffer["terminate"])
                if multi_critics:
                    reward_weights = torch.cat(buffer["reward_weights"])
                    rewards = torch.zeros_like(reward_weights)
                else:
                    reward_weights = None
                    rewards = None
                for name, disc, ob, seq_end_frame in disc_data_raw:     # [4096, 3, 56]
                    r = (disc(ob, seq_end_frame).clamp_(-1, 1)
                            .mean(-1, keepdim=True))
                    if rewards is None:
                        rewards = r
                    else:
                        rewards[:, env.discriminators[name].id] = r.squeeze_(-1)
                if has_goal_reward:
                    rewards_task = torch.cat(buffer["r"])
                    if rewards is None:
                        rewards = rewards_task
                    else:
                        rewards[:, -rewards_task.size(-1):] = rewards_task
                else:
                    rewards_task = None
                rewards[terminate] = training_params.terminate_reward

                values = torch.cat(buffer["v"])
                values_ = torch.cat(buffer["v_"])
                if model.value_normalizer is not None:
                    values = model.value_normalizer(values, unnorm=True)
                    values_ = model.value_normalizer(values_, unnorm=True)
                values_[terminate] = 0
                rewards = rewards.view(HORIZON, -1, rewards.size(-1))
                values = values.view(HORIZON, -1, values.size(-1))
                values_ = values_.view(HORIZON, -1, values_.size(-1))

                not_done = buffer["not_done"]
                advantages = (rewards - values).add_(values_, alpha=GAMMA)
                for t in reversed(range(HORIZON-1)):
                    advantages[t].add_(advantages[t+1]*not_done[t], alpha=GAMMA_LAMBDA)

                advantages = advantages.view(-1, advantages.size(-1))
                returns = advantages + values.view(-1, advantages.size(-1))

                log_probs = torch.cat(buffer["lp"])
                actions = torch.cat(buffer["a"])
                states = torch.cat(buffer["s"])     # [4096, 860]
                ob_seq_lens = torch.cat(buffer["ob_seq_len"])
                ob_seq_end_frames = ob_seq_lens - 1

                sigma, mu = torch.std_mean(advantages, dim=0, unbiased=True)
                advantages = (advantages - mu) / (sigma + 1e-8) # (HORIZON x N_ENVS) x N_DISC
                
                length = torch.arange(env.ob_horizon, 
                    dtype=ob_seq_lens.dtype, device=ob_seq_lens.device)
                mask = length.unsqueeze_(0) < ob_seq_lens.unsqueeze(1)
                states_raw = model.observe(states, norm=False)[0]
                model.ob_normalizer.update(states_raw[mask])
                if model.value_normalizer is not None:
                    model.value_normalizer.update(returns)
                    returns = model.value_normalizer(returns)
                if multi_critics:
                    advantages = advantages.mul_(reward_weights)
                    # for logger only
                    rewards = rewards.view(*reward_weights.shape)
                    reward_tot = (rewards * reward_weights).sum(-1, keepdims=True).mean(0).item()
                    rewards = rewards.mean(0).cpu().tolist()
                    if rewards_task is not None:
                        rewards_task = rewards_task.mean(0).cpu().tolist()
                else:
                    rewards = rewards.view(-1, rewards.size(-1))
                    reward_tot = rewards.mean(0).item()                    
                    rewards = rewards.mean(0).cpu().tolist()

            n_samples = advantages.size(0)
            epoch += 1
            model.train()
            policy_loss, value_loss = [], []
            for _ in range(OPT_EPOCHS):
                idx = torch.randperm(n_samples)
                for batch in range(n_samples // BATCH_SIZE):
                    sample = idx[BATCH_SIZE * batch: BATCH_SIZE *(batch+1)]
                    s = states[sample]
                    a = actions[sample]
                    lp = log_probs[sample]
                    adv = advantages[sample]
                    v_t = returns[sample]
                    end_frame = ob_seq_end_frames[sample]

                    pi_, v_ = model(s, end_frame)
                    lp_ = pi_.log_prob(a).sum(-1, keepdim=True)

                    ratio = torch.exp(lp_ - lp)
                    clipped_ratio = torch.clamp(ratio, 1.0-0.2, 1.0+0.2)
                    pg_loss = -torch.min(adv*ratio, adv*clipped_ratio).sum(-1).mean()
                    vf_loss = (v_ - v_t).square().mean()

                    loss = pg_loss + 0.5*vf_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac_parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    policy_loss.append(pg_loss.item())
                    value_loss.append(vf_loss.item())
            model.eval()
            for v in buffer.values(): v.clear()
            for buf in buffer_disc.values():
                for v in buf.values(): v.clear()

            lifetime = env.lifetime.to(torch.float32).mean().item()
            policy_loss, value_loss = np.mean(policy_loss), np.mean(value_loss)
            print("Epoch: {}, Loss: {:.4f}/{:.4f}, Reward: {}, Lifetime: {:.4f} -- {:.4f}s".format(
                epoch, policy_loss, value_loss, "/".join(list(map("{:.4f}".format, rewards))), lifetime, time.time()-tic
            ))
            if logger is not None:
                logger.add_scalar("train/lifetime", lifetime, epoch)
                logger.add_scalar("train/reward", reward_tot, epoch)
                logger.add_scalar("train/loss_policy", policy_loss, epoch)
                logger.add_scalar("train/loss_value", value_loss, epoch)
                for name, r_loss in real_losses.items():
                    if r_loss: logger.add_scalar("score_real/{}".format(name), sum(r_loss)/len(r_loss), epoch)
                for name, f_loss in fake_losses.items():
                    if f_loss: logger.add_scalar("score_fake/{}".format(name), sum(f_loss)/len(f_loss), epoch)
                if rewards_task is not None: 
                    for i in range(len(rewards_task)):
                        logger.add_scalar("train/task_reward_{}".format(i), rewards_task[i], epoch)
                # discriminator loss
                len_reward_task = len(rewards_task) if rewards_task is not None else 0
                for i in range(len(rewards)-len_reward_task):
                    logger.add_scalar("train/reward_disc_{}".format(i), rewards[i], epoch)

            for v in real_losses.values(): v.clear()
            for v in fake_losses.values(): v.clear()
            
            if ckpt_dir is not None:
                state = None
                if epoch % 50 == 0:
                    state = dict(
                        model=model.state_dict()
                    )
                    optims= {
                        disc_name : disc_optimizer[disc_name].state_dict() for disc_name in disc_optimizer.keys()
                        } 
                    optims['model_optim'] = optimizer.state_dict()
                    torch.save(state, os.path.join(ckpt_dir, "ckpt"))
                    torch.save(optims, os.path.join(ckpt_dir, "ckpt_optims"))

                if epoch % training_params.save_interval == 0:
                    if state is None:
                        state = dict(model=model.state_dict())
                        optims= {
                        disc_name : disc_optimizer[disc_name].state_dict() for disc_name in disc_optimizer.keys()
                        } 
                        optims['model_optim'] = optimizer.state_dict()

                    torch.save(state, os.path.join(ckpt_dir, "ckpt-{}".format(epoch)))
                    torch.save(optims, os.path.join(ckpt_dir, "ckpt-{}_optims".format(epoch)))

                if epoch >= training_params.max_epochs: exit()
            tic = time.time()

if __name__ == "__main__":
    if settings.test:
        num_envs = 1
    else:
        num_envs = NUM_ENVS
        if settings.ckpt:
            if os.path.isfile(settings.ckpt) or os.path.exists(os.path.join(settings.ckpt, "ckpt")):
                raise ValueError("Checkpoint folder {} exists. Add `--test` option to run test with an existing checkpoint file".format(settings.ckpt))
            import shutil, sys
            os.makedirs(settings.ckpt, exist_ok=True)
            shutil.copy(settings.config, settings.ckpt)
            command_name = time.time() if settings.server == None else settings.server
            with open(os.path.join(settings.ckpt, "_command_{}.txt".format(command_name)), "w") as f:
                f.write("python " + " ".join(sys.argv))
                now = datetime.now()
                f.write("\n\n시작 날짜: {}".format(now.date()))
                f.write("\n시작 시각: {}:{}:{}".format(now.hour, now.minute, now.second))

    if os.path.splitext(settings.config)[-1] in [".npy", ".json", ".yaml"]:
        config = object()
        config.env_params = dict(
            motion_file = settings.config
        )
    else:
        spec = importlib.util.spec_from_file_location("config", settings.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    # if headless
    if settings.headless:
        config.env_params['graphics_device'] = -1

    if hasattr(config, "reward_coeff"):
        reward_coeff = config.reward_coeff
        # reward_coeff = namedtuple('x', reward_coeff.keys())(*reward_coeff.values())
    else:
        reward_coeff = dict(dummy = None)       # HumanoidVR 말고는 필요없음

    if hasattr(config, "sensor_input"):
        if settings.test:   # test 일 때는 testset 가져오기
            sensor_input = \
            {
                name: env.SensorInputConfig(**prop)
                for name, prop in config.sensor_input.items() if "test" in name
            }
        else:                # train 일 때는 trainset 가져오기
            sensor_input = \
            {
                name: env.SensorInputConfig(**prop)
                for name, prop in config.sensor_input.items() if "train" in name
            }
    else:
        sensor_input = {"dummy": env.SensorInputConfig()}
                
    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    if not TRAINING_PARAMS["save_interval"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    print(TRAINING_PARAMS)
    training_params = namedtuple('x', TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())

    if hasattr(config, "discriminators"):
        discriminators = {
            name: env.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }

    else:
        discriminators = {"_/full": env.DiscriminatorConfig()}
    if hasattr(config, "env_cls"):
        env_cls = getattr(env, config.env_cls)
    else:
        env_cls = env.ICCGANHumanoid
    print(env_cls, config.env_params)

    env = env_cls(num_envs, FPS, FRAMESKIP,
        control_mode=CONTROL_MODE,
        discriminators=discriminators,
        compute_device=settings.device, 
        sensor_inputs=sensor_input,
        ckpt=settings.ckpt,
        **config.env_params,
        **reward_coeff
    )
    if settings.test:
        if "view" in (config.env_cls).lower():
            pass
        else:
            if EVAL:

                env.episode_length = np.load(config.discriminators["usermotion1/upper"]["motion_file"], allow_pickle=True).item()['rotation']['arr'].shape[0]   # 300
            else:
                env.episode_length = 300

    value_dim = len(env.discriminators)+env.rew_dim         # critic 개수
    if env.goal_embedding:
        model = ACModel_gembed(env.state_dim, env.act_dim, env.goal_dim, env.upper_goal_dim, env.lower_goal_dim, value_dim)
    else:
        model = ACModel(env.state_dim, env.act_dim, env.goal_dim, value_dim)
    
    discriminators = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    })
    device = torch.device(settings.device)
    model.to(device)
    discriminators.to(device)
    model.discriminators = discriminators

    if settings.test:
        if settings.resume is not None:
            raise ValueError("This is test time. You can't use arguments of resume")

        if settings.ckpt is not None and os.path.exists(settings.ckpt):
            if os.path.isdir(settings.ckpt):
                ckpt = os.path.join(settings.ckpt, "ckpt")
            else:
                ckpt = settings.ckpt
                settings.ckpt = os.path.dirname(ckpt)
            if os.path.exists(ckpt):
                print("Load model from {}".format(ckpt))
                state_dict = torch.load(ckpt, map_location=torch.device(settings.device))
                model.load_state_dict(state_dict["model"])
        env.render()
        test(env, model)
    # train 일 때
    else:
        if "view" in (config.env_cls).lower():
            raise ValueError("{} can be only run in test time".format(config.env_cls))
        # resume 시키려면 
        if settings.resume:
            if (os.path.isfile(settings.ckpt)):
                raise ValueError("It should be folder name not checkpoint name!")

            if os.path.isfile(settings.resume) and os.path.exists(settings.resume):
                resume_ckpt = settings.resume
                state_dict = torch.load(resume_ckpt, map_location=torch.device(settings.device))      # loaded model
                model_dict = model.state_dict()                                                        # current model
                model.load_state_dict(state_dict['model'])
                print("\n-----------\nResuming training with checkpoint: {}\n-----------\n".format(resume_ckpt))

                # optim reload
                optim_ckpt = settings.resume + "_optims"
                if os.path.exists(optim_ckpt):
                    resumed_optimizer = torch.load(optim_ckpt, map_location=torch.device(settings.device))      # loaded model
                    train(env, model, settings.ckpt, training_params, reward_coeff, resumed_optimizer)
                else:
                    train(env, model, settings.ckpt, training_params, reward_coeff)

            else:
                raise ValueError("Please correctly type checkpoint path to resume training")

        else:
            #env.render()
            train(env, model, settings.ckpt, training_params, reward_coeff)
