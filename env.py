from typing import Callable, Optional, List, Dict, Any
from collections import namedtuple
import os
from isaacgym import gymapi, gymtorch
import torch
import utils
from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quatdiff_normalized, quat_inverse, calc_heading_quat
from poselib.core import quat_mul
from isaacgym import gymutil
from humanoid_view import HumanoidView, HumanoidViewTennis
from humanoid_extract import HumanoidExtract, ICCGANHumanoidExtractTarget
from env_proj import ICCGANHumanoidProjectile

def parse_kwarg(kwargs: dict, key: str, default_val: Any):
    return kwargs[key] if key in kwargs else default_val

class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[List[str]]=None, ob_horizon: Optional[int]=None, 
        parent_link: Optional[str]=None, local_pos: bool=False, local_height: bool=True,
        replay_speed: Optional[Callable]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.parent_link = parent_link
        self.local_pos = local_pos
        self.local_height = local_height
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight

DiscriminatorProperty = namedtuple("DiscriminatorProperty",
    "name key_links parent_link local_pos local_height replay_speed ob_horizon id"
)

class SensorInputConfig(object):
    def __init__(self,
              rlh_localPos: Optional[str]=None,  
              rlh_localRot: Optional[str]=None, 
              joystick: Optional[str]=None):
        
        self.rlh_localPos = rlh_localPos
        self.rlh_localRot = rlh_localRot
        self.joystick = joystick
        
SensorInputProperty = namedtuple("SensorInputProperty",
    "name rlh_localPos rlh_localRot joystick")

class Env(object):
    UP_AXIS = 2
    CHARACTER_MODEL = None
    
    def __init__(self,
        n_envs: int, fps: int, frameskip: int,
        episode_length: Optional[Callable or int] = 300,
        control_mode: str = "position",
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,
        **kwargs
    ):
        self.viewer = None
        assert(control_mode in ["position", "torque", "free"])
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1./self.fps
        self.control_mode = control_mode
        self.episode_length = episode_length
        self.device = torch.device(compute_device)
        if graphics_device is None:
            graphics_device = compute_device
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model

        sim_params = self.setup_sim_params()
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
        self.add_ground()
        self.envs, self.actors = self.create_envs(n_envs)   # _load_target_asset -> 
        self.setup_action_normalizer()
        self.create_tensors()

        self.gym.prepare_sim(self.sim)

        n_actors_per_env = self.gym.get_actor_count(self.envs[0])
        self.actor_ids = torch.arange(n_actors_per_env * len(self.envs), dtype=torch.int32, device=self.device).view(len(self.envs), -1)
        controllable_actors = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            dof = self.gym.get_actor_dof_count(self.envs[0], i)
            if dof > 0: controllable_actors.append(i)
        self.actor_ids_having_dofs = \
            n_actors_per_env * torch.arange(len(self.envs), dtype=torch.int32, device=self.device).unsqueeze(-1) + \
            torch.tensor(controllable_actors, dtype=torch.int32, device=self.device).unsqueeze(-2)
        
        self.refresh_tensors()
        self.viewer_pause = False
        self.camera_following = True
        self.viewer_advance = False
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        base_pos = self.root_pos[tar_env].cpu()
        self.cam_target = gymapi.Vec3(*self.vector_up(1.0, [base_pos[0], base_pos[1], base_pos[2]]))

        self.simulation_step = 0
        self.lifetime = torch.zeros(len(self.envs), dtype=torch.int64, device=self.device)
        self.done = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)
        self.info = dict(lifetime=self.lifetime)

        self.act_dim = self.action_scale.size(-1)
        self.ob_dim = self.observe().size(-1)
        self.rew_dim = self.reward().size(-1)

        for i in range(self.gym.get_actor_count(self.envs[0])):
            rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], i)
            print(rigid_body, len(rigid_body))
            dof = self.gym.get_actor_dof_dict(self.envs[0], i)
            print(dof, len(dof))

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)

    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[self.UP_AXIS] = val
        return base_vector
    
    def setup_sim_params(self):
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = True # force to enable GPU
        sim_params.dt = self.step_time/self.frameskip
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z if self.UP_AXIS == 2 else gymapi.UP_AXIS_Y
        sim_params.gravity = gymapi.Vec3(*self.vector_up(-9.81))
        sim_params.num_client_threads = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1
        sim_params.physx.num_subscenes = 4  # works only for CPU 
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 10.0
        sim_params.physx.default_buffer_size_multiplier = 5.0
        sim_params.physx.max_gpu_contact_pairs = 8*1024*1024
        # FIXME IsaacGym Pr4 will provide unreliable results when collecting from all substeps
        sim_params.physx.contact_collection = \
            gymapi.ContactCollection(gymapi.ContactCollection.CC_LAST_SUBSTEP) 
        #gymapi.ContactCollection(gymapi.ContactCollection.CC_ALL_SUBSTEPS)
        sim_params.physx.use_gpu = True
        return sim_params

    def add_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(*self.vector_up(1.0))
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def add_actor(self, env, i):
        pass

    def create_envs(self, n: int):
        if self.control_mode == "position":
            control_mode = gymapi.DOF_MODE_POS
        elif self.control_mode == "torque":
            control_mode = gymapi.DOF_MODE_EFFORT
        else:
            control_mode = gymapi.DOF_MODE_NONE

        envs, actors = [], []
        env_spacing = 3
        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        actor_asset = self.gym.load_asset(self.sim, os.path.abspath(os.path.dirname(self.character_model)), os.path.basename(self.character_model), asset_options)
        spacing_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        spacing_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        n_envs_per_row = int(n**0.5)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(0.89))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # humanoid
        self.num_bodies = self.gym.get_asset_rigid_body_count(actor_asset)
        self.num_dof = self.gym.get_asset_dof_count(actor_asset)
        self.num_joints = self.gym.get_asset_joint_count(actor_asset)

        for i in range(n):
            env = self.gym.create_env(self.sim, spacing_lower, spacing_upper, n_envs_per_row)
            actor = self.gym.create_actor(env, actor_asset, start_pose, "actor", i, -1, 0)
            self.add_actor(env, i)
            envs.append(env)
            actors.append(actor)
            # enable PD control
            # Kp (stiffness) and Kd (damping) are defined inside the mjcf xml file
            dof_prop = self.gym.get_asset_dof_properties(actor_asset)
            dof_prop["driveMode"].fill(control_mode)
            self.gym.set_actor_dof_properties(env, actor, dof_prop)

        self.num_envs = len(envs)
        return envs, actors

    def render(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(*self.vector_up(2.0, [base_pos[0], base_pos[1]-4.5, base_pos[2]-4.5]))
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "TOGGLE_CAMERA_FOLLOWING")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "TOGGLE_PAUSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "SINGLE_STEP_ADVANCE")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_0, "UpperBody")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "LowerBody")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_2, "FullBody")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_3, "Default")

    
    def update_viewer(self):
        self.gym.poll_viewer_events(self.viewer)
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "QUIT" and event.value > 0:
                exit()
            if event.action == "TOGGLE_CAMERA_FOLLOWING" and event.value > 0:
                self.camera_following = not self.camera_following
            if event.action == "TOGGLE_PAUSE" and event.value > 0:
                self.viewer_pause = not self.viewer_pause
            if event.action == "SINGLE_STEP_ADVANCE" and event.value > 0:
                self.viewer_advance = not self.viewer_advance

            if event.action == "UpperBody" and event.value > 0:
                self.upper, self.lower, self.full = True, False, False
            if event.action == "LowerBody" and event.value > 0:
                self.upper, self.lower, self.full = False, True, False
            if event.action == "FullBody" and event.value > 0:
                self.upper, self.lower, self.full = False, False, True
            if event.action == "Default" and event.value > 0:
                self.upper, self.lower, self.full = False, False, False

        if self.camera_following: self.update_camera()
        self.gym.step_graphics(self.sim)

    def update_camera(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, self.envs[tar_env])
        dx, dy = cam_trans.p.x - self.cam_target.x, cam_trans.p.y - self.cam_target.y
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(base_pos[0]+dx, base_pos[1]+dy, cam_trans.p.z)
        self.cam_target = gymapi.Vec3(base_pos[0], base_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)

    def refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def create_tensors(self):
        self._all_root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._all_root_tensor = gymtorch.wrap_tensor(self._all_root_tensor)
        #! humanoid root
        self.root_tensor = self._all_root_tensor.view(len(self.envs), -1, 13)[..., 0, :].unsqueeze(1)      # [num_envs, 13]
        
        num_actors = self.get_num_actors_per_env()
        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        num_links = self.gym.get_env_rigid_body_count(self.envs[0])
        self._all_link_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._all_link_tensor = gymtorch.wrap_tensor(self._all_link_tensor)
        #! humanoid link: self.num_bodies
        self.link_tensor = self._all_link_tensor.view(len(self.envs), num_links, -1)[..., :self.num_bodies, :]
        num_dof = self.gym.get_env_dof_count(self.envs[0])
        joint_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        joint_tensor = gymtorch.wrap_tensor(joint_tensor)
        # IsaacGym Preview 3 supports fix, revolute and prismatic (1d) joints only
        #! humanoid num_dof: self.num_dof
        self.joint_tensor = joint_tensor.view(len(self.envs), num_dof, -1)[..., :self.num_dof, :]  # n_envs x n_dof x 2 (pos, vel for each)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.contact_force_tensor = contact_force_tensor.view(len(self.envs), -1, 3)

    def get_num_actors_per_env(self):
        num_actors = self._all_root_tensor.shape[0] // self.num_envs
        return num_actors  

    def setup_action_normalizer(self):
        action_lower, action_upper = [], []
        action_scale = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            actor = self.gym.get_actor_handle(self.envs[0], i)
            n_dof = self.gym.get_actor_dof_count(self.envs[0], actor)
            if n_dof < 1: continue
            dof_prop = self.gym.get_actor_dof_properties(self.envs[0], actor)
            if self.control_mode == "position":
                action_lower.extend([min(dof_prop["lower"][j], dof_prop["upper"][j]) for j in range(n_dof)])
                action_upper.extend([max(dof_prop["lower"][j], dof_prop["upper"][j]) for j in range(n_dof)])
                action_scale.append(2)
            elif self.control_mode == "torque":
                action_lower.extend([-dof_prop["effort"][j] for j in range(n_dof)])
                action_upper.extend([dof_prop["effort"][j] for j in range(n_dof)])
                action_scale.append(1)
        action_offset = 0.5 * np.add(action_upper, action_lower)
        action_scale *= 0.5 * np.subtract(action_upper, action_lower)
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=self.device)
    def process_actions(self, actions):
        if type(actions) is tuple and len(actions) is 1:
            actions = actions[0]
            
        return actions*self.action_scale + self.action_offset

    def reset(self):
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime)
        self.request_quit = False
        self.obs = None

    def reset_done(self):
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)                                # 1. reset_envs를 통해 goal도 reset한다!
                if len(env_ids) == len(self.envs) or self.obs is None:
                    self.obs = self.observe()                           # 2. 이후 새로운 goal과 state를 가지고 다음 obs observe!
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info
    
    def reset_envs(self, env_ids):
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)
        self.root_tensor[env_ids] = ref_root_tensor
        self.link_tensor[env_ids] = ref_link_tensor
        self.joint_tensor[env_ids] = ref_joint_tensor

        #actor_ids = self.actor_ids[env_ids].flatten()
        actor_ids = self.actor_ids[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            actor_ids, n_actor_ids
        )
        actor_ids = self.actor_ids_having_dofs[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor),
            actor_ids, n_actor_ids
        )

        self.lifetime[env_ids] = 0

    def do_simulation(self):
        for _ in range(self.frameskip):
            self.gym.simulate(self.sim)
        self.simulation_step += 1

    def step(self, actions):
        if not self.viewer_pause or self.viewer_advance:
            self.apply_actions(actions)
            self.do_simulation()
            self.refresh_tensors()
            self.lifetime += 1
            if self.viewer is not None:
                self.gym.fetch_results(self.sim, True)
                self.viewer_advance = False

        if self.viewer is not None:
            self.update_viewer()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)    # sync to simulation dt

        rewards = self.reward()
        terminate = self.termination_check()                    # N
        if self.viewer_pause:
            overtime = None
        else:
            overtime = self.overtime_check()
        if torch.is_tensor(overtime):
            self.done = torch.logical_or(overtime, terminate)
        else:
            self.done = terminate
        self.info["terminate"] = terminate
        self.obs = self.observe()
        self.request_quit = False if self.viewer is None else self.gym.query_viewer_has_closed(self.viewer)
        return self.obs, rewards, self.done, self.info

    def apply_actions(self, actions):
        actions = self.process_actions(actions)
        actions = gymtorch.unwrap_tensor(actions)
        if self.control_mode == "position":
            self.gym.set_dof_position_target_tensor(self.sim, actions)
        elif self.control_mode == "torque":
            self.gym.set_dof_actuation_force_tensor(self.sim, actions)
        else:
            self.gym.set_dof_state_tensor(self.sim, actions)

    def init_state(self, env_ids):
        pass
    
    def observe(self, env_ids=None):
        pass
    
    def overtime_check(self):
        if self.episode_length:
            if callable(self.episode_length):
                return self.lifetime >= self.episode_length(self.simulation_step)
            return self.lifetime >= self.episode_length
        return None

    def termination_check(self):
        return torch.zeros(len(self.envs), dtype=torch.bool, device=self.device)

    def reward(self):
        return torch.ones((len(self.envs), 0), dtype=torch.float32, device=self.device)

    def set_char_color(self, col, env_ids, key_links=None):
        n_links = self.char_link_tensor.size(1) 
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.actors[env_id]

        if len(env_ids):
            if key_links is None:
                for j in range(n_links):
                    self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(col[0], col[1], col[2]))
            else:
                for j in key_links:
                    self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(col[0], col[1], col[2]))
        return


from ref_motion import ReferenceMotion
import numpy as np


class ICCGANHumanoid(Env):

    # CHARACTER_MODEL = "assets/humanoid.xml"
    CHARACTER_MODEL = "assets/new_cml.xml"
    CONTROLLABLE_LINKS = ["torso", "head", 
        "right_upper_arm", "right_lower_arm",
        "left_upper_arm", "left_lower_arm", 
        "right_thigh", "right_shin", "right_foot",
        "left_thigh", "left_shin", "left_foot"]
    DOFS =  [3, 3, 
            3, 1, 
            3, 1, 
            3, 1, 3, 
            3, 1, 3]
    DOF_OFFSET =  [ 0,  3,                          # "torso", "head"
                    6,  9, 10,                      # "right_upper_arm", "right_lower_arm", "right_hand"
                    10, 13, 14,                     # "left_upper_arm", "left_lower_arm", "left_hand"
                    14, 17, 18,                     # "right_thigh", "right_shin", "right_foot",
                    21, 24, 25, 28                  # "left_thigh", "left_shin", "left_foot"
                    ]                               
    CONTACTABLE_LINKS = ["right_foot", "left_foot"]
    UP_AXIS = 2

    GOAL_DIM = 0
    GOAL_REWARD_WEIGHT = None
    ENABLE_GOAL_TIMER = False
    GOAL_TENSOR_DIM = None

    OB_HORIZON = 4

    def __init__(self, *args,
        motion_file: str,
        discriminators: Dict[str, DiscriminatorConfig],
    **kwargs):
        contactable_links = parse_kwarg(kwargs, "contactable_links", self.CONTACTABLE_LINKS)
        controllable_links = parse_kwarg(kwargs, "controllable_links", self.CONTROLLABLE_LINKS)
        dofs = parse_kwarg(kwargs, "dofs", self.DOFS)
        goal_reward_weight = parse_kwarg(kwargs, "goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.enable_goal_timer = parse_kwarg(kwargs, "enable_goal_timer", self.ENABLE_GOAL_TIMER)
        self.goal_tensor_dim = parse_kwarg(kwargs, "goal_tensor_dim", self.GOAL_TENSOR_DIM)
        self.ob_horizon = parse_kwarg(kwargs, "ob_horizon", self.OB_HORIZON)

        super().__init__(*args, **kwargs)

        n_envs = len(self.envs)
        n_links = self.char_link_tensor.size(1)
        n_dofs = self.char_joint_tensor.size(1)
        
        controllable_links = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link)
            for link in controllable_links]

        if contactable_links:
            contact = np.zeros((n_envs, n_links), dtype=bool)
            for link in contactable_links:
                lid = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link)
                assert(lid >= 0), "Unrecognized contactable link {}".format(link)
                contact[:, lid] = True
            self.contactable_links = torch.tensor(contact).to(self.contact_force_tensor.device)
        else:
            self.contactable_links = None

        init_pose = motion_file
        self.ref_motion = ReferenceMotion(motion_file=init_pose, character_model=self.character_model,
            key_links=np.arange(n_links), controllable_links=controllable_links, dofs=dofs,
            device=self.device
        )

        ref_motion = {init_pose: self.ref_motion}
        disc_ref_motion = dict()
        for id, config in discriminators.items():
            m = init_pose if config.motion_file is None else config.motion_file
            if m not in ref_motion:
                ref_motion[m] = ReferenceMotion(motion_file=m, character_model=self.character_model,
                    key_links=np.arange(n_links), controllable_links=controllable_links, dofs=dofs,
                    device=self.device
                )
            key = (ref_motion[m], config.replay_speed)
            if config.ob_horizon is None:
                config.ob_horizon = self.ob_horizon+1
            if key not in disc_ref_motion: disc_ref_motion[key] = [0, []]
            disc_ref_motion[key][0] = max(disc_ref_motion[key][0], config.ob_horizon)
            disc_ref_motion[key][1].append(id)

        if goal_reward_weight is not None:
            reward_weights = torch.empty((len(self.envs), self.rew_dim), dtype=torch.float32, device=self.device)
            if not hasattr(goal_reward_weight, "__len__"):
                goal_reward_weight = [goal_reward_weight]
            assert(self.rew_dim == len(goal_reward_weight))
            for i, w in zip(range(self.rew_dim), goal_reward_weight):
                reward_weights[:, i] = w
        elif self.rew_dim:
            goal_reward_weight = []
            assert(self.rew_dim == len(goal_reward_weight))

        n_comp = len(discriminators) + self.rew_dim
        if n_comp > 1:
            self.reward_weights = torch.zeros((n_envs, n_comp), dtype=torch.float32, device=self.device)
            weights = [disc.weight for _, disc in discriminators.items() if disc.weight is not None]
            total_weights = sum(weights) if weights else 0
            assert(total_weights <= 1), "Discriminator weights must not be greater than 1."
            n_unassigned = len(discriminators) - len(weights)
            rem = 1 - total_weights
            for disc in discriminators.values():
                if disc.weight is None:
                    disc.weight = rem / n_unassigned
                elif n_unassigned == 0:
                    disc.weight /= total_weights
        else:
            self.reward_weights = None

        self.discriminators = dict()
        max_ob_horizon = self.ob_horizon+1
        for i, (id, config) in enumerate(discriminators.items()):
            key_links = None if config.key_links is None else sorted([
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], link) for link in config.key_links
            ])
            parent_link = None if config.parent_link is None else \
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], config.parent_link)

            assert(key_links is None or all(lid >= 0 for lid in key_links))
            assert(parent_link is None or parent_link >= 0)
            
            self.discriminators[id] = DiscriminatorProperty(
                name = id,
                key_links = key_links,
                parent_link = parent_link,
                local_pos = config.local_pos,
                local_height = config.local_height,
                replay_speed = config.replay_speed,
                ob_horizon = config.ob_horizon,
                id=i
            )
            if self.reward_weights is not None:
                self.reward_weights[:, i] = config.weight
            max_ob_horizon = max(max_ob_horizon, config.ob_horizon)

        if max_ob_horizon != self.state_hist.size(0):
            self.state_hist = torch.empty((max_ob_horizon, *self.state_hist.shape[1:]),
                dtype=self.root_tensor.dtype, device=self.device)
        self.disc_ref_motion = [
            (ref_motion, replay_speed, max_ob_horizon, [self.discriminators[id] for id in disc_ids])
            for (ref_motion, replay_speed), (max_ob_horizon, disc_ids) in disc_ref_motion.items()
        ]

        if self.rew_dim > 0:
            if self.rew_dim > 1:
                self.reward_weights *= (1-reward_weights.sum(dim=-1, keepdim=True))
            else:
                self.reward_weights *= (1-reward_weights)
            self.reward_weights[:, -self.rew_dim:] = reward_weights
            
        self.info["ob_seq_lens"] = torch.zeros_like(self.lifetime)  # dummy result
        self.info["disc_obs"] = self.observe_disc(self.state_hist)  # dummy result
        self.info["disc_obs_expert"] = self.info["disc_obs"]        # dummy result
        self.goal_dim = self.GOAL_DIM
        self.state_dim = (self.ob_dim-self.goal_dim)//self.ob_horizon
        self.disc_dim = {
            name: ob.size(-1)
            for name, ob in self.info["disc_obs"].items()
        }
    
    def reset_done(self):
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        info["reward_weights"] = self.reward_weights
        return obs, info
    
    def reset(self):
        if self.goal_tensor is not None:
            self.goal_tensor.zero_()
            if self.goal_timer is not None: self.goal_timer.zero_()
        super().reset()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_goal(env_ids)
        
    def reset_goal(self, env_ids):
        pass
    
    def step(self, actions):
        self.state_hist[:-1] = self.state_hist[1:].clone()
        #### added
        if self.goal_embedding:
            self.goal_hist[:-1] = self.goal_hist[1:].clone()

        obs, rews, dones, info = super().step(actions)
        info["disc_obs"] = self.observe_disc(self.state_hist)
        info["disc_obs_expert"], info["disc_seq_len"] = self.fetch_real_samples()
        return obs, rews, dones, info

    def overtime_check(self):
        if self.goal_timer is not None:
            self.goal_timer -= 1
            env_ids = torch.nonzero(self.goal_timer <= 0).view(-1)
            if len(env_ids) > 0: self.reset_goal(env_ids)
        return super().overtime_check()

    def termination_check(self):
        if self.contactable_links is None:
            return torch.zeros_like(self.done)
        masked_contact = self.char_contact_force_tensor.clone()
        masked_contact[self.contactable_links] = 0          # N x n_links x 3

        contacted = torch.any(masked_contact > 1., dim=-1)  # N x n_links
        too_low = self.link_pos[..., self.UP_AXIS] < 0.15    # N x n_links

        terminate = torch.any(torch.logical_and(contacted, too_low), -1)    # N x
        terminate *= (self.lifetime > 1)
        return terminate

    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        return self.ref_motion.state(motion_ids, motion_times)
    
    def create_tensors(self):
        super().create_tensors()
        n_dofs = self.gym.get_actor_dof_count(self.envs[0], 0)
        n_links = self.gym.get_actor_rigid_body_count(self.envs[0], 0)
        self.root_pos, self.root_orient = self.root_tensor[:, 0, :3], self.root_tensor[:, 0, 3:7]
        self.root_lin_vel, self.root_ang_vel = self.root_tensor[:, 0, 7:10], self.root_tensor[:, 0, 10:13]
        self.char_root_tensor = self.root_tensor[:, 0]
        if self.link_tensor.size(1) > n_links:
            self.link_pos, self.link_orient = self.link_tensor[:, :n_links, :3], self.link_tensor[:, :n_links, 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[:, :n_links, 7:10], self.link_tensor[:, :n_links, 10:13]
            self.char_link_tensor = self.link_tensor[:, :n_links]
        else:
            self.link_pos, self.link_orient = self.link_tensor[..., :3], self.link_tensor[..., 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[..., 7:10], self.link_tensor[..., 10:13]
            self.char_link_tensor = self.link_tensor
        if self.joint_tensor.size(1) > n_dofs:
            self.joint_pos, self.joint_vel = self.joint_tensor[:, :n_dofs, 0], self.joint_tensor[:, :n_dofs, 1]
            self.char_joint_tensor = self.joint_tensor[:, :n_dofs]
        else:
            self.joint_pos, self.joint_vel = self.joint_tensor[..., 0], self.joint_tensor[..., 1]
            self.char_joint_tensor = self.joint_tensor
        
        self.char_contact_force_tensor = self.contact_force_tensor[:, :n_links]

        ob_disc_dim = 13 + n_links*13
        self.state_hist = torch.empty((self.ob_horizon+1, len(self.envs), ob_disc_dim),
            dtype=self.root_tensor.dtype, device=self.device)                           # [5, NUM_ENVS, 13 + n_links*13]

        if self.goal_tensor_dim:
            try:
                self.goal_tensor = [
                    torch.zeros((len(self.envs), dim), dtype=self.root_tensor.dtype, device=self.device)
                    for dim in self.goal_tensor_dim
                ]
                ## added
                if self.goal_embedding:
                    self.goal_hist = [
                        torch.zeros((self.ob_horizon+1, len(self.envs), dim), dtype=self.root_tensor.dtype, device=self.device)
                        for dim in self.goal_tensor_dim
                    ]
                    self.goal_rootpos_hist = [
                        torch.zeros((self.ob_horizon+1, len(self.envs), 3), dtype=self.root_tensor.dtype, device=self.device)
                    ]
                    self.goal_rootorient_hist = [
                        torch.zeros((self.ob_horizon+1, len(self.envs), 4), dtype=self.root_tensor.dtype, device=self.device)
                    ]                    
            except TypeError:
                self.goal_tensor = torch.zeros((len(self.envs), self.goal_tensor_dim), dtype=self.root_tensor.dtype, device=self.device)
                ## added
                if self.goal_embedding:
                    self.goal_hist = torch.zeros((self.ob_horizon+1, len(self.envs), self.goal_tensor_dim), dtype=self.root_tensor.dtype, device=self.device)
                    self.goal_rootpos_hist =  torch.zeros((self.ob_horizon+1, len(self.envs), 3), dtype=self.root_tensor.dtype, device=self.device)
                    self.goal_rootorient_hist =  torch.zeros((self.ob_horizon+1, len(self.envs), 4), dtype=self.root_tensor.dtype, device=self.device)
        else:
            self.goal_tensor = None
        self.goal_timer = torch.zeros((len(self.envs), ), dtype=torch.int32, device=self.device) if self.enable_goal_timer else None

    def observe(self, env_ids=None):
        self.ob_seq_lens = self.lifetime+1 #(self.lifetime+1).clip(max=self.state_hist.size(0)-1)
        n_envs = len(self.envs)
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[-1] = torch.cat((
                self.char_root_tensor, self.char_link_tensor.view(n_envs, -1)
            ), -1)
            #### added
            if self.goal_embedding:
                self.goal_hist[-1] = self.goal_tensor.clone()
                self.goal_rootpos_hist[-1] = self.goal_root_tensor[:, 0, :3].clone()
                self.goal_rootorient_hist[-1] = self.goal_root_tensor[:, 0, 3:7].clone()
            #### added

            env_ids = None
        else:
            n_envs = len(env_ids)
            self.state_hist[-1, env_ids] = torch.cat((
                self.char_root_tensor[env_ids], self.char_link_tensor[env_ids].view(n_envs, -1)
            ), -1)
            #### added
            if self.goal_embedding:
                self.goal_hist[-1, env_ids] = self.goal_tensor[env_ids].clone()
                self.goal_rootpos_hist[-1, env_ids] = self.goal_root_tensor[env_ids, 0, :3].clone()
                self.goal_rootorient_hist[-1, env_ids] = self.goal_root_tensor[env_ids, 0, 3:7].clone()
            #### added

        return self._observe(env_ids)
    
    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens
            )
        else:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids]
            )
    
    def observe_disc(self, state):
        seq_len = self.info["ob_seq_lens"]+1
        res = dict()
        if torch.is_tensor(state):
            # fake
            for id, disc in self.discriminators.items():
                # res[id].shape = [512, 3, 56]
                res[id] = observe_disc(state[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link, disc.local_pos, disc.local_height)
            return res
        else:       # state is fed with dictionary
            # real  
            seq_len_ = dict()
            for disc_name, s in state.items():
                disc = self.discriminators[disc_name]
                res[disc_name] = observe_disc(s[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link, disc.local_pos, disc.local_height)
                seq_len_[disc_name] = seq_len
            return res, seq_len_

    def fetch_real_samples(self):
        n_inst = len(self.envs)

        samples = dict()
        for ref_motion, replay_speed, ob_horizon, discs in self.disc_ref_motion:
            dt = self.step_time
            if replay_speed is not None:
                dt /= replay_speed(n_inst)
            motion_ids, motion_times0 = ref_motion.sample(n_inst, truncate_time=dt*(ob_horizon-1))
            motion_ids = np.tile(motion_ids, ob_horizon)
            motion_times = np.concatenate((motion_times0, *[motion_times0+dt*i for i in range(1, ob_horizon)]))
            root_tensor, link_tensor, joint_tensor = ref_motion.state(motion_ids, motion_times)
            real = torch.cat((
                root_tensor, link_tensor.view(root_tensor.size(0), -1)
            ), -1).view(ob_horizon, n_inst, -1)

            for d in discs: samples[d.name] = real
        return self.observe_disc(samples)

    # def update_viewer(self):
    #     super().update_viewer()
    #     self.gym.clear_lines(self.viewer)
    #     self.visualize_axis(self.link_pos[:, [0, 6, 7, 8], :], self.link_orient[:, [0, 6, 7, 8], :], scale = 0.2, y=False, z =False)

    def visualize_axis(self, gpos, gquat, scale, x=True, y=True, z = True):
        gquat = gquat.view(-1, 4).cpu()                                                 # [num_envs x n_links, 4]
        tan_norm = utils.quat_to_tan_norm(gquat).cpu()
        rot_mat = utils.tan_norm_to_rotmat(tan_norm).cpu()
        tan, binorm, norm = rot_mat[..., 0:3], rot_mat[..., 3:6], rot_mat[..., 6:]       # [num_envs x n_links, 3]
        tan, binorm, norm = tan.view(len(self.envs), -1, 3), norm.view(len(self.envs), -1, 3), binorm.view(len(self.envs), -1, 3)   # [num_envs, n_links, 3]
        
        start = gpos.cpu().numpy()                                                      # [5,n_links,3]

        x_end = (gpos.cpu() + tan * scale).cpu().numpy()
        z_end = (gpos.cpu() + binorm * scale).cpu().numpy()
        y_end = (gpos.cpu() + norm * scale).cpu().numpy()
        
        n_lines = 5

        # x-axis
        if x:
            for j in range(gpos.size(1)):
                x_lines = np.stack([
                    np.stack((start[:, j, 0], start[:, j, 1], start[:, j, 2]+0.0015*i, x_end[:, j, 0], x_end[:, j, 1], x_end[:, j, 2]+0.0015*i), -1)
                            for i in range(n_lines)], -2)                                      # [n_envs, n_lines, 6]
                for e, l in zip(self.envs, x_lines):
                    self.gym.add_lines(self.viewer, e, n_lines, l, [[1., 0., 0.] for _ in range(n_lines)])
        # y-axis
        if y:
            for j in range(gpos.size(1)):
                ylines = np.stack([
                    np.stack((start[:, j, 0]+0.0015*i, start[:, j, 1], start[:, j, 2], y_end[:, j, 0]+0.0015*i, y_end[:, j, 1], y_end[:, j, 2]), -1)
                            for i in range(n_lines)], -2)                                      # [n_envs, n_lines, 6]
                for e, l in zip(self.envs, ylines):
                    self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 1., 0.] for _ in range(n_lines)])
        # z-axis
        if z:
            for j in range(gpos.size(1)):
                z_lines = np.stack([
                    np.stack((start[:, j, 0], start[:, j, 1]+0.0015*i, start[:, j, 2], z_end[:, j, 0], z_end[:, j, 1]+0.0015*i, z_end[:, j, 2]), -1)
                            for i in range(n_lines)], -2)                                      # [n_envs, n_lines, 6]
                for e, l in zip(self.envs, z_lines):
                    self.gym.add_lines(self.viewer, e, n_lines, l, [[0., 0., 1.] for _ in range(n_lines)])
        pass
    
    def get_link_len(self, p_idx, c_idx):
        p_pos, c_pos = self.link_pos[0, p_idx, :], self.link_pos[0, c_idx, :] # [n_links, 3]
        link_len = torch.linalg.norm((p_pos - c_pos), ord=2, dim=-1, keepdim=True)  # [n_links, 1]
        return link_len 
        # what I added
    def create_motion_info(self):
        pass
        # what I added
@torch.jit.script
def observe_iccgan(state_hist: torch.Tensor, seq_len: torch.Tensor):
    # state_hist: L x N x D

    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    root_tensor = state_hist[..., :13]
    link_tensor = state_hist[...,13:].view(state_hist.size(0), state_hist.size(1), -1, 13)  # [L X N X n_link, 13]

    root_pos, root_orient = root_tensor[..., :3], root_tensor[..., 3:7]
    link_pos, link_orient = link_tensor[..., :3], link_tensor[..., 3:7]
    link_lin_vel, link_ang_vel = link_tensor[..., 7:10], link_tensor[..., 10:13]

    origin = root_pos[-1].clone()
    origin[..., UP_AXIS] = 0                                            # N x 3
    heading = heading_zup(root_orient[-1])
    up_dir = torch.zeros_like(origin)
    up_dir[..., UP_AXIS] = 1
    heading_orient_inv = axang2quat(up_dir, -heading)                   # N x 4

    heading_orient_inv = (heading_orient_inv                            # L x N x n_links x 4
        .view(1, -1, 1, 4).repeat(n_hist, 1, link_pos.size(-2), 1))
    origin = origin.unsqueeze_(-2)                                      # N x 1 x 3

    ob_link_pos = link_pos - origin                                     # L x N x n_links x 3 
    ob_link_pos = rotatepoint(heading_orient_inv, ob_link_pos)
    ob_link_orient = quatmultiply(heading_orient_inv, link_orient)      # L x N x n_links x 4
    ob_link_lin_vel = rotatepoint(heading_orient_inv, link_lin_vel)     # N x n_links x 3
    ob_link_ang_vel = rotatepoint(heading_orient_inv, link_ang_vel)     # N x n_links x 3

    ob = torch.cat((ob_link_pos, ob_link_orient,
        ob_link_lin_vel, ob_link_ang_vel), -1)                          # L x N x n_links x 13
    ob = ob.view(n_hist, n_inst, -1)                                    # L x N x (n_links x 13)

    ob1 = ob.permute(1, 0, 2)                                           # N x L x (n_links x 13)
    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    return ob2.flatten(start_dim=1)


@torch.jit.script
def observe_disc(state_hist: torch.Tensor, seq_len: torch.Tensor, key_links: Optional[List[int]]=None,
    parent_link: Optional[int]=None, local_pos: bool=False, local_height: bool = True,
):
    # state_hist: L x N x D

    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    root_tensor = state_hist[..., :13]
    link_tensor = state_hist[...,13:].view(n_hist, n_inst, -1, 13)
    if key_links is None:
        link_pos, link_orient = link_tensor[...,:3], link_tensor[...,3:7]
    else:   # DISCRIMINATOR마다 다른 KEY_LINKS를 가짐
        link_pos, link_orient = link_tensor[:,:,key_links,:3], link_tensor[:,:,key_links,3:7]

    if parent_link is None:
        origin = root_tensor[-1,:, :3].clone()               # N x 3
        origin[..., UP_AXIS] = 0
        orient = root_tensor[-1,:,3:7]                    # N x 4
    else:
        origin = link_tensor[:,:, parent_link, :3]      # L x N x 3
        orient = link_tensor[:,:, parent_link,3:7]      # L x N x 4
        if not local_height:
            origin = origin.clone()
            origin[..., UP_AXIS] = 0

    if local_pos:
        orient_inv = quatconj(orient)               # N x 4 or L x N x 4
    else:
        heading = heading_zup(orient)
        up_dir = torch.zeros_like(origin)
        up_dir[..., UP_AXIS] = 1
        orient_inv = axang2quat(up_dir, -heading)
    origin.unsqueeze_(-2)                           # N x 1 x 3 or L x N x 1 x 3

    if parent_link is None:
        orient_inv = orient_inv.view(1, -1, 1, 4)  # 1 x N x 1 x 4
    else:
        orient_inv = orient_inv.unsqueeze_(-2)     # L x N x 1 x 4

    ob_link_pos = link_pos - origin                                     # L x N x n_links x 3 
    ob_link_pos = rotatepoint(orient_inv, ob_link_pos)
    ob_link_orient = quatmultiply(orient_inv, link_orient)              # L x N x n_links x 4

    ob = torch.cat((ob_link_pos, ob_link_orient), -1)                   # L x N x n_links x 7
    ob = ob.view(n_hist, n_inst, -1)                                    # L x N x (n_links x 7)

    ob1 = ob.permute(1, 0, 2)                                           # N x L x (n_links x 7)
    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    return ob2              # ob2.shape:  For each Discriminator: [N_ENVS, n_horizons, n_KEY_links * 7] 

