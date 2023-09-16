import time
import gym
import numpy as np
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.typing import (
    EnvActionType,
    EnvConfigDict,
    EnvInfoDict,
    EnvObsType,
    EnvType,
    PartialTrainerConfigDict,
)
from typing import Callable, List, Optional, Tuple

from gym.utils import seeding
import torch
import math
import pygame
from gym.envs.registration import registry
from scipy.spatial.transform import Rotation as R

X = 0
Y = 1

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)

STATE_INITIAL = 0  # moving towards the passage
STATE_PASSAGE = 1  # inside the passage
STATE_AFTER = 2  # moving towards the goal
STATE_REACHED_GOAL = 3  # goal reached
STATE_FINISHED = 4  # goal reached and reward bonus given

if "MycustomEvn-v0" in registry.env_specs:
    del registry.env_specs["MycustomEvn-v0"]

# Register the environment
gym.envs.register(
    id="MycustomEvn-v0",
    entry_point='env:PassageEnv',  # assuming 'env' is the correct module name
)

class PassageEnv(VectorEnv):
    def __init__(self, config):
        self.cfg = config
        action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low=-self.cfg["max_v"],
                    high=self.cfg["max_v"],
                    shape=(2,),
                    dtype=float,
                ),
            )
            * self.cfg["n_agents"]
        )

        # Limit the start position and goal position between [-6,6]
        # 580 * 890 for lab room
        # Change the "pos" and "goal" from "-6.0, 6.0 to -8.0, 8.0"
        observation_space = gym.spaces.Dict(
            {
                "pos": gym.spaces.Box(
                    -6.0, 6.0, shape=(self.cfg["n_agents"], 2), dtype=float
                ),
                "vel": gym.spaces.Box(
                    -100000.0, 100000.0, shape=(self.cfg["n_agents"], 2), dtype=float
                ),
                "goal": gym.spaces.Box(
                    -6.0, 6.0, shape=(self.cfg["n_agents"], 2), dtype=float
                ),
                "time": gym.spaces.Box(
                    0,
                    self.cfg["max_time_steps"] * self.cfg["dt"],
                    shape=(self.cfg["n_agents"], 1),
                    dtype=float,
                ),
            }
        )

        super().__init__(observation_space, action_space, self.cfg["num_envs"])

        self.device = torch.device(self.cfg["device"])
        self.vec_p_shape = (self.cfg["num_envs"], self.cfg["n_agents"], 2)

        self.vector_reset()

        # Add wall and obstacles here
        self.obstacles = [
            # down wall
            {"min": [-1.6,-2],"max": [-1.2,-1.5],},
            {"min": [-1.5,-1.5],"max": [-1.1,-1],},
            {"min": [-1.4,-1],"max": [-1.0,-0.5],},
            {"min": [-1.3,-0.5],"max": [-0.9,0],},
            {"min": [-1.2,0],"max": [-0.8,0.5],},
            {"min": [-1.1,0.5],"max": [-0.7,1],},
            {"min": [-1.0,1],"max": [-0.6,1.5],},
            {"min": [-0.9,1.5],"max": [-0.5,2],},
            {"min": [-0.8,2],"max": [-0.4,2.5],},
            # up wall
            {"min": [1.2,-2],"max": [1.6,-1.5],},
            {"min": [1.1,-1.5],"max": [1.5,-1],},
            {"min": [1.0,-1],"max": [1.4,-0.5],},
            {"min": [0.9,-0.5],"max": [1.3,0],},
            {"min": [0.8,0],"max": [1.2,0.5],},
            {"min": [0.7,0.5],"max": [1.1,1],},
            {"min": [0.6,1],"max": [1.0,1.5],},
            {"min": [0.5,1.5],"max": [0.9,2],},
            {"min": [0.4,2],"max": [0.8,2.5],},
        ]

        # Initialize all imported pygame modules
        pygame.init()
        size = (
            (torch.Tensor(self.cfg["world_dim"]) * self.cfg["render_px_per_m"])
            .type(torch.int)
            .tolist()
        )
        self.display = pygame.display.set_mode(size)

    def create_state_tensor(self):
        return torch.zeros(self.vec_p_shape, dtype=torch.float32).to(self.device)

    def sample_pos_noise(self):
        if self.cfg["pos_noise_std"] > 0.0:
            return torch.normal(0.0, self.cfg["pos_noise_std"], self.vec_p_shape).to(
                self.device
            )
        else:
            return self.create_state_tensor()

    def compute_agent_dists(self, ps):
        agents_ds = torch.cdist(ps, ps)
        num_agents = ps.shape[1]
        diags = (
            torch.eye(num_agents).unsqueeze(0).repeat(len(ps), 1, 1).bool()
        )
        agents_ds[diags] = float("inf")
        return agents_ds

    def compute_obstacle_dists(self, ps):
        return torch.stack(
            [
                # https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
                torch.linalg.norm(
                    torch.stack(
                        [
                            torch.max(
                                torch.stack(
                                    [
                                        torch.zeros(len(ps), self.cfg["n_agents"]),
                                        o["min"][d] - ps[:, :, d],
                                        ps[:, :, d] - o["max"][d],
                                    ],
                                    dim=2,
                                ),
                                dim=2,
                            )[0]
                            for d in [X, Y]
                        ],
                        dim=2,
                    ),
                    dim=2,
                )
                for o in self.obstacles
            ],
            dim=2,
        )

    def rand(self, size, a: float, b: float):
        return (a - b) * torch.rand(size).to(self.device) + b

    def get_starts_and_goals(self, n):

        # No different formation each time

        def rand_n_agents(a, b):
            return self.rand(n, a, b).unsqueeze(1).repeat(1, self.cfg["n_agents"])

        box = (
            torch.Tensor(self.cfg["world_dim"]) / 2.0
            - self.cfg["placement_keepout_border"]
        )
        formation = torch.Tensor(self.cfg["agent_formation"]).to(self.device)

        starts = formation.repeat(n, 1, 1)
        starts[:, :, X] += rand_n_agents(-box[X], box[X])
        starts[:, :, Y] += rand_n_agents(-3.6, -3.5)

        goals = formation.repeat(n, 1, 1)
        goals[:, :, X] += rand_n_agents(-box[X], box[X])
        goals[:, :, Y] += rand_n_agents(4, box[Y])
        
        return starts, goals


    def vector_reset(self) -> List[EnvObsType]:
        """Resets all sub-environments.
        Returns:
            obs (List[any]): List of observations from each environment.
        """
        starts, goals = self.get_starts_and_goals(self.cfg["num_envs"])
        # positions
        self.ps = starts
        # goal positions
        self.goal_ps = goals
        # measured velocities
        self.measured_vs = self.create_state_tensor()
        # current state to determine next waypoint for reward
        self.states = torch.zeros(self.cfg["num_envs"], self.cfg["n_agents"]).to(
            self.device
        )
        # save goal vectors only for visualization
        self.rew_vecs = torch.zeros(self.cfg["num_envs"], self.cfg["n_agents"], 2).to(
            self.device
        )
        self.timesteps = torch.zeros(self.cfg["num_envs"], dtype=torch.int).to(
            self.device
        )
        return [self.get_obs(index) for index in range(self.cfg["num_envs"])]

    def reset_at(self, index: Optional[int] = None) -> EnvObsType:
        """Resets a single environment.
        Args:
            index (Optional[int]): An optional sub-env index to reset.
        Returns:
            obs (obj): Observations from the reset sub environment.
        """
        start, goal = self.get_starts_and_goals(1)
        self.ps[index] = start[0]
        self.goal_ps[index] = goal[0]
        self.measured_vs[index] = torch.zeros(self.cfg["n_agents"], 2)
        self.states[index] = torch.zeros(self.cfg["n_agents"])
        self.rew_vecs[index] = torch.zeros(self.cfg["n_agents"], 2)
        self.timesteps[index] = 0
        return self.get_obs(index)

    def get_obs(self, index: int) -> EnvObsType:
        return {
            "pos": self.ps[index].tolist(),
            "vel": self.measured_vs[index].tolist(),
            "goal": self.goal_ps[index].tolist(),
            "time": [[(self.timesteps[index] * self.cfg["dt"]).tolist()]]
            * self.cfg["n_agents"],
        }

    def barrier_lyapunov_function(actual_distance, desired_distance, lower_bound = 0.1, upper_bound = 0.2):

        distance_error = np.abs(actual_distance - desired_distance)
        quadratic_cost = distance_error ** 2

        if distance_error <= lower_bound:
            barrier_cost = -np.log(lower_bound - distance_error)

        elif distance_error >= upper_bound:
            barrier_cost = -np.log(distance_error - upper_bound)

        else:
            barrier_cost = 0
        blf_value = quadratic_cost + barrier_cost

        return blf_value
    
    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        """Performs a vectorized step on all sub environments using `actions`.
        Args:
            actions (List[any]): List of actions (one for each sub-env).
        Returns:
            obs (List[any]): New observations for each sub-env.
            rewards (List[any]): Reward values for each sub-env.
            dones (List[any]): Done values for each sub-env.
            infos (List[any]): Info values for each sub-env.
        """
        self.timesteps += 1

        assert len(actions) == self.cfg["num_envs"]
        # Step the agents while considering vel and acc constraints
        desired_vs = torch.clip(
            torch.Tensor(actions).to(self.device), -self.cfg["max_v"], self.cfg["max_v"]
        )

        desired_as = (desired_vs - self.measured_vs) / self.cfg["dt"]
        possible_as = torch.clip(desired_as, self.cfg["min_a"], self.cfg["max_a"])
        possible_vs = self.measured_vs + possible_as * self.cfg["dt"]

        previous_ps = self.ps.clone().to(self.device)

        # check if next position collisides with other agents or wall
        # have to update agent step by step to be able to attribute negative rewards to each agent
        rewards = torch.zeros(self.cfg["num_envs"], self.cfg["n_agents"])
        next_ps = self.ps.clone()
        for i in range(self.cfg["n_agents"]):
            next_ps_agent = next_ps.clone()
            next_ps_agent[:, i] += possible_vs[:, i] * self.cfg["dt"]
            # print("################position: ", i, next_ps_agent[:,i])

            agents_ds = self.compute_agent_dists(next_ps_agent)[:, i]
            agents_coll = torch.min(agents_ds, dim=1)[0] <= 2 * self.cfg["agent_radius"]
            # only update pos if there are no collisions
            next_ps[~agents_coll, i] = next_ps_agent[~agents_coll, i]
            # penalty when colliding
            rewards[agents_coll, i] -= 1.5


        #########################################################################
        # Get the average position to get the mid point of the formation
        x_coordinates_sum = torch.sum(next_ps_agent[:,:,0], dim = 1)
        y_coordinates_sum = torch.sum(next_ps_agent[:,:,1], dim = 1)
        x_average = x_coordinates_sum / 5
        y_average = y_coordinates_sum / 5
        average_position = torch.stack((x_average, y_average), dim= 1)
        
        # Reshape average_positions from [32 x 2] to [32 x 1 x 2]
        expanded_avg_positions = average_position.unsqueeze(1)

        # Concatenate the expanded average positions with next_ps_agent along the agents' dimension
        next_ps_agent_with_avg = torch.cat((next_ps_agent, expanded_avg_positions), dim=1)
        #########################################################################
        # Formation control
        agent_distance = self.compute_agent_dists(next_ps_agent_with_avg)

        #########################################################################
        # Circle Formation
        # Make sure robot 0 is always be the leader
        robot_0_position_y = next_ps_agent[:,0,1]
        robot_1_position_y = next_ps_agent[:,1,1]
        robot_2_position_y = next_ps_agent[:,2,1]
        robot_3_position_y = next_ps_agent[:,3,1]
        robot_4_position_y = next_ps_agent[:,4,1]

        # Element-wise comparisons
        mask_4_0 = robot_4_position_y >= robot_0_position_y
        mask_1_0 = robot_1_position_y >= robot_0_position_y
        mask_2_1 = robot_2_position_y >= robot_1_position_y
        mask_3_4 = robot_3_position_y >= robot_4_position_y

        # Update the rewards tensor based on the masks
        rewards[mask_4_0, 4] -= 1
        rewards[mask_1_0, 1] -= 1
        rewards[mask_2_1, 2] -= 1
        rewards[mask_3_4, 3] -= 1

        #########################################################################
        # Calculate the center distance(radius)
        expected_radius_length = 0.7
        r4 = agent_distance[:, 4, 5]
        r3 = agent_distance[:, 3, 5]
        r2 = agent_distance[:, 2, 5]
        r1 = agent_distance[:, 1, 5]
        r0 = agent_distance[:, 0, 5]

        rewards[:, 4] -= torch.abs(r4 - expected_radius_length)
        rewards[:, 3] -= torch.abs(r3 - expected_radius_length)
        rewards[:, 2] -= torch.abs(r2 - expected_radius_length)
        rewards[:, 1] -= torch.abs(r1 - expected_radius_length)
        rewards[:, 0] -= torch.abs(r0 - expected_radius_length)
        # Calculate the side distance by using radius
        # Calculate average radius for each environment
        average_radius = (r0 + r1 + r2 + r3 + r4) / 5

        # Calculate expected side length for each environment
        # Side = 2 * radius * sin(pi/5)
        expected_side_length = 2 * average_radius * torch.sin(torch.tensor(np.pi) / 5)

        # Compute side lengths using agent distances
        s04 = agent_distance[:, 0, 4]
        s43 = agent_distance[:, 4, 3]
        s32 = agent_distance[:, 3, 2]
        s21 = agent_distance[:, 2, 1]
        s10 = agent_distance[:, 1, 0]

        # Calculate rewards based on deviation from expected side length
        rewards[:, 4] -= torch.abs(s04 - expected_side_length)
        rewards[:, 3] -= torch.abs(s43 - expected_side_length)
        rewards[:, 2] -= torch.abs(s32 - expected_side_length)
        rewards[:, 1] -= torch.abs(s21 - expected_side_length)
        rewards[:, 0] -= torch.abs(s10 - expected_side_length)
        ########################################################################
        # Maximize the scale of formation
        agent1_cur_state = self.states[0, 1].item()
        agent4_cur_state = self.states[0, 4].item()
        obstacle_ds = self.compute_obstacle_dists(next_ps)

        if agent1_cur_state == 1.0 or agent4_cur_state == 1.0:
    
            agent1_min_obstacle_distance = torch.min(obstacle_ds[:, 1, :], dim=1)[0]
            agent4_min_obstacle_distance = torch.min(obstacle_ds[:, 4, :], dim=1)[0]
            desired_ds = 0.1 + self.cfg["agent_radius"]
            agent1_distance_penalty = torch.abs(agent1_min_obstacle_distance - desired_ds)
            agent4_distance_penalty = torch.abs(agent4_min_obstacle_distance - desired_ds)

            # Control the scale formation
            rewards[:, 4] -= 2 * agent4_distance_penalty
            rewards[:, 3] -= 2 * agent4_distance_penalty
            rewards[:, 2] -= 2 * agent1_distance_penalty
            rewards[:, 1] -= 2 * agent1_distance_penalty
        #######################################################################

        # Prevent collisions between robots and walls
        obstacles_coll = torch.min(obstacle_ds, dim=2)[0] <= self.cfg["agent_radius"]
        rewards[obstacles_coll] -= 0.5
        self.ps[~obstacles_coll] = next_ps[~obstacles_coll]

        self.ps += self.sample_pos_noise()
        dim = torch.Tensor(self.cfg["world_dim"]) / 2
        self.ps[:, :, X] = torch.clip(self.ps[:, :, X], -dim[X], dim[X])
        self.ps[:, :, Y] = torch.clip(self.ps[:, :, Y], -dim[Y], dim[Y])

        self.measured_vs = (self.ps - previous_ps) / self.cfg["dt"]

        #######################################################################
        # update passage states
        wall_robot_offset = self.cfg["wall_width"] / 2 + self.cfg["agent_radius"]
        self.rew_vecs[self.states == STATE_INITIAL] = (
            torch.Tensor([0.0, -wall_robot_offset])
            - self.ps[self.states == STATE_INITIAL]
        )
        self.rew_vecs[self.states == STATE_PASSAGE] = (
            torch.Tensor([0.0, 3.0])
            - self.ps[self.states == STATE_PASSAGE]
        )

        self.rew_vecs[self.states >= STATE_AFTER] = (
            self.goal_ps[self.states >= 2] - self.ps[self.states >= STATE_AFTER]
        )
        rew_vecs_norm = torch.linalg.norm(self.rew_vecs, dim=2)

        self.states[self.states == STATE_REACHED_GOAL] = STATE_FINISHED
        # move to next state if distance to waypoint is small enough
        # changed from 0.1 to 0.3
        self.states[(self.states < STATE_AFTER) & (rew_vecs_norm < 0.5)] += 1
        self.states[(self.states == STATE_AFTER) & (rew_vecs_norm < 0.3)] += 1

        # reward: dense shaped reward following waypoints
        
        vs_norm = torch.linalg.norm(self.measured_vs, dim=2)
        rew_vecs_norm = torch.linalg.norm(self.rew_vecs, dim=2).unsqueeze(2)
        rewards_dense = (
            torch.bmm(
                (self.rew_vecs / rew_vecs_norm).view(-1, 2).unsqueeze(1),
                (self.measured_vs / vs_norm.unsqueeze(2)).view(-1, 2).unsqueeze(2),
            ).view(self.cfg["num_envs"], self.cfg["n_agents"])
            * vs_norm
        )
        rewards[vs_norm > 0.0] += 3.0 * rewards_dense[vs_norm > 0.0]

        # bonus when reaching the goal
        rewards[self.states == STATE_REACHED_GOAL] += 10.0

        obs = [self.get_obs(index) for index in range(self.cfg["num_envs"])]
        all_reached_goal = (self.states == STATE_FINISHED).all(1)
        timeout = self.timesteps >= self.cfg["max_time_steps"]
        dones = (all_reached_goal | timeout).tolist()
        infos = [
            {"rewards": {k: r for k, r in enumerate(env_rew)}}
            for env_rew in rewards.tolist()
        ]

        return obs, torch.sum(rewards, dim=1).tolist(), dones, infos


    def get_unwrapped(self) -> List[EnvType]:
        return []


class PassageEnvRender(PassageEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, config):
        super().__init__(config)

    def seed(self, seed=None):
        rng = torch.manual_seed(seed)
        initial = rng.initial_seed()
        return [initial]

    def reset(self):
        return self.reset_at(0)

    def step(self, actions):
        vector_actions = self.create_state_tensor()
        vector_actions[0] = torch.Tensor(actions)
        obs, r, done, info = self.vector_step(vector_actions)
        return obs[0], r[0], done[0], info[0]

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        AGENT_COLOR = BLUE
        BACKGROUND_COLOR = WHITE
        WALL_COLOR = GRAY

        index = 0

        def point_to_screen(point):
            return [
                int((p * f + world_dim / 2) * self.cfg["render_px_per_m"])
                for p, f, world_dim in zip(point, [-1, 1], self.cfg["world_dim"])
            ]

        self.display.fill(BACKGROUND_COLOR)
        img = pygame.Surface(self.display.get_size(), pygame.SRCALPHA)
        font = pygame.font.Font(None, 24)

        for agent_index in range(self.cfg["n_agents"]):
            agent_p = self.ps[index, agent_index]
            pygame.draw.circle(
                img,
                AGENT_COLOR,
                point_to_screen(agent_p),
                self.cfg["agent_radius"] * self.cfg["render_px_per_m"],
            )
            pygame.draw.line(
                img,
                AGENT_COLOR,
                point_to_screen(agent_p),
                point_to_screen(self.goal_ps[index, agent_index]),
                4,
            )
            rew_vec = self.rew_vecs[index, agent_index]
            rew_vec_norm = torch.linalg.norm(rew_vec)
            if rew_vec_norm > 0.0:
                pygame.draw.line(
                    img,
                    RED,
                    point_to_screen(agent_p),
                    point_to_screen(agent_p + rew_vec / rew_vec_norm * 0.5),
                    2,
                )

            state = self.states[index, agent_index].item()  # Get the state of the robot
            text = font.render('{}/{}'.format(state, agent_index), True, (0, 0, 0), (255, 255, 255))  # Prepare the text
            text = pygame.transform.flip(text, True, False)  # Flip the text along the x-axis
            textpos = text.get_rect(centerx=point_to_screen(agent_p)[0], centery=point_to_screen(agent_p)[1] - 20)  # Position the text
            img.blit(text, textpos)  # Draw the text

        for o in self.obstacles:
            tl = point_to_screen([o["max"][X], o["min"][Y]])
            width = [
                int((o["max"][d] - o["min"][d]) * self.cfg["render_px_per_m"])
                for d in [X, Y]
            ]
            pygame.draw.rect(img, WALL_COLOR, tl + width)
        self.display.blit(img, (0, 0))

        if mode == "human":
            pygame.display.update()
        elif mode == "rgb_array":
            return pygame.surfarray.array3d(self.display)

    def try_render_at(self, index: Optional[int] = None) -> None:
        """Renders a single environment.
        Args:
            index (Optional[int]): An optional sub-env index to render.
        """
        return self.render(mode="rgb_array")


if __name__ == "__main__":
    pentagon_coords = np.array([
        [np.cos(2 * np.pi * i / 5 + np.pi/2 + np.pi/6), np.sin(2 * np.pi * i / 5 + np.pi/2 + np.pi/6)]
        for i in range(5)
    ])

    # Define your desired scale factor
    scale_factor = 0.5

    # Scale the coordinates
    scaled_pentagon_coords = pentagon_coords * scale_factor

    # Convert to list for use in the configuration dictionary
    agent_formation = scaled_pentagon_coords.tolist()

    config = {   
            #Modified: world_dim (4.0, 6.0)
            "world_dim": (6.0, 10.0),
            "dt": 0.05,
            "num_envs": 3,
            "device": "cpu",
            "n_agents": 5,
            # Modified: scale 0.6 and formation
            "agent_formation": agent_formation,

            "placement_keepout_border": 1.0,
            "placement_keepout_wall": 1.5,
            "pos_noise_std": 0.0,
            "max_time_steps": 10000,
            # Modified: wall_width 0.3
            "wall_width": 5.5,
            # Modified: gap_length 1.0
            "gap_length": 2.0,
            "grid_px_per_m": 40,
            # Modified: agent_radius : 0.25
            "agent_radius": 0.1,
            "render": False,
            "render_px_per_m": 160,
            "max_v": 10.0,
            "max_a": 5.0,
        }
    '''
    import time
    torch.manual_seed(0)
    env.vector_reset()
    # env.reset()
    returns = torch.zeros((env.cfg["n_agents"]))
    selected_agent = 0
    rew = 0
    while True:

        a = torch.zeros((env.cfg["n_agents"], 2))
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                env.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                selected_agent += 1
                if selected_agent >= env.cfg["n_agents"]:
                    selected_agent = 0
            elif event.type == pygame.MOUSEMOTION:
                v = (
                    torch.clip(torch.Tensor([-event.rel[0], event.rel[1]]), -20, 20)
                    / 20
                )
                a[selected_agent] = v

        # env.ps[0, 0, X] = 1.0
        env.render(mode="human")

        obs, r, done, info = env.step(a)
        rew += r
        for key, agent_reward in info["rewards"].items():
            returns[key] += agent_reward
        print(returns)
        if done:
            env.reset()
            returns = torch.zeros((env.cfg["n_agents"]))'''