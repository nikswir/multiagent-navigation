"""Multi-robot 2-D navigation world: obstacles, lidar robots, SimpleEnv.

A bounded rectangular world with axis-aligned rectangular obstacles. Up to
`max_robots` disc robots (unicycle kinematics, forward half-circle lidar)
navigate to individual goals. Each robot observes a 24-d state — 20 normalized
lidar ranges plus normalized distance / relative angle to its goal and its own
linear & angular velocity — and is rewarded +100 for reaching the goal, -100
for any collision, and a shaped step reward otherwise (`get_reward`).

`max_robots` is the env's *capacity* (internal arrays are sized to it); the
*active* robot count is chosen per episode via `reset(n_robots=...)` — the
training curriculum in `lib` ramps it up over time.
"""

from __future__ import annotations

import math
import random

import numpy as np

from collections.abc import Sequence

########################################
#              Constants               #
########################################

GOAL_REACHED_DIST = 0.3
TIME_DELTA = 0.1
ROBOT_RADIUS = 0.15
LIDAR_MAX_RANGE = 5.0
MAX_ROBOTS = 16

# The default obstacle course: (x, y, width, height) rectangles.
DEFAULT_OBSTACLES: list[list[float]] = [
    [-3, 1, 1, 2],
    [1, -2, 2, 1],
    [-1, -3, 3, 0.5],
    [2, 2, 0.5, 3],
]

########################################
#          Obstacles & world           #
########################################


class Obstacle:
    """An axis-aligned rectangle: collision target and lidar occluder."""

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = (x, y, width, height)

    def intersects_segment(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float | None:
        """Distance from `p1` to the nearest side hit, or None if clear."""
        x1, y1 = p1
        x2, y2 = p2

        # ── The rectangle as its four boundary segments ──
        sides = [
            ((self.x, self.y), (self.x + self.width, self.y)),
            (
                (self.x + self.width, self.y),
                (self.x + self.width, self.y + self.height),
            ),
            (
                (self.x + self.width, self.y + self.height),
                (self.x, self.y + self.height),
            ),
            ((self.x, self.y + self.height), (self.x, self.y)),
        ]

        min_dist_to_intersection = float("inf")

        for (sx1, sy1), (sx2, sy2) in sides:
            # ── Parametric segment-segment intersection ──
            den = (x1 - x2) * (sy1 - sy2) - (y1 - y2) * (sx1 - sx2)
            if den == 0:
                continue
            t_num = (x1 - sx1) * (sy1 - sy2) - (y1 - sy1) * (sx1 - sx2)
            u_num = -((x1 - x2) * (y1 - sy1) - (y1 - y2) * (x1 - sx1))
            t = t_num / den
            u = u_num / den

            if 0 <= t <= 1 and 0 <= u <= 1:
                dist = math.sqrt(
                    ((x1 + t * (x2 - x1)) - x1) ** 2
                    + ((y1 + t * (y2 - y1)) - y1) ** 2,
                )
                min_dist_to_intersection = min(min_dist_to_intersection, dist)

        if min_dist_to_intersection != float("inf"):
            return min_dist_to_intersection
        return None


class World:
    """The bounded field: interior obstacles plus four enclosing walls."""

    def __init__(
        self,
        width: float,
        height: float,
        obstacle_definitions: Sequence[Sequence[float]],
    ) -> None:
        self.width = width
        self.height = height

        # ── Interior obstacles, then walls just outside the bounds ──
        self.obstacles = [
            Obstacle(x, y, w, h) for x, y, w, h in obstacle_definitions
        ]
        self.world_bounds = [
            Obstacle(-width / 2 - 1, -height / 2 - 1, 1, height + 2),
            Obstacle(width / 2, -height / 2 - 1, 1, height + 2),
            Obstacle(-width / 2 - 1, -height / 2 - 1, width + 2, 1),
            Obstacle(-width / 2 - 1, height / 2, width + 2, 1),
        ]
        self.all_obstacles = self.obstacles + self.world_bounds

    def check_collision_robot(
        self,
        robot_x: float,
        robot_y: float,
        robot_radius: float,
    ) -> bool:
        """True if a disc at (x, y) overlaps any obstacle or wall."""
        for obs in self.all_obstacles:
            # ── Closest rectangle point to the disc centre ──
            closest_x = max(obs.x, min(robot_x, obs.x + obs.width))
            closest_y = max(obs.y, min(robot_y, obs.y + obs.height))
            distance_x = robot_x - closest_x
            distance_y = robot_y - closest_y

            if (distance_x**2 + distance_y**2) < robot_radius**2:
                return True

        return False


########################################
#                Robot                 #
########################################


class Robot:
    """A disc robot with unicycle kinematics and a forward lidar fan."""

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        theta: float = 0,
        robot_radius: float = ROBOT_RADIUS,
        lidar_num_beams: int = 20,
        lidar_max_range: float = LIDAR_MAX_RANGE,
    ) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.radius = robot_radius
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # ── Lidar: a half-circle fan; range extended by the body radius ──
        self.lidar_num_beams = lidar_num_beams
        self.lidar_max_range = lidar_max_range + robot_radius
        self.lidar_fov = math.pi
        self.lidar_start_angle_offset = -self.lidar_fov / 2
        self.current_lidar_data = (
            np.ones(self.lidar_num_beams) * self.lidar_max_range
        )
        self.current_lidar_data_display = self.current_lidar_data.copy()

    def set_velocity(self, linear: float, angular: float) -> None:
        self.linear_velocity = linear
        self.angular_velocity = angular

    def update_pose(self, dt: float) -> None:
        """Integrate the unicycle model; heading wraps to [-pi, pi)."""
        self.theta += self.angular_velocity * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        self.x += self.linear_velocity * math.cos(self.theta) * dt
        self.y += self.linear_velocity * math.sin(self.theta) * dt

    def intersects_segment(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float | None:
        """Distance from `p1` to this robot's disc along p1->p2, or None."""
        x1, y1 = p1
        x2, y2 = p2

        # ── Ray direction and the segment's length ──
        vec_x = x2 - x1
        vec_y = y2 - y1
        segment_length = math.sqrt(vec_x**2 + vec_y**2)
        ray_dx = vec_x / segment_length
        ray_dy = vec_y / segment_length

        # ── Quadratic ray-circle intersection ──
        ox = x1 - self.x
        oy = y1 - self.y
        a = 1.0
        b = 2 * (ox * ray_dx + oy * ray_dy)
        c = ox**2 + oy**2 - self.radius**2
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None

        sqrt_discriminant = math.sqrt(discriminant)
        root1 = (-b - sqrt_discriminant) / (2 * a)
        root2 = (-b + sqrt_discriminant) / (2 * a)

        # ── Keep roots that fall on the segment itself ──
        valid_roots = []
        if 0 <= root1 <= segment_length:
            valid_roots.append(root1)
        if 0 <= root2 <= segment_length:
            valid_roots.append(root2)

        if not valid_roots:
            return None
        return min(valid_roots)

    def lidar_scan(
        self,
        world_obstacles: list[Obstacle],
        robots: list[Robot],
    ) -> np.ndarray:
        """Sweep the fan; readings are body-relative (radius subtracted)."""
        scan_data = np.ones(self.lidar_num_beams) * self.lidar_max_range

        for i in range(self.lidar_num_beams):
            # ── 1. Beam endpoint at max range in world coordinates ──
            if self.lidar_num_beams == 1:
                relative_beam_angle = (
                    self.lidar_start_angle_offset + 0.5 * self.lidar_fov
                )
            else:
                relative_beam_angle = (
                    self.lidar_start_angle_offset
                    + (i / (self.lidar_num_beams - 1)) * self.lidar_fov
                )
            global_beam_angle = self.theta + relative_beam_angle
            x_end_far = self.x + self.lidar_max_range * math.cos(
                global_beam_angle,
            )
            y_end_far = self.y + self.lidar_max_range * math.sin(
                global_beam_angle,
            )
            p1 = (self.x, self.y)
            p2 = (x_end_far, y_end_far)

            # ── 2. Nearest hit over obstacles and the other robots ──
            min_dist_to_obstacle = self.lidar_max_range

            for obs in world_obstacles:
                intersection_dist = obs.intersects_segment(p1, p2)
                if (
                    intersection_dist is not None
                    and intersection_dist < min_dist_to_obstacle
                ):
                    min_dist_to_obstacle = intersection_dist

            for other_robot in robots:
                if other_robot is self:
                    continue
                intersection_dist = other_robot.intersects_segment(p1, p2)
                if (
                    intersection_dist is not None
                    and intersection_dist < min_dist_to_obstacle
                ):
                    min_dist_to_obstacle = intersection_dist

            scan_data[i] = min_dist_to_obstacle

        # ── Body-relative readings; raw distances kept for display ──
        self.current_lidar_data = scan_data - self.radius
        self.current_lidar_data_display = scan_data

        return self.current_lidar_data

    def check_collision_to_robots(self, other_robots: list[Robot]) -> bool:
        for other_robot in other_robots:
            if other_robot is self:
                continue
            dist = math.sqrt(
                (self.x - other_robot.x) ** 2 + (self.y - other_robot.y) ** 2,
            )
            if dist < self.radius + other_robot.radius:
                return True
        return False


########################################
#             Environment              #
########################################


class SimpleEnv:
    """The multi-robot episode: spawn, step, reward and termination."""

    def __init__(
        self,
        world_width: float = 10,
        world_height: float = 10,
        environment_dim: int = 20,
        robot_radius: float = ROBOT_RADIUS,
        max_steps: int = 200,
        n_robots: int = MAX_ROBOTS,
        max_robots: int = MAX_ROBOTS,
        time_delta: float = TIME_DELTA,
        goal_reached_dist: float = GOAL_REACHED_DIST,
        lidar_max_range: float = LIDAR_MAX_RANGE,
        obstacle_definitions: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if obstacle_definitions is None:
            obstacle_definitions = DEFAULT_OBSTACLES

        self.n_robots = n_robots
        self.max_robots = max_robots
        self.time_delta = time_delta
        self.goal_reached_dist = goal_reached_dist

        # ── Arrays are sized to capacity; episodes use the first n ──
        self.world = World(world_width, world_height, obstacle_definitions)
        self.robots = [
            Robot(
                robot_radius=robot_radius,
                lidar_num_beams=environment_dim,
                lidar_max_range=lidar_max_range,
            )
            for _ in range(max_robots)
        ]
        self.goals_x = np.zeros(max_robots)
        self.goals_y = np.zeros(max_robots)

        self.current_step = 0
        self.max_steps = max_steps

        self.reset(self.n_robots)

    ########################################
    #           Episode sampling           #
    ########################################

    def _is_position_valid(
        self,
        x: float,
        y: float,
        radius: float,
        robot_idx: int | None = None,
    ) -> bool:
        # ── 1. Clear of obstacles and walls ──
        if self.world.check_collision_robot(x, y, radius):
            return False

        # ── 2. Inside the (slightly shrunk) world bounds ──
        if not (
            -self.world.width / 2 * 0.95 < x < self.world.width / 2 * 0.95
            and -self.world.height / 2 * 0.95 < y < self.world.height / 2 * 0.95
        ):
            return False

        # ── 3. Clear of the robots already placed before this one ──
        if robot_idx is not None:
            for j in range(robot_idx):
                dist = math.sqrt(
                    (x - self.robots[j].x) ** 2 + (y - self.robots[j].y) ** 2,
                )
                if dist < radius + self.robots[j].radius:
                    return False

        return True

    def set_random_goal(self) -> None:
        for i in range(self.n_robots):
            while True:
                self.goals_x[i] = random.uniform(
                    -self.world.width / 2 * 0.8,
                    self.world.width / 2 * 0.8,
                )
                self.goals_y[i] = random.uniform(
                    -self.world.height / 2 * 0.8,
                    self.world.height / 2 * 0.8,
                )

                if not self._is_position_valid(
                    self.goals_x[i],
                    self.goals_y[i],
                    0.05,
                ):
                    continue

                # ── Keep goals mutually separated by the body radii ──
                if i == 0:
                    break
                repeat = False
                for j in range(i):
                    dist = math.sqrt(
                        (self.goals_x[i] - self.goals_x[j]) ** 2
                        + (self.goals_y[i] - self.goals_y[j]) ** 2,
                    )
                    if dist < self.robots[i].radius + self.robots[j].radius:
                        repeat = True
                if not repeat:
                    break

    def set_random_start_pos(self) -> None:
        for i in range(self.n_robots):
            while True:
                self.robots[i].x = random.uniform(
                    -self.world.width / 2 * 0.8,
                    self.world.width / 2 * 0.8,
                )
                self.robots[i].y = random.uniform(
                    -self.world.height / 2 * 0.8,
                    self.world.height / 2 * 0.8,
                )
                self.robots[i].theta = random.uniform(-math.pi, math.pi)

                if self._is_position_valid(
                    self.robots[i].x,
                    self.robots[i].y,
                    self.robots[i].radius,
                    i,
                ):
                    break

    def reset(self, n_robots: int | None = None) -> list[np.ndarray]:
        """Start a fresh episode with `n_robots` active robots."""
        if n_robots is None:
            n_robots = self.max_robots

        self.current_step = 0
        self.n_robots = n_robots

        for i in range(self.n_robots):
            self.robots[i].set_velocity(0, 0)

        self.set_random_start_pos()
        self.set_random_goal()

        self.current_state = [self._get_state(i) for i in range(self.n_robots)]
        self.episode_done = [False for _ in range(self.n_robots)]
        self.cumulative_reward = [0.0 for _ in range(self.n_robots)]
        self.info: list[dict[str, bool | str] | None] = [
            None for _ in range(self.n_robots)
        ]

        return self.current_state

    ########################################
    #          Observation & step          #
    ########################################

    def _get_state(self, i: int) -> np.ndarray:
        robot = self.robots[i]

        # ── 1. Normalized lidar over obstacles and the active robots ──
        scan = robot.lidar_scan(
            self.world.all_obstacles,
            self.robots[: self.n_robots],
        )
        norm_lidar_data = scan / (robot.lidar_max_range - robot.radius)

        # ── 2. Normalized distance to this robot's own goal ──
        norm_dist_to_goal = math.sqrt(
            (robot.x - self.goals_x[i]) ** 2 + (robot.y - self.goals_y[i]) ** 2,
        ) / math.sqrt(self.world.width**2 + self.world.height**2)

        # ── 3. Relative goal bearing wrapped to [-pi, pi), scaled by pi ──
        angle_to_goal = math.atan2(
            self.goals_y[i] - robot.y,
            self.goals_x[i] - robot.x,
        )
        relative_angle_to_goal = angle_to_goal - robot.theta
        relative_angle_to_goal = (relative_angle_to_goal + math.pi) % (
            2 * math.pi
        ) - math.pi
        norm_relative_angle_to_goal = relative_angle_to_goal / math.pi

        # ── 4. Lidar plus the robot's own scalar state (24-d total) ──
        robot_state_part = [
            norm_dist_to_goal,
            norm_relative_angle_to_goal,
            robot.linear_velocity,
            robot.angular_velocity,
        ]
        return np.concatenate((norm_lidar_data, np.array(robot_state_part)))

    def step(
        self,
        action: Sequence[Sequence[float]] | np.ndarray,
    ) -> tuple[
        list[np.ndarray],
        list[float],
        list[bool],
        list[dict[str, bool | str] | None],
    ]:
        """Advance every active robot one tick; done robots are skipped."""
        rewards: list[float] = []
        self.current_state = []
        self.current_step += 1

        # ── 1. Apply actions and integrate poses for live robots ──
        for i in range(self.n_robots):
            if self.episode_done[i]:
                continue
            linear_vel, angular_vel = action[i]
            self.robots[i].set_velocity(linear_vel, angular_vel)
            self.robots[i].update_pose(self.time_delta)

        # ── 2. Observe, reward and settle termination per robot ──
        for i in range(self.n_robots):
            self.current_state.append(self._get_state(i))

            if self.episode_done[i]:
                rewards.append(0)
                continue

            collision = self.world.check_collision_robot(
                self.robots[i].x,
                self.robots[i].y,
                self.robots[i].radius,
            ) or self.robots[i].check_collision_to_robots(
                self.robots[: self.n_robots],
            )

            dist_to_goal = math.sqrt(
                (self.robots[i].x - self.goals_x[i]) ** 2
                + (self.robots[i].y - self.goals_y[i]) ** 2,
            )
            target_reached = dist_to_goal < self.goal_reached_dist

            self.episode_done[i] = (
                collision
                or target_reached
                or (self.current_step >= self.max_steps)
            )

            min_laser_reading = (
                float(min(self.robots[i].current_lidar_data))
                if len(self.robots[i].current_lidar_data) > 0
                else self.robots[i].lidar_max_range
            )
            reward_i = self.get_reward(
                target_reached,
                collision,
                action[i],
                min_laser_reading,
            )
            self.cumulative_reward[i] += reward_i
            rewards.append(reward_i)

            # ── Terminal diagnostics for logging and evaluation ──
            info_i: dict[str, bool | str] = {
                "target_reached": target_reached,
                "collision": collision,
            }
            self.info[i] = info_i
            if self.current_step >= self.max_steps and not (
                target_reached or collision
            ):
                info_i["reason"] = "max_steps_reached"
            elif target_reached:
                info_i["reason"] = "target_reached"
            elif collision:
                info_i["reason"] = "collision"

        return self.current_state, rewards, self.episode_done, self.info

    @staticmethod
    def get_reward(
        target: bool,
        collision: bool,
        action: Sequence[float] | np.ndarray,
        min_laser: float,
    ) -> float:
        """+100 at the goal, -100 on any collision, shaped step otherwise."""
        if target:
            return 100.0
        if collision:
            return -100.0

        # ── Proximity penalty: active once the nearest reading is < 1 ──
        def r3(x: float) -> float:
            return 1 - x if x < 1 else 0.0

        reward = (
            float(action[0]) / 2 - abs(float(action[1])) / 2 - r3(min_laser)
        )
        reward -= 0.05
        return reward
