from pprint import pprint
import matplotlib.pyplot as plt
import time
import random
import prior
import math
import heapq
import numpy as np
import json
from pydantic import BaseModel
from collections import defaultdict
from scipy.spatial import cKDTree
import copy


from constants import GRID_SIZE
import utils


class ActionRealCost(BaseModel):
    action: str  # pickup(RemoteControl|surface|3|18)
    state: str  # at(Sofa|3|3)
    value: float  # 6.8


def direction_from_to(p1, p2):
    dx = round(p2[0] - p1[0], 4)
    dz = round(p2[1] - p1[1], 4)

    if dx == 0 and dz > 0:
        return 0  # +Z
    elif dx > 0 and dz == 0:
        return 90  # +X
    elif dx == 0 and dz < 0:
        return 180  # -Z
    elif dx < 0 and dz == 0:
        return 270  # -X
    else:
        raise ValueError(f"Invalid step from {p1} to {p2}, must be axis-aligned.")


def get_turn_actions(current_ori, target_ori):
    diff = (target_ori - current_ori) % 360
    if diff == 0:
        return []
    elif diff == 90:
        return ["turn_right"]
    elif diff == 180:
        return ["turn_right", "turn_right"]
    elif diff == 270:
        return ["turn_left"]
    else:
        raise ValueError(
            f"Unexpected orientation change from {current_ori} to {target_ori}"
        )


def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def direction_to_vector(direction):
    # Converts direction in degrees to movement vector
    return {
        0: (GRID_SIZE, 0),  # up
        90: (0, GRID_SIZE),  # right
        180: (-GRID_SIZE, 0),  # down
        270: (0, -GRID_SIZE),  # left
    }[direction % 360]


class MotionPlanner:
    def __init__(self, controller, fur_geo, scene_id):

        self.controller = controller
        self.fur_geo = fur_geo
        self.scene_id = scene_id

    def execute(self, action):
        # motion planner systematically collects the cost of executing each action
        # cost = +inf if the action fails

        print(f"Motion planner executing action {action}")

        action_info = {}

        if action.name == "navigate":

            fur_id, room_id = action.argument.split(",")
            fur_id = fur_id.strip()

            fur_points = self.sample_points_around_furniture(fur_id)

            # navigate the robot to a nearby location around the furniture
            fur_loc = tuple(fur_points[0])

            (
                collision_counts,
                navigation_time,
                navigated_distance,
                navigated_path,
            ) = self.navigate(fur_loc)

            action_cost = 10 * collision_counts + navigation_time + navigated_distance

            action_info = {
                "action_type": "navigate",
                "action_arguments": [fur_id, room_id],
                "collision_counts": collision_counts,
                "navigation_time": round(navigation_time, 2),
                "navigated_distance": navigated_distance,
                "navigated_path": navigated_path,
                "action_cost": round(action_cost),
            }

            return action_info

        elif action.name == "pickup":

            obj_id, fur_id = action.argument.split(",")
            obj_id = obj_id.strip()
            fur_id = fur_id.strip()

            fur_points = self.sample_points_around_furniture(fur_id)

            # try to pick up the object at the sampled points
            success, pickup_time, pickup_trials = self.pickup(obj_id, fur_points)

            if success:
                print("Success! The robot picked up the object", obj_id)
                action_cost = pickup_trials + pickup_time
            else:
                print("Failed to pick up the object", obj_id)
                action_cost = 10 + pickup_time

            action_info = {
                "action_type": "pickup",
                "action_arguments": [obj_id, fur_id],
                "success": success,
                "pickup_time": round(pickup_time, 2),
                "pickup_trials": pickup_trials,
                "action_cost": round(action_cost),
            }

            return action_info

        elif action.name == "place":

            obj_id, fur_id = action.argument.split(",")
            obj_id = obj_id.strip()
            fur_id = fur_id.strip()

            held_objects = self.controller.last_event.metadata["arm"]["heldObjects"]

            if obj_id in held_objects:

                success, place_time = self.place(
                    obj_id, fur_id
                )  # try to place the object on the furniture

                if success:

                    print(
                        f"Success! The robot placed the object {obj_id} on the furniture {fur_id}"
                    )

                    action_cost = 1 + place_time

                    action_info = {
                        "action_type": "place",
                        "action_arguments": [obj_id, fur_id],
                        "success": True,
                        "place_time": round(place_time, 2),
                        "action_cost": round(action_cost),
                    }

                else:
                    print(
                        f"Failed to place the object {obj_id} on the furniture {fur_id}"
                    )

                    action_cost = 5 + place_time

                    action_info = {
                        "action_type": "place",
                        "action_arguments": [obj_id, fur_id],
                        "success": False,
                        "place_time": round(place_time, 2),
                        "action_cost": round(action_cost),
                    }

            else:
                # robot is not holding the object, so 'place' is a dummy action and of course it is a waste of time

                print(
                    f"The robot is not holding the object {obj_id}, so the 'place' action is dummy execution."
                )

                waste_of_time = 2

                time.sleep(waste_of_time)

                action_cost = 5 + waste_of_time

                action_info = {
                    "action_type": "place",
                    "action_arguments": [obj_id, fur_id],
                    "success": None,
                    "place_time": waste_of_time,
                    "action_cost": round(action_cost),
                }

            return action_info

    def navigate(self, goal_location):
        # navigate the robot to a goal location based on the default grid map
        # replan with the high-resolution map when the robot collides with something
        # goal_location: (x, z)

        collision_counts = 0

        start_time = time.time()

        navigation_steps = 0

        navigated_path = []

        while True:

            free_grids = utils.thor_reachable_positions(self.controller)

            # make sure the goal location is a free grid
            valid_goal_location = utils.find_closest_free_grid(
                goal_location, free_grids
            )

            # plan a list of free grids towards the goal location
            planned_path = self.a_star(valid_goal_location, free_grids)

            # convert the path to a list of navigation steps (move_ahead, turn_left, turn_right)
            if planned_path is None:
                input("Error! No path found")

            else:
                steps = self.convert_path_to_steps(planned_path)

                if steps == []:
                    print("Success! The robot arrives at the goal location")
                    break

            for step in steps:

                collided = self.go(step)

                navigation_steps += 1

                # real navigated path consisting of real-time robot locations
                navigated_path.append(utils.thor_agent_location(self.controller))

                if collided:
                    # replan a path with the high-resolution map when the robot collides with something

                    collision_counts += 1

                    print(
                        "Attention! The robot collided with object during navigation."
                    )
                    print("Replanning path...")

                    free_grids = utils.read_high_resolution_gridmap(self.scene_id)

                    # make sure the goal location is a free grid
                    valid_goal_location = utils.find_closest_free_grid(
                        goal_location, free_grids
                    )

                    # replan a new path
                    new_planned_path = self.a_star(valid_goal_location, free_grids)

                    self.plot_path(free_grids, new_planned_path, valid_goal_location)

                    # convert the path to a list of steps (move_ahead, turn_left, turn_right)
                    if new_planned_path is None:
                        input("Error! No path found")
                    else:
                        new_steps = self.convert_path_to_steps(new_planned_path)

                    # execute the new plan for at most 10 steps
                    for step in new_steps[:10]:

                        self.go(step)

                        navigation_steps += 1

                        # real navigated path consisting of real-time robot locations
                        navigated_path.append(
                            utils.thor_agent_location(self.controller)
                        )

                    # back to the planning loop with the default grid map
                    break

        navigation_time = time.time() - start_time

        navigated_distance = navigation_steps * GRID_SIZE

        return collision_counts, navigation_time, navigated_distance, navigated_path

    def pickup(self, obj_id, fur_points):
        # try to pick up an object from points sampled around a furniture

        start_time = time.time()

        for ii in range(len(fur_points)):

            # since pickup requires nuanced movement around the furniture,
            # we directly use the high resolution gridmap
            free_grids = utils.read_high_resolution_gridmap(self.scene_id)

            # make sure the goal location is a free grid
            goal_location = utils.find_closest_free_grid(fur_points[ii], free_grids)

            # may need to pay attention to the initial rotation of facing the object
            # since the camera may detect the object from two directions, but one direction is better for picking up
            path = self.a_star(goal_location, free_grids)

            if path is None:
                input("Error! No path found")
            else:
                steps = self.convert_path_to_steps(path)

            for step in steps:
                self.go(step)

            # turn the robot to face the furniture
            # face_furniture(fur_id)
            self.face_target_vision(obj_id)
            time.sleep(1)

            # attempt to pick up the object
            object_picked = self.attempt_pickup(obj_id)
            time.sleep(1)

            self.reset_arm()

            if object_picked:
                # successfully picked up the object

                self.reset_arm()

                success = True
                action_time = time.time() - start_time
                pickup_trials = ii + 1

                return success, action_time, pickup_trials

        success = False
        action_time = time.time() - start_time
        pickup_trials = len(fur_points)

        return success, action_time, pickup_trials

    def place(self, obj_id, fur_id):
        # place the object in hand on the furniture

        start_time = time.time()

        # try to place the object on the furniture
        self.face_target_vision(fur_id)
        self.release_object(fur_id)
        self.reset_arm()

        containing_objects = utils.thor_object_with_id(self.controller, fur_id)[
            "receptacleObjectIds"
        ]

        if obj_id in containing_objects:
            success = True
            action_time = time.time() - start_time

            return success, action_time

        else:
            success = False
            action_time = time.time() - start_time

            return success, action_time

    def face_furniture(self, fur_id):
        # turn the robot to face the furniture center

        fur_center = self.fur_geo[fur_id]["position"]

        c_x, c_z = fur_center[0], fur_center[1]

        # robot_pos: (x, y, z), (pitch, yaw, roll)
        robot_pos = utils.thor_agent_pose(self.controller)
        r_x, r_z = robot_pos[0]["x"], robot_pos[0]["z"]

        dx = c_x - r_x
        dz = c_z - r_z

        if abs(dx) > abs(dz):
            if dx > 0:
                self.turn_right()
            else:
                self.turn_left()

        else:
            self.turn_left()
            self.turn_left()

    def move_arm_to_position(self, target_position):
        # move arm to a position in the global coordinate system

        event = self.controller.step(
            action="MoveArm", position=target_position, coordinateSpace="world"
        )

        if not event.metadata["lastActionSuccess"]:
            print(f"Failed to move the arm. Error: {event.metadata['errorMessage']}")

        return

    def attempt_pickup(self, obj_id):

        obj_position = utils.thor_object_with_id(self.controller, obj_id)["position"]

        self.move_arm_to_position(obj_position)

        """
        arm_metadata = self.controller.last_event.metadata['arm']

        hand_sphere_center = arm_metadata['handSphereCenter']

        hand_sphere_radius = arm_metadata['handSphereRadius']
        
        distance_to_object = calculate_distance(hand_sphere_center, obj_position)
        print(f"Distance to object: {distance_to_object}")
        
        # If the object is out of reach, move the arm closer
        if distance_to_object > hand_sphere_radius:
            print("Object is out of reach. Moving the arm closer...")

            move_arm_to_position(obj_position)'
        """

        event = self.controller.step(action="PickupObject", objectIdCandidates=[obj_id])

        if event.metadata["lastActionSuccess"]:
            # print(f"Successfully picked up the object {obj_id}")
            return True
        else:
            # print(f"Failed to pick up the object {obj_id}. Error: {event.metadata['errorMessage']}")
            return False

    def release_object(self, fur_id):

        fur_center = utils.thor_object_with_id(self.controller, fur_id)["position"]

        arm_metadata = self.controller.last_event.metadata["arm"]
        hand_sphere_center = arm_metadata["handSphereCenter"]
        hand_sphere_radius = arm_metadata["handSphereRadius"]

        distance_to_object = utils.calculate_distance(hand_sphere_center, fur_center)

        if distance_to_object > hand_sphere_radius:
            self.move_arm_to_position(fur_center)

        event = self.controller.step(action="ReleaseObject")

        time.sleep(1)

        if not event.metadata["lastActionSuccess"]:
            input(
                f"Failed to release the object. Error: {event.metadata['errorMessage']}"
            )

    def neighbors(self, pos, direction, free_grids_set):
        moves = []

        # Turn left
        left_dir = (direction - 90) % 360
        moves.append((pos, left_dir))

        # Turn right
        right_dir = (direction + 90) % 360
        moves.append((pos, right_dir))

        # Move ahead
        dx, dy = direction_to_vector(direction)
        new_pos = (round(pos[0] + dx, 2), round(pos[1] + dy, 2))
        if new_pos in free_grids_set:
            moves.append((new_pos, direction))

        return moves

    def a_star(self, goal_location, free_grids):
        # goal_location: (x, z)

        start_location = utils.thor_agent_location(self.controller)

        print("motion planner astar start_location", start_location)

        goal_x = round(goal_location[0] / GRID_SIZE) * GRID_SIZE
        goal_z = round(goal_location[1] / GRID_SIZE) * GRID_SIZE
        goal_location = (goal_x, goal_z)

        print("motion planner astar goal_location", goal_location)

        free_grids_set = set(tuple(cell) for cell in free_grids)

        start_dir = 0  # robot starts facing up (0 degrees)

        frontier = []
        heapq.heappush(frontier, (0, start_location, start_dir, [start_location]))
        visited = set()

        while frontier:

            _, current_location, current_dir, path = heapq.heappop(frontier)
            state = (current_location, current_dir)

            if current_location == goal_location:
                return path

            if state in visited:
                continue

            visited.add(state)

            for next_location, next_dir in self.neighbors(
                current_location, current_dir, free_grids_set
            ):
                if (next_location, next_dir) in visited:
                    continue
                new_path = (
                    path
                    if next_location == current_location
                    else path + [next_location]
                )
                cost = len(new_path) + heuristic(next_location, goal_location)
                heapq.heappush(frontier, (cost, next_location, next_dir, new_path))

        return None  # no path found

    def convert_path_to_steps(self, path):
        # Convert a path of 2D locations to a list of actions (move_ahead, turn_left, turn_right)

        if path is None:
            input("Error! No path found")

        if (
            len(path) == 1
        ):  # start location and goal location are the same, no movement needed
            return []

        start_pose = utils.thor_agent_pose(self.controller)
        start_orientation = start_pose[1]["y"]

        steps = []
        orientation = start_orientation

        for i in range(len(path) - 1):
            current = path[i]
            next_pt = path[i + 1]

            desired_ori = direction_from_to(current, next_pt)
            turns = get_turn_actions(orientation, desired_ori)
            steps.extend(turns)
            steps.append("move_ahead")
            orientation = desired_ori  # update current orientation

        return steps

    def plot_path(self, grid_map, path, goal_location):
        """
        Plots the grid map and overlays the A* path on top.

        Args:
            grid_map (list of [x, z]): List of free grid coordinates.
            path (list of [x, z]): A* path from start to goal.
            start_location ([x, z]): Starting location.
            goal_location ([x, z]): Goal location.
        """

        start_location = utils.thor_agent_location(self.controller)

        goal_x = round(goal_location[0] / GRID_SIZE) * GRID_SIZE
        goal_z = round(goal_location[1] / GRID_SIZE) * GRID_SIZE
        goal_location = (goal_x, goal_z)

        grid_map = list(set(tuple(pt) for pt in grid_map))  # remove duplicates

        # unzip grid map points
        grid_x, grid_z = zip(*grid_map)

        # unzip path points
        if path:
            path_x, path_z = zip(*path)
        else:
            path_x, path_z = [], []

        plt.figure(figsize=(8, 8))
        plt.scatter(grid_x, grid_z, color="gray", label="Free Grid")
        plt.plot(path_x, path_z, color="blue", linewidth=2, label="A* Path")

        # start and goal
        plt.scatter(*start_location, color="green", s=100, label="Start")
        plt.scatter(*goal_location, color="red", s=100, label="Goal")

        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.title("Grid Map with A* Path")
        plt.legend()
        plt.axis("equal")
        plt.pause(3)
        plt.close()

        return

    def go(self, step):

        if step == "move_ahead":
            collided = self.move_ahead()

        elif step == "turn_left":
            collided = self.turn_left()

        elif step == "turn_right":
            collided = self.turn_right()

        return collided

    def move_ahead(self):
        # actually move ahead in the simulator

        event = self.controller.step(
            action="MoveAgent",
            ahead=GRID_SIZE,
            right=0.0,
            returnToStart=True,
            speed=1,
            fixedDeltaTime=0.02,
        )

        collided = None

        if event.metadata["lastActionSuccess"]:
            collided = False

        elif "Collided" in event.metadata["errorMessage"]:
            print(
                f"Error! {event.metadata['errorMessage']} Robot location {utils.thor_agent_location(self.controller)}"
            )
            collided = True

        return collided

    def turn_left(self):
        # actually turn left in the simulator

        event = self.controller.step(
            action="RotateAgent",
            degrees=-90,
            returnToStart=True,
            speed=1,
            fixedDeltaTime=0.02,
        )

        collided = None

        if event.metadata["lastActionSuccess"]:
            collided = False

        elif "Collided" in event.metadata["errorMessage"]:
            print(
                f"Error! {event.metadata['errorMessage']} Robot location {utils.thor_agent_location(self.controller)}"
            )
            collided = True

        return collided

    def turn_right(self):
        # actually turn right in the simulator

        event = self.controller.step(
            action="RotateAgent",
            degrees=90,
            returnToStart=True,
            speed=1,
            fixedDeltaTime=0.02,
        )

        collided = None

        if event.metadata["lastActionSuccess"]:
            collided = False

        elif "Collided" in event.metadata["errorMessage"]:
            print(
                f"Error! {event.metadata['errorMessage']} Robot location {utils.thor_agent_location(self.controller)}"
            )
            collided = True

        return collided

    def move_arm_by_distance(self, distancex, distancey, distancez):
        # distancex: left and right
        # distancey: up and down
        # distancez: forward and backward

        current_arm_position = self.controller.last_event.metadata["arm"]["joints"][0][
            "position"
        ]

        new_position = {
            "x": current_arm_position["x"] + distancex,
            "y": current_arm_position["y"] + distancey,
            "z": current_arm_position["z"] + distancez,
        }

        event = self.controller.step(
            action="MoveArm", position=new_position, coordinateSpace="world"
        )

        if not event.metadata["lastActionSuccess"]:
            print(f"Failed to move the arm. Error: {event.metadata['errorMessage']}")

        return

    def sample_edge(self, edge):
        """
        Sample points along an edge based on grid size, inclusive.
        """

        x0, z0 = edge[0]
        x1, z1 = edge[1]

        points = []

        if x0 == x1:

            if z0 > z1:
                z_list = [
                    round(z0 - i * GRID_SIZE, 6)
                    for i in range(int((z0 - z1) / GRID_SIZE) + 1)
                ]
            else:
                z_list = [
                    round(z0 + i * GRID_SIZE, 6)
                    for i in range(int((z1 - z0) / GRID_SIZE) + 1)
                ]

            for z in z_list:
                points.append((x0, z))

        elif z0 == z1:

            if x0 > x1:
                x_list = [
                    round(x0 - i * GRID_SIZE, 6)
                    for i in range(int((x0 - x1) / GRID_SIZE) + 1)
                ]
            else:
                x_list = [
                    round(x0 + i * GRID_SIZE, 6)
                    for i in range(int((x1 - x0) / GRID_SIZE) + 1)
                ]

            for x in x_list:
                points.append((x, z0))

        return points

    def edge_is_aligned_with_wall(self, a, b, wall, epsilon):
        # returns True if edge (a->b) is vertical or horizontal and
        # is within `epsilon` of wall.x or wall.z.

        (x0, z0), (x1, z1) = a, b
        (wx, wz) = wall

        # if vertical
        if abs(x0 - x1) < 1e-7:
            if abs(x0 - wx) < epsilon:
                return True

        # if horizontal
        if abs(z0 - z1) < 1e-7:
            if abs(z0 - wz) < epsilon:
                return True

        return False

    def face_target_vision(self, obj_id):

        # check if the robot can see the target
        target_visible = utils.thor_object_with_id(self.controller, obj_id)["visible"]

        max_turns = 3
        turns_done = 0

        while not target_visible and turns_done <= max_turns:
            print(f"{obj_id} not visible, turn right attempt #{turns_done + 1}...")

            self.turn_right()

            turns_done += 1

            # re-check if the robot can see the target
            target_visible = utils.thor_object_with_id(self.controller, obj_id)[
                "visible"
            ]

        if target_visible:
            print(f"{obj_id} became visible after {turns_done} turns.")
        else:
            print(
                f"{obj_id} still not visible after {turns_done} turns. Proceeding anyway..."
            )

    def sample_points_around_furniture(self, fur_id):
        """
        1. Build the 4 edges from the 4 bed corners.
        2. Mark edges that are "aligned" with wall_1 or wall_2
        if they are very close to the wall center (within dist_thres).
        3. The other edges are "non-aligned."
        4. Sample n_edge_1 points on the first non-aligned edge,
        and n_edge_2 points on the second, returning total n_edge_1 + n_edge_2 points.
        """

        fur_points = []

        free_edges = self.fur_geo[fur_id]["free_edges"]

        for edge in free_edges:

            points = self.sample_edge(edge)

            for point in points:

                fur_points.append(point)

        # remove duplicate points
        fur_points = list(set(fur_points))

        return fur_points

    def reset_arm(self):
        # reset the arm pose to point upwards
        # so that the arm will not collide with nearby stuff when the robot is turning around

        self.move_arm_by_distance(distancex=0, distancey=0.5, distancez=0)

        return

    def update_scene_graph(self, action_info, scene_graph):
        # update scene graph after executing each action

        if action_info["action_type"] == "pickup":
            # success == True: successfully picked up the object
            # success == False: failed to pick up the object, it is left on the furniture

            obj_id = action_info["action_arguments"][0]
            fur_id = action_info["action_arguments"][1]
            success = action_info["success"]

            if success:  # remove obj_id from fur_id in the scene graph
                scene_graph = utils.remove_object_from_furniture(
                    scene_graph, obj_id, fur_id
                )

        elif action_info["action_type"] == "place":
            # success == True: successfully placed the object on the furniture
            # success == False: failed to place the object on the furniture, it dropped off onto the ground
            # success == None: not even holding the object, the object is left on its original furniture

            obj_id = action_info["action_arguments"][0]
            fur_id = action_info["action_arguments"][1]
            success = action_info["success"]

            if success:  # add obj_id into fur_id in the scene graph
                scene_graph = utils.add_object_to_furniture(scene_graph, obj_id, fur_id)

        return scene_graph

    def update_known_costs(
        self, plan_info, known_navigation_costs, known_manipulation_costs
    ):
        # Update known costs by the newly collected action costs
        # If we encounter a new action, we directly append it to the known costs.
        # If we encounter the same action, we take average of the cost.

        cur_room, cur_furniture = utils.get_cur_room_furniture(
            self.scene_id, self.controller
        )

        for action_info in plan_info:

            if action_info["action_type"] == "navigate":

                dest_furniture = action_info["action_arguments"][0]
                navigated_path = action_info["navigated_path"]
                value = action_info["action_cost"]

                new_cost = {
                    "action": "navigate",
                    "start_furniture": cur_furniture,
                    "dest_furniture": dest_furniture,
                    "navigated_path": navigated_path,
                    "value": value,
                }

                # update destination
                cur_furniture = dest_furniture

                known_navigation_costs.append(new_cost)

            elif action_info["action_type"] == "pickup":

                obj_id = action_info["action_arguments"][0]
                fur_id = action_info["action_arguments"][1]

                action = f"pickup({obj_id})"
                state = f"at({fur_id})"
                value = action_info["action_cost"]

                new_cost = ActionRealCost(action=action, state=state, value=value)

                known_manipulation_costs.append(new_cost)

            elif action_info["action_type"] == "place":

                obj_id = action_info["action_arguments"][0]
                fur_id = action_info["action_arguments"][1]

                action = f"place({obj_id})"
                state = f"at({fur_id})"
                value = action_info["action_cost"]

                new_cost = ActionRealCost(action=action, state=state, value=value)

                known_manipulation_costs.append(new_cost)

        # check duplicate manipulation actions, e.g., pickup the same object on the same furniture
        # if new, directly append
        # if duplicate, take average
        grouped = defaultdict(list)

        for cost in known_manipulation_costs:
            key = (cost.action, cost.state)
            grouped[key].append(cost.value)

        # create new fused list
        known_manipulation_costs = []

        for (action, state), values in grouped.items():
            avg_value = sum(values) / len(values)
            known_manipulation_costs.append(
                ActionRealCost(action=action, state=state, value=avg_value)
            )

        return known_navigation_costs, known_manipulation_costs

    def infer_navigation_costs(self, navigation_costs_to_infer, known_navigation_costs):
        """
        navigation_costs_to_infer = {'action': 'navigate',
                                     'start_furniture': cur_furniture,
                                     'dest_furniture': dest_furniture,
                                     'presumed_path': presumed_path}

        known_navigation_costs = {'action': 'navigate',
                                  'start_furniture': cur_furniture,
                                  'dest_furniture': dest_furniture,
                                  'navigated_path': navigated_path,
                                  'value': value}
        """

        inferred_navigation_costs = copy.deepcopy(navigation_costs_to_infer)

        for cost in inferred_navigation_costs:

            inferred_value = 0

            for known_cost in known_navigation_costs:

                if known_cost["navigated_path"] == []:
                    # skip if the start furniture and destination furniture are the same, i.e., robot does not move
                    continue

                overlap_percentage = self.compute_path_overlap(
                    cost["presumed_path"], known_cost["navigated_path"], visualize=False
                )

                inferred_value += known_cost["value"] * overlap_percentage / 100

            cost["value"] = inferred_value

        return inferred_navigation_costs

    def compute_path_overlap(self, path_1, path_2, visualize):
        """
        Compute symmetric overlap between two paths of 2D points.

        Args:
            path_1 (list of tuple): First path [(x1, y1), (x2, y2), ...]
            path_2 (list of tuple): Second path [(x1, y1), (x2, y2), ...]
            visualize (bool): Whether to plot the paths and overlaps.

        Returns:
            float: Overlap percentage (0-100).
        """

        # Distance threshold to consider points as overlapping, beyond which overlap is 0.
        threshold = 2 * GRID_SIZE

        # Convert to numpy arrays
        path_1_np = np.array(path_1)
        path_2_np = np.array(path_2)

        # Build KD-Trees for fast nearest-neighbor lookup
        tree_1 = cKDTree(path_1_np)
        tree_2 = cKDTree(path_2_np)

        # Check for path_1 → path_2
        dists_1, _ = tree_2.query(path_1_np)
        match_1 = np.sum(dists_1 < threshold)

        # Check for path_2 → path_1
        dists_2, _ = tree_1.query(path_2_np)
        match_2 = np.sum(dists_2 < threshold)

        # Average overlap from both directions
        total_matches = match_1 + match_2
        total_points = len(path_1) + len(path_2)
        overlap_percentage = (total_matches / total_points) * 100

        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(*zip(*path_1), "o-", label="Path 1", color="tab:blue")
            plt.plot(*zip(*path_2), "s--", label="Path 2", color="tab:orange")

            # Draw dotted lines between close points from both sides
            for i, pt in enumerate(path_1_np):
                dist, idx = tree_2.query(pt)
                if dist < threshold:
                    neighbor = path_2_np[idx]
                    plt.plot(
                        [pt[0], neighbor[0]], [pt[1], neighbor[1]], "k:", alpha=0.5
                    )

            for i, pt in enumerate(path_2_np):
                dist, idx = tree_1.query(pt)
                if dist < threshold:
                    neighbor = path_1_np[idx]
                    plt.plot(
                        [pt[0], neighbor[0]], [pt[1], neighbor[1]], "k:", alpha=0.5
                    )

            plt.title(f"Path Overlap: {overlap_percentage:.2f}%")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.show()

        return overlap_percentage

    def navigate_old(self, goal_location):
        # navigate the robot to a goal location based on the grid map
        # this old version only replans once

        free_grids = utils.thor_reachable_positions(self.controller)

        # make sure the goal location is a free grid
        valid_goal_location = utils.find_closest_free_grid(goal_location, free_grids)

        # plan a list of free grids towards the goal location
        path = self.a_star(valid_goal_location, free_grids)

        # convert the path to a list of navigation steps (move_ahead, turn_left, turn_right)
        if path is None:
            input("Error! No path found")
        else:
            steps = self.convert_path_to_steps(path)

        collided = False

        collision_counts = 0

        start_time = time.time()

        navigation_steps = 0

        for step in steps:

            collided = self.go(step)

            navigation_steps += 1

            if collided:
                # use the refined grid map to replan when the robot collides with something
                print("Attention! The robot collided with object during navigation.")
                print("Replanning path...")

                free_grids = utils.read_high_resolution_gridmap(self.scene_id)

                # make sure the goal location is a free grid
                valid_goal_location = utils.find_closest_free_grid(
                    goal_location, free_grids
                )

                # replan a new path
                new_path = self.a_star(valid_goal_location, free_grids)

                self.plot_path(free_grids, new_path, valid_goal_location)

                # convert the path to a list of steps (move_ahead, turn_left, turn_right)
                if new_path is None:
                    input("Error! No path found")
                else:
                    new_steps = self.convert_path_to_steps(new_path)

                break

        # execute the new plan if the robot collided with something
        if collided:

            collision_counts += 1

            for step in new_steps:

                self.go(step)

                navigation_steps += 1

        navigation_time = time.time() - start_time

        navigated_distance = navigation_steps * GRID_SIZE

        return collision_counts, navigation_time, navigated_distance
