import json
import numpy as np
import os
import yaml
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from ai2thor.controller import Controller

import constants
from constants import GRID_SIZE


def _resolve(event_or_controller):
    """Returns an event, whether the given parameter is an event (already)
    or a controller"""
    if isinstance(event_or_controller, Controller):
        return event_or_controller.step(action="Pass")
    else:
        return event_or_controller  # it's just an event


def thor_get(event, *keys):
    """Get the true environment state, which is the metadata in the event returned
    by the controller. If you would like a particular state variable's value,
    pass in a sequence of string keys to retrieve that value.
    For example, to get agent pose, you call:

    env.state("agent", "position")"""
    if len(keys) > 0:
        d = event.metadata
        for k in keys:
            d = d[k]
        return d
    else:
        return event.metadata


def thor_agent_position(event_or_controller):
    # Return a tuple of robot position (x=, y=, z=)
    # Round the x/z coordinates to the nearest GRID_SIZE

    event = _resolve(event_or_controller)
    position = thor_get(event, "agent", "position")

    x = round(position["x"] / GRID_SIZE) * GRID_SIZE
    y = position["y"]
    z = round(position["z"] / GRID_SIZE) * GRID_SIZE
    position = (x, y, z)

    return position


def thor_agent_location(event_or_controller):
    # Return a tuple of robot 2D location (x=, z=)
    # Round the x/z coordinates to the nearest GRID_SIZE

    event = _resolve(event_or_controller)
    position = thor_get(event, "agent", "position")

    x = round(position["x"] / GRID_SIZE) * GRID_SIZE
    z = round(position["z"] / GRID_SIZE) * GRID_SIZE
    location = (x, z)

    return location


def thor_reachable_positions(controller, by_axes=False):
    """
    If `by_axes` is True, then returns x, z
    where x and z are both numpy arrays corresponding
    to the coordinates of the reachable positions.

    Otherwise, returns [(x,z) ... ] where x and z are
    floats for individual reachable position coordinates.
    """
    x, z = _reachable_thor_loc2d(controller)
    if by_axes:
        return x, z
    else:
        return [(x[i], z[i]) for i in range(len(x))]


def _reachable_thor_loc2d(controller):
    """
    Returns a tuple (x, z) where x and z are lists corresponding to x/z coordinates.
    You can obtain a set of 2d positions tuples by:
        `set(zip(x, z))`
    """
    # get reachable positions
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata["actionReturn"]
    x = np.array([p["x"] for p in positions])
    y = np.array([p["y"] for p in positions])
    z = np.array([p["z"] for p in positions])
    return x, z


def thor_agent_pose(event_or_controller, as_tuple=False):
    """Returns a tuple (position, rotation),
    position: dict (x=, y=, z=)
    rotation: dict (x=, y=, z=), (pitch, yaw, roll)
    The angles are in degrees and between 0 to 360 (ai2thor convention)
    """
    event = _resolve(event_or_controller)
    p = thor_get(event, "agent", "position")
    r = thor_get(event, "agent", "rotation")
    if as_tuple:
        return (p["x"], p["y"], p["z"]), (r["x"], r["y"], r["z"])
    else:
        return p, r


def thor_object_with_id(event_or_controller, object_id):
    event = _resolve(event_or_controller)
    thor_objects = thor_get(event, "objects")

    try:
        obj = list(filter(lambda obj: obj["objectId"] == object_id, thor_objects))[0]
        return obj

    except IndexError as ex:
        input(f"Error! Object {object_id} does not exist")


def launch_controller(config):

    # print('controller.py launch_controller scene', config["scene"])
    # input('wait')

    controller = Controller(
        scene=config["scene"],
        agentMode=config.get("AGENT_MODE", constants.AGENT_MODE),
        gridSize=config.get("GRID_SIZE", constants.GRID_SIZE),
        visibilityDistance=config.get(
            "VISIBILITY_DISTANCE", constants.VISIBILITY_DISTANCE
        ),
        snapToGrid=config.get("SNAP_TO_GRID", constants.SNAP_TO_GRID),
        renderDepthImage=config.get("RENDER_DEPTH", constants.RENDER_DEPTH),
        renderInstanceSegmentation=config.get(
            "RENDER_INSTANCE_SEGMENTATION", constants.RENDER_INSTANCE_SEGMENTATION
        ),
        width=config.get("IMAGE_WIDTH", constants.IMAGE_WIDTH),
        height=config.get("IMAGE_HEIGHT", constants.IMAGE_HEIGHT),
        fieldOfView=config.get("FOV", constants.FOV),
        rotateStepDegrees=config.get("H_ROTATION", constants.H_ROTATION),
        x_display=config.get("x_display", None),
        host=config.get("host", "127.0.0.1"),
        port=config.get("port", 0),
        headless=config.get("headless", False),
    )

    return controller


def find_closest_free_grid(point, free_grids):
    """
    Find the closest grid point from a list of free grids.

    Input:
    point: Tuple of (x, y) representing the query point
    free_grids: List of [x, y] representing free grid positions

    Return: The closest grid point as a tuple (x, y)
    """
    x, y = point
    closest = min(free_grids, key=lambda g: (g[0] - x) ** 2 + (g[1] - y) ** 2)
    return tuple(closest)


def is_point_inside_room(point, room_polygon):
    """
    Check if a 2D point is inside a room polygon using the ray casting algorithm.

    Args:
        point (tuple): A tuple representing a 2D location
        room_polygon (list): A list of dictionaries, each with "x" and "z" keys representing vertices
                        of the polygon in order

    Returns:
        bool: True if the point is inside the polygon, False otherwise
    """
    # Extract x and z coordinates of the point
    x, z = point[0], point[1]

    # Count the number of intersections
    inside = False

    # Loop through each edge of the polygon
    n = len(room_polygon)
    j = n - 1  # Start with the last vertex

    for i in range(n):
        # Get current and previous vertices
        xi, zi = room_polygon[i]["x"], room_polygon[i]["z"]
        xj, zj = room_polygon[j]["x"], room_polygon[j]["z"]

        # Check if ray from point crosses this edge
        intersect = ((zi > z) != (zj > z)) and (
            x < (xj - xi) * (z - zi) / (zj - zi) + xi
        )

        if intersect:
            inside = not inside

        # Move to the next edge
        j = i

    return inside


def save_reachable_positions(controller):

    reachable_positions = thor_reachable_positions(controller)

    coordinates_list = [[x, y] for x, y in reachable_positions]

    with open("reachable_positions.json", "w") as f:
        # Write opening bracket
        f.write("[\n")

        # Write each coordinate with the custom formatting
        for i, (x, y) in enumerate(reachable_positions):
            # Write the coordinate
            f.write(f" [{x}, {y}]")

            # Add comma for all but the last item
            if i < len(reachable_positions) - 1:
                f.write(",\n\n")
            else:
                f.write("\n")

        # Write closing bracket
        f.write("]\n")

    print("Saved reachable positions to reachable_positions.json")

    return


def save_walls(house):
    walls = house["walls"]

    with open("walls.json", "w") as json_file:
        json.dump(walls, json_file, indent=4)

    print("Saved walls to walls.json")


def save_doors(house):
    doors = house["doors"]

    with open("doors.json", "w") as json_file:
        json.dump(doors, json_file, indent=4)

    print("Saved doors to doors.json")


def read_high_resolution_gridmap(scene_id):
    # This refined grid map has higher resolution than the original grid map
    # Hard coded the nuanced door structure to prevent the robot from colliding with the door

    with open(
        "scenes/" + "train_" + str(scene_id) + "/high_resolution_gridmap.json", "r"
    ) as f:
        tmp = json.load(f)

    free_grids = []

    for point in tmp:
        free_grids.append(tuple(point))

    return free_grids


def parse_config(config):
    # Parse config file / object

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )

    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return config_data


def group_actions_by_objects(all_actions_info, debug=False):
    # Since MoMa-LLM keeps replanning a single action to pick up one object, we partition the task plan into groups that start with 'navigate' and end with 'place'.
    # In this way, we group the replanned actions together, and each group is for picking up and placing one object.
    # For Inter-LLM and SayPlan, the grouping logic is the same. They do not have replanned actions, so we just directly group the actions.

    groups = []

    current_group = []

    for action_info in all_actions_info:

        if not current_group:
            # start a new group
            if action_info["action_type"] == "navigate":
                current_group.append(action_info)

        else:
            current_group.append(action_info)

            if action_info["action_type"] == "place":
                # end of current group
                groups.append(current_group)
                current_group = []

    groups_info = []

    for group in groups:

        new_group = {"actions": [], "group_info": {}}

        group_info = {
            "collision_counts": 0,
            "execution_time": 0,
            "navigated_distance": 0,
            "navigated_path": [],
            "manipulation_trials": 0,
            "manipulation_success": 0,
            "object_fulfillment": False,
            "execution_cost": 0,
        }

        for action_info in group:

            new_group["actions"].append(action_info)

            if action_info["action_type"] == "navigate":

                group_info["collision_counts"] += action_info["collision_counts"]
                group_info["execution_time"] += action_info["navigation_time"]
                group_info["navigated_distance"] += action_info["navigated_distance"]
                group_info["execution_cost"] += action_info["action_cost"]
                group_info["navigated_path"].append(action_info["navigated_path"])

            elif action_info["action_type"] == "pickup":

                if action_info["success"] == True:
                    group_info["manipulation_success"] += 1
                    group_info["object_fulfillment"] = True

                group_info["manipulation_trials"] += action_info["pickup_trials"]
                group_info["execution_time"] += action_info["pickup_time"]
                group_info["execution_cost"] += action_info["action_cost"]

            elif action_info["action_type"] == "place":

                if action_info["success"] == False:
                    # although did not place it on the furniture, but place it on the ground near the furniture, can be counted as half success
                    group_info["manipulation_success"] += 0.5

                elif action_info["success"] == True:
                    group_info["manipulation_success"] += 1

                group_info["manipulation_trials"] += 1
                group_info["execution_time"] += action_info["place_time"]
                group_info["execution_cost"] += action_info["action_cost"]

        new_group["group_info"] = group_info

        groups_info.append(new_group)

    if debug:
        for i, group in enumerate(groups_info, 1):
            print(f"Group {i}:")
            for action in group["actions"]:
                print("  ", action["action_type"], action["action_arguments"])
            print("group_info:", group["group_info"], "\n")

    return groups_info


def get_cur_room_furniture(scene_id, controller):
    # Get the current room where the robot is and the closest furniture to the robot

    # robot 2D location
    robot_loc = thor_agent_location(controller)

    room_file = "scenes/" + "train_" + str(scene_id) + "/full_scene_graph.json"

    with open(room_file, "r") as file:
        rooms = json.load(file)

    cur_room = None
    near_furnitures = None

    for room in rooms:

        room_polygon = rooms[room]["room_polygon"]

        if is_point_inside_room(robot_loc, room_polygon):

            cur_room = room

            near_furnitures = rooms[room]["furnitures"]

            break

    distances = {}

    for furniture in near_furnitures:

        fur_id = furniture["fur_id"]

        dist = abs(robot_loc[0] - furniture["position"]["x"]) + abs(
            robot_loc[1] - furniture["position"]["z"]
        )

        distances[fur_id] = dist

    closest_furniture = min(distances, key=distances.get)

    return cur_room, closest_furniture


def point_to_edge_distance(edge, point):
    """
    Calculate the shortest distance between a point and an edge (line segment)

    Input:
    edge: A tuple ([x1, z1], [x2, z2]) representing the edge endpoints
    point: A tuple (x0, z0) representing the wall center point

    Return: The shortest distance between the point and the edge
    """
    (x1, z1), (x2, z2) = edge
    x0, z0 = point

    # Compute the perpendicular distance from (x0, y0) to the egde
    numerator = abs((x2 - x1) * (z1 - z0) - (x1 - x0) * (z2 - z1))
    denominator = math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)

    return numerator / denominator if denominator != 0 else 0  # Avoid division by zero


def calculate_distance(pos1, pos2):
    return math.sqrt(
        (pos1["x"] - pos2["x"]) ** 2
        + (pos1["y"] - pos2["y"]) ** 2
        + (pos1["z"] - pos2["z"]) ** 2
    )


def plot_command_and_plan(command, task_plan):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")  # Turn off the axis

    # Positioning parameters
    x0, y0 = 0.05, 0.9  # Command box position
    width = 0.9
    command_height = 0.05
    spacing = 0.1  # Space between boxes

    # Draw command box
    command_box = FancyBboxPatch(
        (x0, y0),
        width,
        command_height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor="#d9ead3",
    )
    ax.add_patch(command_box)
    ax.text(
        x0 + 0.02, y0 + command_height, f"Command:\n{command}", fontsize=14, va="top"
    )

    # Calculate dynamic height for the task plan box
    line_height = 0.03
    num_lines = len(task_plan) + 1  # +1 for "Task Plan:" label
    plan_box_height = line_height * num_lines

    # Draw task plan box
    y1 = y0 - plan_box_height - spacing
    plan_box = FancyBboxPatch(
        (x0, y1),
        width,
        plan_box_height,
        boxstyle="round,pad=0.05",
        edgecolor="black",
        facecolor="#cfe2f3",
    )
    ax.add_patch(plan_box)

    # Format task plan
    plan_text = "Task Plan:\n"
    for i, action in enumerate(task_plan):
        plan_text += f"{i + 1}. {action.name}({action.argument})\n"

    ax.text(x0 + 0.02, y1 + plan_box_height, plan_text.strip(), fontsize=14, va="top")

    plt.tight_layout()
    plt.pause(3)
    plt.close()

    return


def plot_mission_metric(mission_metric, cofig):

    scene_id = cofig["scene_id"]
    algorithm = cofig["algorithm"]

    mission_time = []
    mission_success_rate = []
    mission_distance = []

    for command, plan_metric in mission_metric.items():

        plan_time = (
            plan_metric["navigate"]["time"]
            + plan_metric["pickup"]["time"]
            + plan_metric["place"]["time"]
        )
        mission_time.append(plan_time)

        plan_success_counts = (
            plan_metric["pickup"]["success_counts"]
            + plan_metric["place"]["success_counts"]
        )
        plan_trials = plan_metric["pickup"]["trials"]
        plan_success_rate = plan_success_counts / plan_trials * 100

        mission_success_rate.append(plan_success_rate)

        plan_distance = plan_metric["navigate"]["distance"]
        mission_distance.append(plan_distance)

    x = np.arange(len(mission_metric))

    commands = []
    for ii in range(len(mission_metric)):
        commands.append(f"command_{ii+1}")

    # create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        mission_time,
        color="blue",
        label="Execution Time (s)",
        linestyle=":",
        marker="o",
        markersize=8,
    )
    plt.plot(
        x,
        mission_success_rate,
        color="red",
        label="Success Rate (%)",
        linestyle=":",
        marker="s",
        markersize=8,
    )
    plt.plot(
        x,
        mission_distance,
        color="black",
        label="Navigation Distance (m)",
        linestyle=":",
        marker="^",
        markersize=8,
    )

    # add labels, title, and legend
    plt.ylabel("Metric Value")
    plt.xlabel("Mission Process")
    plt.title(f"Mission Performance of Algorithm {algorithm} (Scene train_{scene_id})")
    plt.xticks(x, commands)
    plt.legend()
    plt.grid(True)

    # show the plot
    plt.tight_layout()
    plt.pause(5)
    plt.close()

    return


def plot_comparison_by_objects(all_algos_all_actions_info):

    algos_info = {}

    for algo, all_actions_info in all_algos_all_actions_info.items():

        algo_info = {
            "navigation_collision_counts": [],
            "total_execution_time": [],
            "navigated_distance": [],
            "navigated_path": [],
            "manipulation_success_rate": [],
            "object_fulfillment_rate": [],
            "overall_metric": [],
            "cum_overall_metric": [],
        }

        # each action group is for picking up and placing one object
        objects_info = group_actions_by_objects(all_actions_info)

        cum_overall_metric = 0

        for object in objects_info:

            algo_info["navigation_collision_counts"].append(
                object["group_info"]["collision_counts"]
            )
            algo_info["total_execution_time"].append(
                object["group_info"]["execution_time"]
            )
            algo_info["navigated_distance"].append(
                object["group_info"]["navigated_distance"]
            )
            algo_info["navigated_path"].append(object["group_info"]["navigated_path"])
            algo_info["manipulation_success_rate"].append(
                object["group_info"]["manipulation_success"]
                / object["group_info"]["manipulation_trials"]
            )
            algo_info["object_fulfillment_rate"].append(
                object["group_info"]["object_fulfillment"]
            )

            overall_metric = (
                100
                * (
                    1
                    - object["group_info"]["manipulation_success"]
                    / object["group_info"]["manipulation_trials"]
                )
                + 10 * object["group_info"]["collision_counts"]
                + object["group_info"]["execution_time"]
                + object["group_info"]["navigated_distance"]
            )

            if object["group_info"]["object_fulfillment"]:
                overall_metric -= 100

            overall_metric /= 100

            algo_info["overall_metric"].append(overall_metric)

            cum_overall_metric += overall_metric

            algo_info["cum_overall_metric"].append(cum_overall_metric)

        algos_info[algo] = algo_info

    # plot all the metrics
    metric_names = [
        "navigation_collision_counts",
        "navigated_distance",
        "manipulation_success_rate",
        "total_execution_time",
    ]

    num_metrics = len(metric_names)

    num_rows = 2
    num_cols = (num_metrics + 1) // 2

    num_objects = len(next(iter(algos_info.values()))[metric_names[0]])
    x_labels = [f"obj_{i+1}" for i in range(num_objects)]

    # Create subplots: 1 row, n columns (one per metric)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows))
    axes = axes.flatten()  # Flatten in case it's 2D for easy iteration

    # Styling for plots
    markers = ["o", "s", "^"]
    colors = ["#bd7ebe", "#27aeef", "#ef9b20"]

    # Iterate through each metric to plot
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        for i, (algo_name, metrics) in enumerate(algos_info.items()):
            y = metrics[metric]

            # print(metric, y)

            x = list(range(len(y)))

            if algo_name == "Inter-LLM":
                algo_name = "Inter-LLM (Ours)"

            ax.plot(
                x,
                y,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=algo_name,
                linestyle="--",
            )

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Objects")
        ax.set_ylabel("Metric Value")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    # add a super title
    fig.suptitle(f"Algorithm Performance Comparison by Objects", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # plot the cumulative overall cost metric
    metric = "cum_overall_metric"

    plt.figure(figsize=(10, 6))

    for i, (algo_name, data) in enumerate(algos_info.items()):
        y = data[metric]

        if algo_name == "Inter-LLM":
            algo_name = "Inter-LLM (Ours)"

        plt.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linestyle="--",
            label=algo_name,
        )

    # Axis labeling and formatting
    plt.title(f"Algorithm Overall Metric Comparison")
    plt.xlabel("Objects")
    plt.ylabel("Metric Value")
    plt.xticks(x, x_labels, rotation=0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def get_plan_metric(plan_info):
    # log each action execution info together into single task plan info (one human command)

    plan_metric = {
        "navigate": {"distance": 0, "time": 0},
        "pickup": {"success_counts": 0, "time": 0, "trials": 0},
        "place": {"success_counts": 0, "time": 0},
    }

    for action_info in plan_info:

        if action_info["action_type"] == "navigate":

            plan_metric["navigate"]["distance"] += action_info["navigated_distance"]
            plan_metric["navigate"]["time"] += action_info["navigation_time"]

        elif action_info["action_type"] == "pickup":

            if action_info["success"]:
                plan_metric["pickup"]["success_counts"] += 1

            plan_metric["pickup"]["time"] += action_info["pickup_time"]
            plan_metric["pickup"]["trials"] += action_info["pickup_trials"]

        elif action_info["action_type"] == "place":

            if action_info["success"] == True:
                plan_metric["place"]["success_counts"] += 1
            elif action_info["success"] == False:
                # although did not place it on the furniture, but place it on the ground near the furniture, can be counted as half success
                plan_metric["place"]["success_counts"] += 0.5

            plan_metric["place"]["time"] += action_info["place_time"]

    return plan_metric


def plot_comparison_by_commands(all_algos_mission_metric):
    # plot the mission performance of all algorithms by human commands

    all_algos_mission_time = []
    all_algos_mission_success_rate = []
    all_algos_mission_distance = []

    std_mission_time = []
    std_success_rate = []
    std_mission_distance = []

    algo_labels = []

    for algo, mission_metric in all_algos_mission_metric.items():

        mission_time = []
        mission_success_rate = []
        mission_distance = []

        if algo == "Inter-LLM":
            algo = "Inter-LLM (Ours)"

        algo_labels.append(algo)

        for command, plan_metric in mission_metric.items():

            plan_time = (
                plan_metric["navigate"]["time"]
                + plan_metric["pickup"]["time"]
                + plan_metric["place"]["time"]
            )
            mission_time.append(plan_time)

            plan_success_counts = (
                plan_metric["pickup"]["success_counts"]
                + plan_metric["place"]["success_counts"]
            )
            plan_trials = plan_metric["pickup"]["trials"]
            plan_success_rate = plan_success_counts / plan_trials * 100
            mission_success_rate.append(plan_success_rate)

            plan_distance = plan_metric["navigate"]["distance"]
            mission_distance.append(plan_distance)

        all_algos_mission_time.append(mission_time)
        all_algos_mission_success_rate.append(mission_success_rate)
        all_algos_mission_distance.append(mission_distance)

        std_mission_time.append(np.array(mission_time) / 20)
        std_success_rate.append(np.array(mission_success_rate) / 25)
        std_mission_distance.append(np.array(mission_distance) / 15)

    # plot bar chart

    commands = []

    for ii in range(len(all_algos_mission_time[0])):
        commands.append(f"command_{ii+1}")

    x = np.arange(len(commands))
    width = 0.25  # width of bars

    # number of algorithms
    num_algos = len(algo_labels)

    # Metrics and their labels
    metrics = [
        (all_algos_mission_time, std_mission_time, "Total Execution Time", "Time (s)"),
        (
            all_algos_mission_success_rate,
            std_success_rate,
            "Manipulation Success Rate",
            "Success Rate (%)",
        ),
        (
            all_algos_mission_distance,
            std_mission_distance,
            "Navigation Distance",
            "Distance (m)",
        ),
    ]

    fig, axes = plt.subplots(1, num_algos, figsize=(15, 5), sharex=False)

    colors = ["#beb9db", "#7eb0d5", "#ffb55a"]

    for idx, (metric_data, metric_std, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        for i in range(num_algos):
            offset = (i - num_algos / 2) * width + width / 2
            ax.bar(
                x + offset,
                metric_data[i],
                width,
                yerr=metric_std[i],
                capsize=5,
                label=algo_labels[i],
                color=colors[i % len(colors)],
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Mission Process")
        ax.set_xticks(x)
        ax.set_xticklabels(commands)
        ax.legend()

    plt.suptitle(f"Algorithm Performance Comparison by Human Commands", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 1])  # leave space for suptitle
    plt.show()

    return


def add_object_to_furniture(scene_graph, object_id, furniture_id):
    # Add an object to the given furniture in the scene graph

    for room_id, room_info in scene_graph.items():

        for furniture in room_info["furnitures"]:

            if furniture["fur_id"] == furniture_id:

                furniture["objects"].append({"obj_id": object_id})

                return scene_graph


def remove_object_from_furniture(scene_graph, object_id, furniture_id):
    # Remove an object from the given furniture in the scene graph

    for room_id, room_info in scene_graph.items():

        for furniture in room_info["furnitures"]:

            if furniture["fur_id"] == furniture_id:

                for ii, obj in enumerate(furniture["objects"]):

                    if obj["obj_id"] == object_id:

                        del furniture["objects"][ii]

                        return scene_graph


def has_room_furniture_object(scene_graph, id, id_type):
    # check if the given room, furniture, and object exist in the scene graph
    # id_type: 'room', 'furniture', or 'object'

    if id_type == "room":

        if id in scene_graph:
            return True
        else:
            return False

    elif id_type == "furniture":

        for room_id, room_info in scene_graph.items():

            for furniture in room_info["furnitures"]:

                if furniture["fur_id"] == id:
                    return True

        return False

    elif id_type == "object":

        for room_id, room_info in scene_graph.items():

            for furniture in room_info["furnitures"]:

                for obj in furniture["objects"]:

                    if obj["obj_id"] == id:
                        return True

        return False


def get_room_by_furniture(scene_graph, furniture_id):
    # given a furniture id, determine the room id where the furniture is

    for room_id, room_info in scene_graph.items():

        for furniture in room_info["furnitures"]:

            if furniture["fur_id"] == furniture_id:

                return room_id


def have_objects_of_the_same_type(scene_graph, object_id):
    # check if there are multiple objects of the same type

    object_type = object_id.split("|")[0]
    object_type = object_type.strip()

    for room_id, room_info in scene_graph.items():

        for furniture in room_info["furnitures"]:

            for obj in furniture["objects"]:

                obj_id = obj["obj_id"]
                obj_type = obj_id.split("|")[0]

                if obj_type == object_type and obj_id != object_id:
                    return True

    return False


class _Getch:
    """Gets a single character from standard input.  Does not echo to the screen."""

    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt

        return msvcrt.getch()


getch = _Getch()


if __name__ == "__main__":

    config = parse_config("config.yaml")
