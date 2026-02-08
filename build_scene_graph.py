import matplotlib.pyplot as plt
import prior
import math
import json

import constants
from utils import is_point_inside_room, launch_controller, thor_object_with_id


def compute_fur_obj(fur_obj_hier, obj_full_info):
    """
    Compute funiture-object hierarchy from a list of objects.
    And classify objects not on any furniture as obstacles.

    Args:
        fur_obj_hier (list): A list of dictionaries containing furniture-object information.
        obj_full_info (dict): A dict of detailed object information

    Returns:
        fur_obj_info: A list of dictionaries with the extracted infomration of funiture-object hierarchy.
        obstacles: A list of dictionaries with obstacle information
    """
    fur_obj_info = []

    obstacles = {}

    for candidate in fur_obj_hier:

        can_id = candidate["id"]

        # Furnitures should have objects on them
        if "children" in candidate:

            # extract furniture
            extracted_fur = {
                "fur_id": can_id,
                "position": candidate["position"],
                "fur_type": obj_full_info[can_id]["objectType"],
            }

            _, corners = get_furniture_geometry(extracted_fur["fur_id"])

            extracted_fur["corners"] = corners

            # extract objects on the funiture
            extracted_objects = []

            for child in candidate["children"]:

                obj_id = child["id"]

                extracted_obj = {
                    "obj_id": obj_id,
                    "position": child["position"],
                    "obj_type": obj_full_info[obj_id]["objectType"],
                }

                extracted_objects.append(extracted_obj)

            extracted_fur["objects"] = extracted_objects

            fur_obj_info.append(extracted_fur)

        # Objects not on any furniture are classified as obstacles like "GarbageBag_21_1", "Chair_223_1"
        else:

            extracted_obstacle = {
                "position": candidate["position"],
                "obs_type": obj_full_info[can_id]["objectType"],
            }

            obstacles[can_id] = extracted_obstacle

    return fur_obj_info, obstacles


def process_room(rooms):

    processed_rooms = {}

    for room in rooms:

        id = room["id"]
        room_type = room["roomType"]
        room_polygon = room["floorPolygon"]

        processed_rooms[id] = {
            "room_type": room_type,
            "room_polygon": room_polygon,
            "furnitures": [],
        }

    return processed_rooms


def get_object_full_info():

    obj_full_info = {}

    for obj in controller.last_event.metadata["objects"]:

        obj_id = obj["objectId"]

        position = obj["position"]
        pickupable = obj["pickupable"]
        objectType = obj["objectType"]

        obj_full_info[obj_id] = {
            "position": position,
            "pickupable": pickupable,
            "objectType": objectType,
        }

    return obj_full_info


def sort_corners_clockwise(corners):
    # Sort four corner points in a clockwise order

    # Step 1: Compute center of the four points
    cx = sum(x for x, z in corners) / 4
    cz = sum(z for x, z in corners) / 4

    # Step 2: Sort by angle from the center (clockwise)
    sorted_corners = sorted(
        corners, key=lambda p: math.atan2(p[1] - cz, p[0] - cx), reverse=True
    )

    return sorted_corners


def get_furniture_geometry(furniture_id):
    # Input: "id" of furniture, e.g., "Bed|6|0|0"
    #
    # Output:
    # center: {'x': x, 'y': y, 'z': z}
    # corners_2D: [[x1, z1], [x2, z2], [x3, z3], [x4, z4]]

    furniture = thor_object_with_id(controller, furniture_id)

    if furniture is None:
        input(f"Error! Furniture with ID {furniture_id} not found.")

    center = furniture["axisAlignedBoundingBox"]["center"]

    corners_3D = furniture["axisAlignedBoundingBox"]["cornerPoints"]

    corners_2D = [
        [corners_3D[0][0], corners_3D[0][2]],
        [corners_3D[1][0], corners_3D[1][2]],
        [corners_3D[4][0], corners_3D[4][2]],
        [corners_3D[5][0], corners_3D[5][2]],
    ]

    corners_2D = sort_corners_clockwise(corners_2D)

    return center, corners_2D


def compute_scene_graph(house):
    # convert a house into 1) a hierarchical scene graph of room/furniture/object; 2) individual obstacles

    rooms = house["rooms"]
    rooms = process_room(rooms)

    fur_obj_hier = house["objects"]

    obj_full_info = get_object_full_info()

    fur_obj_info, obstacles = compute_fur_obj(fur_obj_hier, obj_full_info)

    for fur in fur_obj_info:

        fur_position = fur["position"]

        for room_id, room in rooms.items():

            room_polygon = room["room_polygon"]

            if is_point_inside_room(fur_position, room_polygon):

                room["furnitures"].append(fur)

                print(fur["fur_id"], "is inside", room_id, "\n")

                break

    return rooms, obstacles


def simplify_scene_graph(full_scene_graph):

    simple_scene_graph = {}

    for room_id, room in full_scene_graph.items():

        simple_scene_graph[room_id] = {"room_type": room["room_type"], "furnitures": []}

        furnitures = room["furnitures"]

        for fur in furnitures:

            fur_tmp = {"fur_id": fur["fur_id"], "fur_type": fur["fur_type"]}

            if "objects" in fur:

                objects_tmp = []

                for obj in fur["objects"]:

                    obj_tmp = {"obj_id": obj["obj_id"], "obj_type": obj["obj_type"]}

                    objects_tmp.append(obj_tmp)

                fur_tmp["objects"] = objects_tmp

            simple_scene_graph[room_id]["furnitures"].append(fur_tmp)

    return simple_scene_graph


if __name__ == "__main__":

    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][1]
    controller = launch_controller({**constants.CONFIG, **{"scene": house}})

    full_scene_graph, obstacles = compute_scene_graph(house)

    with open("full_scene_graph.json", "w") as json_file:
        json.dump(full_scene_graph, json_file, indent=4)

    with open("obstacles.json", "w") as json_file:
        json.dump(obstacles, json_file, indent=4)

    simple_scene_graph = simplify_scene_graph(full_scene_graph)

    with open("simple_scene_graph.json", "w") as json_file:
        json.dump(simple_scene_graph, json_file, indent=4)
