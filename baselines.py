import os
import random
import numpy as np
import utils

import llm_planning
from llm_planning import LLMPlanner
from motion_planning import MotionPlanner


def moma_llm(
    human_commands,
    scene_graph,
    config,
    llm_planner: LLMPlanner,
    motion_planner: MotionPlanner,
):
    # Baseline algorithm MoMa-LLM
    # MoMa-LLM implements closed-loop LLM planning with replanning,
    # but the replanning process is random without considering action costs.

    mission_metric = {}
    mission_info = {}

    for command in human_commands:

        print("########################################")
        print(f"Trying to fulfill human command '{command}'")
        print("########################################")

        valid = False
        reason = None

        while not valid:

            task_plan = llm_planner.naive_plan(command, scene_graph, reason)

            # append the invalidation reason to LLM prompt to generate a new and valid task plan
            valid, reason = llm_planner.validate_plan(task_plan, scene_graph)

        print("Valid task plan! Now motion planner executing it...")

        plan_info = []
        failed_actions = []

        for ii in range(len(task_plan)):

            action = task_plan[ii]

            action_info = motion_planner.execute(action)

            plan_info.append(action_info)

            # Since MoMa-LLM enables the robot to open doors while navigation,
            # we assume it does not need to replan 'navigate' actions, i.e., all of them can be successfully executed.
            # But of course, we still acount for posssible collision during the navigation process.
            # Also, this is for fair comparison in terms of performance comparison.

            if (action_info["action_type"] == "pickup") and (
                not action_info["success"]
            ):

                obj_id, fur_id = action.argument.split(",")
                obj_id = obj_id.strip()

                original_type = obj_id.split("|")[0]
                original_type = original_type.strip()

                if utils.have_objects_of_the_same_type(scene_graph, obj_id):
                    print(f"Action {action} failed, LLM replanning...")
                else:
                    print(
                        f"Action {action} failed, no need for LLM to replan since there is only one object of this type..."
                    )
                    continue

                failed_actions.append(action)

                # MoMa-LLM naively replans if the pickup action fails, i.e., ask LLM to randomly select a new action
                # because MoMa-LLM assumes all the manipulation actions have the same cost.

                num_retries = 0
                max_retries = 5

                while (not action_info["success"]) and (num_retries < max_retries):

                    while (
                        True
                    ):  # make sure the LLM replans an object of the original type

                        pickup = llm_planner.naive_replan(
                            failed_actions, scene_graph, original_type
                        )

                        obj_id, fur_id = pickup.argument.split(",")
                        obj_id = obj_id.strip()
                        obj_type = obj_id.split("|")[0]
                        obj_type = obj_type.strip()

                        if original_type == obj_type:
                            break
                        else:
                            print(
                                f"LLM failed to select an object of the same type {original_type}, retrying..."
                            )

                    obj_id, fur_id = pickup.argument.split(",")
                    obj_id = obj_id.strip()
                    fur_id = fur_id.strip()

                    # first navigate to the furniture
                    room_id = utils.get_room_by_furniture(scene_graph, fur_id)
                    nav = llm_planning.Action(
                        name="navigate", argument=f"{fur_id}, {room_id}"
                    )

                    nav_info = motion_planner.execute(nav)

                    plan_info.append(nav_info)

                    # then try to pick up the object
                    action_info = motion_planner.execute(pickup)

                    plan_info.append(action_info)

                    num_retries += 1

                    if not action_info["success"]:
                        failed_actions.append(pickup)

                # robot successfully pick up an object after replanning
                # we change the corresponding 'place' action so that the robot will place it on the correct furniture
                if action_info["success"]:

                    next_nav = task_plan[ii + 1]
                    dest_fur, _ = next_nav.argument.split(",")
                    dest_fur = dest_fur.strip()

                    picked_obj = action_info["action_arguments"][0]

                    # change the corresponding 'place' action
                    task_plan[ii + 2] = llm_planning.Action(
                        name="place", argument=f"{picked_obj}, {dest_fur}"
                    )

            # update scene graph
            scene_graph = motion_planner.update_scene_graph(action_info, scene_graph)

        # get plan metric
        plan_metric = utils.get_plan_metric(plan_info)

        # get mission metric
        mission_metric[command] = plan_metric

        utils.plot_mission_metric(mission_metric, config)

        mission_info[command] = plan_info

    return mission_info


def sayplan(
    human_commands,
    scene_graph,
    config,
    llm_planner: LLMPlanner,
    motion_planner: MotionPlanner,
):
    # Baseline algorithm SayPlan
    # SayPlan implements open-loop LLM planning without any replanning.
    # It assumes every action can be perfectly executed by a black-box motion planner.

    mission_metric = {}
    mission_info = {}

    for command in human_commands:

        print("########################################")
        print(f"Trying to fulfill human command '{command}'")
        print("########################################")

        valid = False
        reason = None

        while not valid:

            # open-loop LLM planning
            task_plan = llm_planner.naive_plan(command, scene_graph, reason)

            # append the invalidation reason to LLM prompt to generate a new and valid task plan
            valid, reason = llm_planner.validate_plan(task_plan, scene_graph)

        print("Valid task plan! Now motion planner executing it...")

        plan_info = []

        for action in task_plan:

            # SayPlan does not even replan. It does not consider success or failure of the action.
            action_info = motion_planner.execute(action)

            plan_info.append(action_info)

            # update scene graph
            scene_graph = motion_planner.update_scene_graph(action_info, scene_graph)

        # get plan metric
        plan_metric = utils.get_plan_metric(plan_info)

        # get mission metric
        mission_metric[command] = plan_metric

        utils.plot_mission_metric(mission_metric, config)

        mission_info[command] = plan_info

    return mission_info
