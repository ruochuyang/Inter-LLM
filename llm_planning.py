from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, NamedTuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import distance_matrix
from typing import List
from pydantic import BaseModel
import copy
import heapq

from llm_model import LLM, Conversation
import utils
from utils import has_room_furniture_object


class ActionCost(BaseModel):
    action: str  # 'pickup(RemoteControl|surface|3|18)'
    state: str  # 'at(Sofa|3|3)'
    score: str  # 'easy', 'medium', 'hard', or 'unknown'


class InferredCosts(BaseModel):
    inferred_costs: List[ActionCost]


class Action(BaseModel):
    name: str  # 'pickup'
    argument: str  # 'Newspaper|surface|3|17, Sofa|3|3'


class TaskPlan(BaseModel):
    task_plan: List[Action]  # a list of actions


class TaskPlans(BaseModel):
    task_plans: List[TaskPlan]  # a list of task plans


COST_TO_SCORE = {"hard": [15, np.inf], "medium": [5, 15], "easy": [0, 5]}


class LLMPlanner:
    def __init__(self, llm: LLM):

        self.llm = llm

        self.high_level_actions = {
            "navigate": [
                "(fur_id, room_id)",
                "navigate to this furniture in this room.",
            ],
            "pickup": [
                "(obj_id, fur_id)",
                "pick up this object from this furniture and hold it in your hand.",
            ],
            "place": ["(obj_id, fur_id)", "place this object on this furniture."],
        }

    def generate_plan_candidates(
        self, human_command, scene_graph, number_of_candidates
    ):
        # Our interleaved plan algorithm considers action costs for more optimal planning
        # This 'prompt' version generates multiple task plan candidates through pure prompts

        # create the LLM prompt
        system_prompt = f"You are a household robot in a big house with multiple rooms, which is represented as a scene graph.\n"

        system_prompt += f"Here is the scene graph: {scene_graph}\n"

        system_prompt += (
            f"Your task is to fullfill this human command: {human_command}\n"
        )

        system_prompt += "You have the following actions to use:\n"

        for i, (name, description) in enumerate(self.high_level_actions.items()):

            system_prompt += f"""action_{i+1}: {name}, arguments: {description[0]}, effect: {description[1]}\n"""

        system_prompt += f"The user needs three objects.\
            You need to move to different rooms and furnitures, pick up different objects, and place them on the correct furniture.\
            You can only hold one object in your hand anytime.\n"

        system_prompt += "Reason about where you could find the objects of interest and what actions you need to execute to get them.\n"

        prompt = f"Give me {number_of_candidates} possible task plans to fulfill the human command. Each task plan should be different.\n"

        prompt += "Each task plan is a sequence of actions. For each action in each task plan, strictly follow the format of action arguments."

        prompt += "The arguments of 'navigate' action are (fur_id, room_id). The arguments of 'pickup' action are (obj_id, fur_id). The arguments of 'place' action are (obj_id, fur_id).\n"

        conversation = Conversation(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

        # query the LLM to generate a list of task plans
        response = self.llm.send_query(
            conversation=conversation, response_format=TaskPlans
        )

        task_plans = response.task_plans

        return task_plans

    def replan_candidate(self, invalid_task_plan, scene_graph, human_command, reason):
        # ask LLM to replan a new valid task plan given the previous invalid task plan and the invalidation reason

        system_prompt = f"You are a household robot in a big house with multiple rooms, which is represented as a scene graph.\n"

        system_prompt += f"Here is the scene graph: {scene_graph}\n"

        system_prompt += (
            f"Your task is to fullfill this human command: {human_command}\n"
        )

        system_prompt += "You have the following actions to use:\n"

        for i, (name, description) in enumerate(self.high_level_actions.items()):

            system_prompt += f"""action_{i+1}: {name}, arguments: {description[0]}, effect: {description[1]}\n"""

        prompt = f"Now you have an invalid task plan {invalid_task_plan}.\n"

        prompt += f"The reason why the task plan is invalid is because: {reason}.\n"

        prompt += "Modify this task plan to make it valid. Also make sure it can fulfill the human command.\n"

        prompt += "Strictly follow the format of action arguments. The arguments of 'navigate' action are (fur_id, room_id). The arguments of 'pickup' action are (obj_id, fur_id). The arguments of 'place' action are (obj_id, fur_id).\n"

        conversation = Conversation(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

        # query the LLM to generate a task plan given the prompt
        response = self.llm.send_query(
            conversation=conversation, response_format=TaskPlan
        )

        # return a list of Action
        task_plan = response.task_plan

        return task_plan

    def naive_plan(self, human_command, scene_graph, reason):
        # ask LLM to generate one single task plan without considering action costs

        # create the LLM prompt
        system_prompt = f"You are a household robot in a big house with multiple rooms, which is represented as a scene graph.\n"

        system_prompt += f"Here is the scene graph: {scene_graph}\n"

        system_prompt += (
            f"Your task is to fullfill this human command: {human_command}\n"
        )

        system_prompt += "You have the following actions to use:\n"

        for i, (name, description) in enumerate(self.high_level_actions.items()):

            system_prompt += f"""action_{i+1}: {name}, arguments: {description[0]}, effect: {description[1]}\n"""

        system_prompt += f"The user needs three objects.\
            You need to move to different rooms and furnitures, pick up different objects, and place them on the correct furniture.\
            You can only hold one object in your hand anytime.\n"

        system_prompt += "Reason about where you could find the objects of interest and what actions you need to execute to get there.\n"

        prompt = "Question: what is a good task plan to fulfill the human command?\n"

        prompt += "The task plan is a sequence of actions and you can only include actions I provide with you.\n"

        prompt += "Strictly follow the format of action arguments. The arguments of 'navigate' action are (fur_id, room_id). The arguments of 'pickup' action are (obj_id, fur_id). The arguments of 'place' action are (obj_id, fur_id).\n"

        # reason: the reason why the previous task plan is invalid, e.g., "pick up an object from a wrong furniture"
        if reason is not None:
            prompt += f"Previous task plan is invalid because: {reason}. Update your task plan to make it valid.\n"

        conversation = Conversation(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

        # query the LLM to generate a task plan given the prompt
        response = self.llm.send_query(
            conversation=conversation, response_format=TaskPlan
        )

        # return a list of Action
        task_plan = response.task_plan

        return task_plan

    def naive_replan(self, failed_actions, scene_graph, original_type):
        # LLM naively replan if the action fails
        # ask the LLM to try a new action without considering costs, since MoMa-LLM sets all action costs as a fixed value

        system_prompt = "You are a robot in a big house with multiple rooms, which is represented as a scene graph."
        system_prompt += f"Here is the scene graph: {scene_graph}\n"
        system_prompt += "You have a pickup action to use. The format is pickup(obj_id, fur_id), i.e., pick up this object from this furniture\n"

        # MoMa-LLM only encodes 'success or failure' into the action history
        prompt = f"Now you are trying to pick up an object but previous actions {failed_actions} all failed. Luckily, in the scene graph, some objects are of the same type."
        prompt += f"You can go to another furniture and pick up another object. Make sure the object is of the type {original_type}."
        prompt += f"You should also avoid the failed actions to optimize your choice."

        prompt += "Question: select another pickup action you think will succeed based on the scene graph.\n"
        prompt += f"Remember: the object you try to pick up should be of the type {original_type}."

        conversation = Conversation(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

        # LLM replans a new action
        new_action = self.llm.send_query(
            conversation=conversation, response_format=Action
        )

        return new_action

    def validate_plan(self, task_plan, scene_graph):
        # check if the plan is valid
        # 1. check if each action follows the precondition and effect
        # 2. check if the room/furniture/object is in the scene graph

        last_navigated = None  # store the last navigated furniture
        last_action = None  # store the last performed action
        carried_object = None  # store the object currently being carried

        for action in task_plan:

            if action.name == "navigate":

                furniture, room = action.argument.split(",")
                furniture = furniture.strip()
                room = room.strip()

                if not has_room_furniture_object(scene_graph, furniture, "furniture"):

                    reason = f"Invalid task plan! The funiture {furniture} is not in the scene graph."
                    print(reason)

                    return False, reason

                if not has_room_furniture_object(scene_graph, room, "room"):

                    reason = (
                        f"Invalid task plan! The room {room} is not in the scene graph."
                    )
                    print(reason)

                    return False, reason

                last_navigated = furniture

            elif action.name == "pickup":

                if last_action == "pickup":

                    reason = (
                        "Invalid task plan! Cannot perform 'pickup' twice in a row."
                    )
                    print(reason)

                    return False, reason

                if carried_object is not None:

                    reason = f"Invalid task plan! Already carrying {carried_object}, cannot pick up another object."
                    print(reason)

                    return False, reason

                obj, furniture = action.argument.split(", ")
                obj = obj.strip()
                furniture = furniture.strip()

                if not has_room_furniture_object(scene_graph, obj, "object"):

                    reason = f"Invalid task plan! The object {obj} is not in the scene graph."
                    print(reason)

                    return False, reason

                if not has_room_furniture_object(scene_graph, furniture, "furniture"):

                    reason = (
                        f"Invalid task plan! The room {room} is not in the scene graph."
                    )
                    print(reason)

                    return False, reason

                if last_navigated != furniture:

                    reason = f"Invalid task plan! {obj} is picked from {furniture}, but last navigated to {last_navigated}."
                    print(reason)

                    return False, reason

                # store the picked-up object
                carried_object = obj

            elif action.name == "place":

                if last_action == "place":

                    reason = "Invalid task plan! Cannot perform 'place' twice in a row."
                    print(reason)

                    return False, reason

                obj, furniture = action.argument.split(", ")
                obj = obj.strip()
                furniture = furniture.strip()

                if not has_room_furniture_object(scene_graph, obj, "object"):

                    reason = f"Invalid task plan! The object {obj} is not in the scene graph."
                    print(reason)

                    return False, reason

                if not has_room_furniture_object(scene_graph, furniture, "furniture"):

                    reason = (
                        f"Invalid task plan! The room {room} is not in the scene graph."
                    )
                    print(reason)

                    return False, reason

                if last_navigated != furniture:

                    reason = f"Invalid task plan! {obj} is placed on {furniture}, but last navigated to {last_navigated}."
                    print(reason)

                    return False, reason

                if carried_object != obj:

                    reason = f"Invalid task plan! {obj} has not been picked up or does not match the currently carried object."
                    print(reason)

                    return False, reason

                # set the arm to empty
                carried_object = None

            # update last action
            last_action = action.name

        return True, None

    def infer_manipulation_costs(self, costs_to_infer, known_costs_text):
        # This function enables us to go one step further beyond MoMa-LLM.
        # We do not naively append the binary execution feedback (success or failure) to the LLM prompt.
        # This can just tell the LLM this action is bad and you need to replan,
        # but does not tell the LLM what is actually the best action to take.​
        # Instead, we systematically collect more nuanced costs (easy, medium, hard) and upload them to the LLM
        # so that the LLM performs a comprehensive BnB for its search tree and selects the best plan.
        
        system_prompt = f"You are a robot in a big house with multiple rooms. Your task is to fullfill a series of commands for a human user. Basically, you need to navigate to different rooms and furnitures, pick up different objects, and place them on the correct furniture. Each navigate, pickup, place action has a cost, i.e., hard or easy to execute the action. \n"

        system_prompt += "I give you some known costs of performing some actions at some states.\
            The feedback is in the format of (action, state, cost). \
            Try your best to reasonably infer the unknown costs.  \
            You can think about semantic attributes like location, category, usage of the object and the associated furniture. \
            If you think it is unreasonable to infer a cost, you just leave its score as 'unknown'.\
            For example, if you already know the cost of picking up a remote control at a TV stand is hard, you can infer the cost of picking up a credit card at the same TV stand is even harder, since the credit card is smaller and thinner than the remote control."

        # cost = (action, state, score)
        # MoMa-LLM naively sets all the actions as a fixed cost 30.
        # We consider nuanced scores to reflect realistic action exeuction in the complicated world.

        prompt = "Here are the known costs."

        for ii in range(len(known_costs_text)):
            prompt += f"cost_{ii+1}: {known_costs_text[ii].action}, {known_costs_text[ii].state}, {known_costs_text[ii].score}; "

        prompt += f"\nQuestion: infer the unknown cost of {costs_to_infer} given the known costs\n"

        prompt += "Remember: if you think it is unreasonable to infer a cost, you just leave its score as 'unknown'.\n"

        conversation = Conversation(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

        response = self.llm.send_query(
            conversation=conversation, response_format=InferredCosts
        )

        inferred_costs = response.inferred_costs

        return inferred_costs

    def convert_costs_to_text(self, known_manipulation_costs):
        # convert the known costs to text format for LLM input
        # known_manipulation_costs: a list of ActionRealCost objects with numerical action costs
        # known_manipulation_costs_text: a list of ActionCost objects in text format

        known_manipulation_costs_text = []

        for cost in known_manipulation_costs:

            action = cost.action
            state = cost.state

            for key, val_range in COST_TO_SCORE.items():
                if val_range[0] <= cost.value < val_range[1]:
                    score = key
                    break

            known_manipulation_costs_text.append(
                ActionCost(action=action, state=state, score=score)
            )

        return known_manipulation_costs_text
