from llm_planning import LLMPlanner, ActionCost
import utils
from motion_planning import MotionPlanner


class InterleavedPlanner:
    # Our interleaved planner enables that llm planner and motion planner interact with each other

    def __init__(
        self, config, controller, llm_planner: LLMPlanner, motion_planner: MotionPlanner
    ):

        self.config = config
        self.controller = controller
        self.llm_planner = llm_planner
        self.motion_planner = motion_planner

        self.SCORE_TO_COST = {"hard": 20, "medium": 10, "easy": 5, "unknown": None}

    def interleaved_planning_by_prompt(self, human_commands, scene_graph):
        # our interleaved plan algorithm considers action costs for more optimal planning
        # This 'prompt' version generates multiple task plan candidates through Text Prompts

        mission_metric = {}
        mission_info = {}

        known_navigation_costs = []
        known_manipulation_costs = []

        for command in human_commands:

            print("########################################")
            print(f"Fulfilling human command '{command}'")
            print("########################################")

            # interleave llm planning with action costs from motion planning
            plan_candidates = self.llm_planner.generate_plan_candidates(
                command, scene_graph, number_of_candidates=3
            )

            plans_costs = []

            for ii in range(len(plan_candidates)):

                task_plan = plan_candidates[ii].task_plan

                while True:  # make sure the plan candidate is valid

                    valid, reason = self.llm_planner.validate_plan(
                        task_plan, scene_graph
                    )

                    if valid:
                        break
                    else:
                        # append the invalidation reason and re-generate a valid task plan candidate
                        task_plan = self.llm_planner.replan_candidate(
                            task_plan, scene_graph, command, reason
                        )

                print(f"Valid task plan candidate {task_plan}")
                print("Estimating the total cost of the task plan candidate...")

                # update the valid task plan in the plan_candidates list
                plan_candidates[ii].task_plan = task_plan

                # The key of our interleaved algorithm is using similarity function
                # to perform LLM planning BnB given the action costs.
                (
                    inferred_navigation_costs,
                    inferred_manipulation_costs,
                ) = self.similarity_function(
                    task_plan, known_navigation_costs, known_manipulation_costs
                )

                # evaluate the task plan by the estimated costs
                plan_cost = self.calculate_plan_cost(
                    inferred_navigation_costs, inferred_manipulation_costs
                )

                plans_costs.append(plan_cost)

            # select the best task plan with the minimum cost
            min_index = plans_costs.index(min(plans_costs))
            best_candidate = plan_candidates[min_index]
            best_task_plan = best_candidate.task_plan

            print(f"\nBest task plan:\n{best_task_plan}\n")
            print(
                "Motion planner executing the best task plan and collecting real costs..."
            )

            utils.plot_command_and_plan(command, best_task_plan)

            plan_info = []

            # execute the best task plan through motion planner
            for action in best_task_plan:

                action_info = self.motion_planner.execute(action)

                plan_info.append(action_info)

                # update scene graph for the next round of planning
                scene_graph = self.motion_planner.update_scene_graph(
                    action_info, scene_graph
                )

            # update known costs by the newly collected action costs
            (
                known_navigation_costs,
                known_manipulation_costs,
            ) = self.motion_planner.update_known_costs(
                plan_info, known_navigation_costs, known_manipulation_costs
            )

            # print('known_navigation_costs', known_navigation_costs)
            # print('known_manipulation_costs', known_manipulation_costs)

            plan_metric = utils.get_plan_metric(plan_info)

            print("plan metric", plan_metric)

            mission_metric[command] = plan_metric

            print("Mission Metric", mission_metric)

            utils.plot_mission_metric(mission_metric, self.config)

            mission_info[command] = plan_info

        return mission_info

    def generate_costs_to_infer(self, task_plan, cur_furniture):
        # convert each action in the task plan into a format ready for cost inference
        # task_plan: [navigate(fur_id), pickup(obj_id), navigate(fur_id), place(obj_id), ...]
        #
        # for navigation actions, we associate presumed paths with furnitures
        # navigate(fur_id) ---> navigate(cur_furniture, dest_furniture) ---> presumed_path from cur_furniture to dest_furniture
        #
        # for manipulation actions, we associate furnitures with objects
        # pickup(obj_id) ---> pickup(object, cur_furniture)
        # place(obj_id) ---> place(object, cur_furniture)

        navigation_costs_to_infer = []

        manipulation_costs_to_infer = []

        for action in task_plan:

            if action.name == "navigate":

                dest_furniture, dest_room = action.argument.split(",")
                dest_furniture = dest_furniture.strip()
                dest_room = dest_room.strip()

                fur_points = self.motion_planner.sample_points_around_furniture(
                    dest_furniture
                )
                fur_loc = tuple(fur_points[0])

                # pre-plan a path and presume the robot will execute it
                free_grids = utils.thor_reachable_positions(self.controller)
                valid_goal_location = utils.find_closest_free_grid(fur_loc, free_grids)
                presumed_path = self.motion_planner.a_star(
                    valid_goal_location, free_grids
                )

                navigation_cost = {
                    "action": "navigate",
                    "start_furniture": cur_furniture,
                    "dest_furniture": dest_furniture,
                    "presumed_path": presumed_path,
                }

                navigation_costs_to_infer.append(navigation_cost)

                # update destination
                cur_furniture = dest_furniture

            elif action.name == "pickup" or action.name == "place":

                cur_object, cur_furniture = action.argument.split(",")
                cur_object = cur_object.strip()
                cur_furniture = cur_furniture.strip()

                # convert the action to ActionCost format
                pickup_or_place_cost = ActionCost(
                    action=f"{action.name}({cur_object})",
                    state=f"at({cur_furniture})",
                    score="unknown",
                )

                manipulation_costs_to_infer.append(pickup_or_place_cost)

        return navigation_costs_to_infer, manipulation_costs_to_infer

    def similarity_function(
        self, task_plan, known_navigation_costs, known_manipulation_costs
    ):
        # This function is used to infer the unknown action costs
        # For manipulation actions, we use the LLM similarity function to infer the costs
        # For navigation actions, we use the distance between the current and target locations

        cur_room, cur_furniture = utils.get_cur_room_furniture(
            self.config["scene_id"], self.controller
        )

        (
            navigation_costs_to_infer,
            manipulation_costs_to_infer,
        ) = self.generate_costs_to_infer(task_plan, cur_furniture)

        # For navigation actions, which do not hold too much semantic information for LLM to infer,
        # we quantitatively compute the ovelapping percentage between two paths
        # to infer unknown navigation action costs based on the known costs
        inferred_navigation_costs = self.motion_planner.infer_navigation_costs(
            navigation_costs_to_infer, known_navigation_costs
        )

        # For manipulation actions, which are associated with objects and furnitures
        # we leverage LLM generalized inference to estimate unknown manipulation action costs
        # based on object/furniture semantic similarity
        known_manipulation_costs_text = self.llm_planner.convert_costs_to_text(
            known_manipulation_costs
        )

        inferred_manipulation_costs = self.llm_planner.infer_manipulation_costs(
            manipulation_costs_to_infer, known_manipulation_costs_text
        )

        return inferred_navigation_costs, inferred_manipulation_costs

    def calculate_plan_cost(
        self, inferred_navigation_costs, inferred_manipulation_costs
    ):
        # calculate the total estimated cost of the plan based on the inferred costs

        count = 0
        plan_cost = 0

        for cost in inferred_navigation_costs:

            cost_value = cost["value"]

            if cost_value is not None:
                count += 1
                plan_cost += cost_value

        for cost in inferred_manipulation_costs:

            cost_value = self.SCORE_TO_COST[cost.score]

            if cost_value is not None:
                count += 1
                plan_cost += cost_value

        # normalize the plan cost
        plan_cost = (
            plan_cost
            * count
            / (len(inferred_navigation_costs) + len(inferred_manipulation_costs))
        )

        return round(plan_cost)
