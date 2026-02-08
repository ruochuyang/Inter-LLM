import json
import prior

from baselines import moma_llm, sayplan
from llm_planning import LLMPlanner
from llm_model import LLM
import utils
from motion_planning import MotionPlanner
import constants
from interleaved_planning import InterleavedPlanner


if __name__ == "__main__":

    ##############  Intialization  ##############
    config = utils.parse_config("config.yaml")
    scene_id = config["scene_id"]

    dataset = prior.load_dataset("procthor-10k")
    house = dataset["train"][scene_id]
    controller = utils.launch_controller({**constants.CONFIG, **{"scene": house}})

    graph_file = f"./scenes/train_{config['scene_id']}/simple_scene_graph.json"
    with open(graph_file, "r") as f:
        scene_graph = json.load(f)

    fur_geo_file = "scenes/" + "train_" + str(scene_id) + "/furniture_geometry.json"
    with open(fur_geo_file, "r") as f:
        fur_geo = json.load(f)

    ##############  Planner Setup  ##############
    llm = LLM(
        model=config["llm_model"], temperature=config["temperature"], debug=True
    )  # "gpt-4o-mini"
    llm_planner = LLMPlanner(llm)

    motion_planner = MotionPlanner(controller, fur_geo, scene_id)
    motion_planner.reset_arm()

    ##############  Human Commands  ##############
    command_1 = "I need to have the breakfast. Set it up on the dinning table."

    command_2 = "I have a phone meeting in 10 minutes. Set it up in my bedroom."

    command_3 = "I need to read a book for a break. Also, let me check what is going on in USA today."

    human_commands = [command_1, command_2, command_3]

    ##############  Run Algorithm  ##############

    if config["algorithm"] == "Inter-LLM":

        inter_planner = InterleavedPlanner(
            config, controller, llm_planner, motion_planner
        )

        mission_info = inter_planner.interleaved_planning_by_prompt(
            human_commands, scene_graph
        )

    elif config["algorithm"] == "MoMa-LLM":

        mission_info = moma_llm(
            human_commands, scene_graph, config, llm_planner, motion_planner
        )

    elif config["algorithm"] == "SayPlan":

        mission_info = sayplan(
            human_commands, scene_graph, config, llm_planner, motion_planner
        )

    else:
        raise ValueError(f"Unknown algorithm {config['algorithm']}")

    print("\nDone!")
