
# Codes of Interleaved LLM and Motion Planning (2026 UR Submission)

## Prerequisite

1. **General Environment**

   Ubuntu 20.04, Python 3.8

   Simulation Scenes: ProcTHOR train_1, train_8, train_13 https://procthor.allenai.org/

   Robot Entity: ManipulaTHOR https://ai2thor.allenai.org/manipulathor/

2. **Anaconda Environment**

   (1) Install [Anaconda](https://www.anaconda.com/)

   (2) In the main folder, run `conda env create -f conda_env.yml` to install the conda environment

   (3) Run `conda activate ai2thor` to activate the installed conda environment

## Run Algorithms

1. Specify the algorithm, scene_id, and LLM configurations in the `config.yaml` file

2. In the `OPENAI_API_KEY` field of the `config.yaml` file, insert you own key
 
3. In the main folder, run `python -m build_scene_graph` to build the scene graph for the ProcTHOR scene

4. In the main folder, run `python -m main` to evaluate the specified algorithm in the ProcTHOR scene

## Large ProcTHOR Scenes
**scene train_1**
![scene train_1](https://github.com/icrasubmission/Inter-LLM/blob/main/scenes/train_1/train_1.png)

**scene train_8**
![scene train_1](https://github.com/icrasubmission/Inter-LLM/blob/main/scenes/train_8/train_8.png)

**scene train_13**
![scene train_1](https://github.com/icrasubmission/Inter-LLM/blob/main/scenes/train_13/train_13.png)
