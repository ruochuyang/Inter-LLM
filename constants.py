# -------------------------------------------------------------------------------
# Overall AI2Thor parameters (v2.7.2)
GRID_SIZE = 0.25  # 0.25 (high-resolution); 0.5 (low-resolution)
MOVE_STEP_SIZE = GRID_SIZE

H_ROTATION = 90  # Yaw; body rotation. Only 90 won't stuck
V_ROTATION = 30  # Pitch; camera up and down

H_ANGLES = [i * H_ROTATION for i in range(int(360 / H_ROTATION))]

# Reference v3.3.4
# https://ai2thor.allenai.org/ithor/documentation/navigation/#Teleport-horizon
# Negative camera horizon values correspond to the agent looking up, whereas
# positive horizon values correspond to the agent looking down.
V_ANGLES = [-30, 0, 30, 60]

FOV = 90  # from official doc: The default field of view when agentMode="default" is 90 degrees.

VISIBILITY_DISTANCE = 1.5
INTERACTION_DISTANCE = 1.5  # objects farther away than this cannot be interacted with.
AGENT_MODE = "arm"  # 'default', 'arm'. From official doc: For iTHOR, it is often safest to stick with the 'default' agent.

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

RENDER_DEPTH = True
RENDER_INSTANCE_SEGMENTATION = True

# Need in order to not stuck the agent for sub-90 degree rotation.
# BUT, it actually DOES NOT WORK
CONTINUOUS = True
SNAP_TO_GRID = not CONTINUOUS

# ------------------------------------------------------------------------------
# Ai2thor parameters related to object search
GOAL_DISTANCE = 1.0
MAX_STEPS = 100

# ------------------------------------------------------------------------------
# Ai2Thor spawning

# When enabled, the scene will attempt to randomize all moveable objects outside
# of receptacles in plain view. Use this if you want to avoid objects spawning
# out of view inside closed drawers, cabinets, etc.
FORCE_VISIBLE = False

# Determines if spawned objects will be settled completely static and unmoving,
# or if non-determenistic physics resolve their final position. Setting this to
# False will allow physics to resolve final positions, which can be used to
# spawn an object on a sloped receptacle but have it end up rolling off.
PLACE_STATIONARY = True

# Used to specify how many objects of a certain type will attempt to be
# duplicated somewhere in the scene. It does not guarantee this number of
# duplicated objects, only the number of attempted spawned objects, so this is
# the max it will be. This will only create copies of objects already in the
# scene, so if you request an object which is not in the scene upon reset, it
# will not work.
NUM_DUPLICATES_OF_TYPE = []  # not actually used

# A list of receptacle object types to exclude from valid receptacles that can
# be randomly chosen as a spawn location. An example use case is excluding all
# CounterTop receptacles to allow for a scene configuration that has more free
# space on CounterTops in case you need free space for interaction. Note that
# this will not guarantee all listed receptacles as being completely clear of
# objects, as any objects that failed to reposition will remain in their default
# position, which might have been on the excluded receptacle type. Check the
# Actionable Properties section of the Objects documentation for a full list of
# Receptacle objects.  **Note**: Receptacle objects allow other objects to be
# placed on or in them if the other object can physically fit the receptacle.
EXCLUDED_RECEPTACLES = []  # not actually used


# -------------------------------------------------------------------------------
# Defines what objects the agent is able to interact with, and the corresponding
# actions to interact with those objects.
INTERACTION_PROPERTIES = [
    # can interact with pickupable objects
    (
        "pickupable",
        lambda obj: ["PickupObject"] if not obj["isPickedUp"] else ["DropObject"],
    ),
    # can interact with openable objects
    ("openable", lambda obj: ["OpenObject"] if not obj["isOpen"] else ["CloseObject"]),
]

# Interactions allowed; Note that you need to change thor_interact.js too.
INTERACTIONS = ["PickupObject", "DropObject", "OpenObject", "CloseObject"]

# Defines navigation actions, with parameters
def get_movement_params(step_size, v_rot, h_rot):
    return {
        "MoveAhead": {"moveMagnitude": step_size},
        "LookUp": {"degrees": v_rot},
        "LookDown": {"degrees": v_rot},
        "RotateLeft": {"degrees": h_rot},
        "RotateRight": {"degrees": h_rot},
    }


MOVEMENT_PARAMS = get_movement_params(MOVE_STEP_SIZE, V_ROTATION, H_ROTATION)

MOVEMENTS = list(MOVEMENT_PARAMS.keys())


def get_acceptable_thor_actions():
    return MOVEMENTS + INTERACTIONS


SCATTER_GRANULARITY = GRID_SIZE * 2


# ------------------------------------------------------------------------------
# Logistics of data collection
TRAIN_TIME = 90  # 90 seconds to play around in the home
TEST_TIME = 180  # 180 seconds to search for the object

# Create config defined in this file as a dictionary
def _load_config():
    config = {}
    for k, v in globals().items():
        if k.startswith("__"):
            continue
        if k == "CONFIG":
            continue
        if callable(eval(k)):
            continue
        config[k] = v
    return config


CONFIG = _load_config()
