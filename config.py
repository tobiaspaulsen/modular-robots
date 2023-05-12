import platform

if platform.system() == "Darwin":
    BASE_PATH = "/Users/tobias/My Drive/Studier/master/modular_robots"
    UNITY_BUILD_BASE_PATH = "/Users/tobias/My Drive/Studier/master"
    UNITY_BUILD_PATH = f"{UNITY_BUILD_BASE_PATH}/flat"
else:
    # Linux:
    BASE_PATH = "/fp/homes01/u01/ec-tobiasrp/modular_robots"
    UNITY_BUILD_BASE_PATH = "/fp/homes01/u01/ec-tobiasrp/modular_robots/linux_builds"
    UNITY_BUILD_PATH = f"{UNITY_BUILD_BASE_PATH}/flat/flat"

ROBOT_PATH = f"{BASE_PATH}/robot/robot_configurations/four_legged.json"
LOG_PATH = f"{BASE_PATH}/logs"
RESULTS_PATH = f"{BASE_PATH}/results"

EVALUATION_STEPS = 300
WAIT_WHILE_FALLING_STEPS = 12
MAX_FITNESS = 80  # Set fitness to 0 if impossibly high fitness

# Has to be set in unity as well, both continuous actions and max number:
MAX_MODULES_UNITY = 50
MAX_MODULES_PYTHON = 30  # Used in the evolution in python
MIN_CONTROLLER_OUTPUT = -1
MAX_CONTROLLER_OUTPUT = 1

SEED = 12
PYTHON_DELTA_TIME = 0.05
BODY_JOINTS = ["BodyJoint1", "BodyJoint2", "BodyJoint3", "BodyJoint4"]
LIMB_JOINTS = ["LimbJoint1", "LimbJoint2", "LimbJoint3", "LimbJoint4"]
ROTATIONS = [0, 90, 180, 270]

CLEAN_UP_GENOMES = True
MAX_ADD_DEPTH = 6
REPEAT_ADD_PROB = 0.5
