import time
import numpy as np
from klampt import WorldModel, vis
from pathlib import Path
import threading


# ------- TODO --------

# forward_kinematics optimization?

# On ESP32 utilize ESP-DSP for
# MAC (Multiply-Accumulate) instructions and 
# Single Instruction Multiple Data (SIMD) optimizations

# HARDCODED TO ESP32:
#
# PRECOMPUTED_TRANSFORMS
# JOINT_LIMITS
# CONTROLLABLE JOINTS
# DT?

# Functions transferred over to ESP32:
# forward_kinematics
# numerical and analytical jacobian
# IK methods (DLS, NEWTONRAPHSON, CCD)

# ======================== ROBOT CONFIGURATION ========================
JOINT_LIMITS = [
    (-2.618, 2.168),  # Joint1 (base_link -> link1)
    (0.0, 3.14),      # Joint2 (link1 -> link2)
    (-2.967, 0),      # Joint3 (link2 -> link3)
    (-1.745, 1.745),  # Joint4 (link3 -> link4)
    (-1.22, 1.22),    # Joint5 (link4 -> link5)
    (-2.0944, 2.0944) # Joint6 (link5 -> link6)
    #link6 toimii pääte-efektorina ilman fyysistä pituutta.
]

LINK_TRANSFORMS = [
    {'trans': [0, 0, 0.123], 'rot': [0, 0, 0]},  # Joint1: base_link to link1
    {'trans': [0, 0, 0], 'rot': [1.5708, -0.10095, -3.1416]},  # Joint2: link1 to link2
    {'trans': [0.28503, 0, 0], 'rot': [0, 0, -1.759]},  # Joint3: link2 to link3
    {'trans': [-0.021984, -0.25075, 0], 'rot': [1.5708, 0, 0]},  # Joint4: link3 to link4
    {'trans': [0, 0, 0], 'rot': [-1.5708, 0, 0]},  # Joint5: link4 to link5
    {'trans': [0, -0.091, 0], 'rot': [1.5708, 0, 0]}  # Joint6: link5 to link6
]

URDF_FILE = 'pipermanual.urdf'
CONTROLLABLE_JOINTS = [1, 2, 3, 4, 5, 6]
BASE_LINK = "base_link"
END_EFFECTOR = "link6"
DT = 0.01 # Delta time, frequency of simulation

IK_CONFIG = {
    # General IK parameters (applies to all solvers)
    "max_iterations": 200,       # Maximum iterations before giving up
    "tolerance": 1e-3,           # Acceptable error (meters) for success
    "angle_update_rate": 0.3,    # Smoothing factor for visualization updates (0-1)

    # DLS-specific parameters
    "dls_step_size": 0.5,        # Learning rate for DLS updates (alpha)
    "dls_damping": 0.01,         # Lambda value for singularity avoidance

    # Newton-Raphson parameters
    "nr_step_size": 0.1,         # Conservative learning rate for NR stability

    # CCD-specific parameters
    "ccd_epsilon": 1e-6,         # Small value to prevent division by zero
    "ccd_axis_limit": 1e-6,      # Threshold for valid rotation axis

    # Numerical differentiation
    "finite_difference_eps": 1e-6,  # Perturbation size for numerical Jacobian
}

TARGETS = [
    np.array([0.2, 0.0, 0.3]),  # Right
    np.array([-0.2, 0.0, 0.3]),  # Left
    np.array([0.0, 0.2, 0.3]),  # Forward
    np.array([0.0, -0.2, 0.3]),  # Backward
    np.array([0.0, 0.0, 0.4]),  # Upper center
]


# ======================== WHICH OPERATION TO RUN ========================
JACOBIAN_TYPE = 'numerical' # Options: 'numerical', 'analytical'
IK_TYPE = 'NewtonRaphson' #           # Options: 'DLS', 'NewtonRaphson', 'CCD'

# ======================== GLOBAL STATE ========================
simulate = True
current_angles = [0.0] * len(CONTROLLABLE_JOINTS)
current_target = 0
last_target_change = time.time()
robot = None

angle_lock = threading.Lock()  # prevents race conditions
unreachable_target = False

# ======================== TRANSFORM MATRIX CREATION ========================
def create_transform_matrix(trans, rot):
    """Create 4x4 homogenous transformation matrix from translation and Euler angles"""
    tx, ty, tz = trans
    rx, ry, rz = rot

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    rotation = Rz @ Ry @ Rx
    translation = np.array([tx, ty, tz])

    return np.vstack([np.hstack([rotation, translation[:, np.newaxis]]), [0, 0, 0, 1]])

# Precompute transformation matrices
PRECOMPUTED_TRANSFORMS = [create_transform_matrix(link['trans'], link['rot']) for link in LINK_TRANSFORMS]


# ======================== KINEMATICS FUNCTIONS ========================

def forward_kinematics(joint_angles, return_all=False):
    """Calculate end effector position using precomputed transformations"""
    
    T = np.eye(4)  # Start with identity matrix
    transforms = [T.copy()]
    for i in range(len(joint_angles)):
        link_tf = PRECOMPUTED_TRANSFORMS[i]
        joint_rot = np.array([[np.cos(joint_angles[i]), -np.sin(joint_angles[i]), 0, 0],
                              [np.sin(joint_angles[i]), np.cos(joint_angles[i]), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        T = T @ link_tf @ joint_rot
        
        transforms.append(T.copy())

    return transforms if return_all else T[:3, 3] #transforms = kaikki välivaiheet, T[:3, 3]= pelkkä end effectorin sijainti


def numerical_jacobian(joint_angles, base_position):
    """Calculate Jacobian matrix numerically with base position reuse"""
    jac = np.zeros((3, len(joint_angles)))
    epsilon = IK_CONFIG["finite_difference_eps"]  # Use config value

    for i in range(len(joint_angles)):
        perturbed = np.array(joint_angles)
        perturbed[i] += epsilon
        new_pos = forward_kinematics(perturbed)
        jac[:, i] = (new_pos - base_position) / epsilon

    return jac

def analytical_jacobian(joint_angles):
    jac = np.zeros((3, len(joint_angles)))
    transforms = forward_kinematics(joint_angles, return_all=True)
    ee_pos = transforms[-1][:3, 3]  # End-effector position
    
    for i in range(len(joint_angles)):
        joint_pos = transforms[i+1][:3, 3]  # Joint position
        rotation_axis = transforms[i+1][:3, 2]  # Z-axis in world coordinates
        jac[:, i] = np.cross(rotation_axis, ee_pos - joint_pos)
    
    return jac

def calculate_ik_dls(current_angles, target_pos, joint_limits):
    """Damped Least Squares IK Solver"""
    max_iterations = IK_CONFIG["max_iterations"]
    tolerance = IK_CONFIG["tolerance"]
    alpha = IK_CONFIG["dls_step_size"]  # Step size
    lambda_ = IK_CONFIG["dls_damping"]  # Damping factor

    angles = np.array(current_angles)
    current_pos = forward_kinematics(angles)
    error = target_pos - current_pos

    for _ in range(max_iterations):
        if np.linalg.norm(error) < tolerance:
            return angles  # Success!

         # Select Jacobian type based on global setting
        if JACOBIAN_TYPE == 'numerical':
            J = numerical_jacobian(angles.copy(), current_pos)
        elif JACOBIAN_TYPE == 'analytical':
            J = analytical_jacobian(angles.copy())
        else:
            raise ValueError(f"Wrong jacobian input: pick 'numerical' or 'analytical'")

        try:
            inv = np.linalg.inv(J @ J.T + (lambda_**2) * np.eye(3)) #I=3x3, because jacobian=3xn
            delta = J.T @ inv @ error
        except np.linalg.LinAlgError:
            return None  # Singular matrix

        new_angles = angles + alpha * delta
        new_angles = np.clip(new_angles, [lim[0] for lim in joint_limits], [lim[1] for lim in joint_limits])

        if np.linalg.norm(new_angles - angles) < 1e-6:
            break

        angles = new_angles
        current_pos = forward_kinematics(angles)
        error = target_pos - current_pos

    return angles if np.linalg.norm(error) < tolerance else None


def calculate_ik_newton_raphson(current_angles, target_pos, joint_limits):
    """Newton-Raphson IK Solver"""
    max_iterations = IK_CONFIG["max_iterations"]
    tolerance = IK_CONFIG["tolerance"]
    alpha = IK_CONFIG["nr_step_size"]  # Step size

    angles = np.array(current_angles)
    current_pos = forward_kinematics(angles)
    error = target_pos - current_pos

    for _ in range(max_iterations):
        if np.linalg.norm(error) < tolerance:
            return angles  # Success!

        # Select Jacobian type based on global setting
        if JACOBIAN_TYPE == 'numerical':
            J = numerical_jacobian(angles.copy(), current_pos)
        elif JACOBIAN_TYPE == 'analytical':
            J = analytical_jacobian(angles.copy())
        else:
            raise ValueError(f"Invalid Jacobian type: {JACOBIAN_TYPE}")

        try:
            # Raw pseudoinverse without damping
            delta = np.linalg.pinv(J) @ error
        except np.linalg.LinAlgError:
            return None  # Singular matrix

        # Update angles with step size control
        new_angles = angles + alpha * delta
        new_angles = np.clip(new_angles, 
                           [lim[0] for lim in joint_limits], 
                           [lim[1] for lim in joint_limits])

        # Early exit if no progress
        if np.linalg.norm(new_angles - angles) < 1e-6:
            break

        angles = new_angles
        current_pos = forward_kinematics(angles)
        error = target_pos - current_pos

    return angles if np.linalg.norm(error) < tolerance else None

def calculate_ik_ccd(current_angles, target_pos, joint_limits):
    """Cyclic Coordinate Descent IK Solver"""
    max_iterations = IK_CONFIG["max_iterations"]
    tolerance = IK_CONFIG["tolerance"]
    epsilon = IK_CONFIG["ccd_epsilon"]
    axis_limit = IK_CONFIG["ccd_axis_limit"]

    angles = np.array(current_angles.copy())
    prev_error = np.inf

    for _ in range(max_iterations):
        # Compute all joint transforms and end-effector position
        transforms = forward_kinematics(angles, return_all=True)
        current_ee = transforms[-1][:3, 3]  # End-effector position
        current_error = np.linalg.norm(target_pos - current_ee)
        
        # Check convergence
        if current_error < tolerance:
            break
        if abs(prev_error - current_error) < 1e-6:
            break  # No progress
        prev_error = current_error

        # Process joints from end-effector to base
        for i in reversed(range(len(angles))):
            # Get joint position and orientation
            joint_pos = transforms[i+1][:3, 3]  # Joint position in world frame
            joint_rot = transforms[i+1][:3, :3]  # Joint rotation matrix

            # Calculate vectors to end-effector and target
            to_ee = current_ee - joint_pos
            to_target = target_pos - joint_pos
            
            # Normalize vectors (add epsilon to avoid division by zero)
            to_ee_norm = to_ee / (np.linalg.norm(to_ee) + epsilon)
            to_target_norm = to_target / (np.linalg.norm(to_target) + epsilon)

            # Calculate rotation axis and angle
            rotation_axis = np.cross(to_ee_norm, to_target_norm)
            axis_norm = np.linalg.norm(rotation_axis)
            
            if axis_norm < axis_limit:
                continue  # Skip if vectors are parallel

            # Convert rotation axis to joint's local frame (Z-axis only)
            local_axis = joint_rot.T @ (rotation_axis / axis_norm)  # World → Local
            angle_update = np.arctan2(axis_norm, np.dot(to_ee_norm, to_target_norm)) * local_axis[2]

            # Update joint angle (Z-axis only, respecting limits)
            new_angle = angles[i] + angle_update
            angles[i] = np.clip(new_angle, joint_limits[i][0], joint_limits[i][1])

            # Immediately update transforms for subsequent joints
            transforms = forward_kinematics(angles, return_all=True)
            current_ee = transforms[-1][:3, 3]

    return angles if current_error < tolerance else None


def calculate_ik(current_angles, target_pos, joint_limits):
    """Main IK solver router"""
    if IK_TYPE == 'DLS':
        return calculate_ik_dls(current_angles, target_pos, joint_limits)
    elif IK_TYPE == 'NewtonRaphson':
        return calculate_ik_newton_raphson(current_angles, target_pos, joint_limits)
    elif IK_TYPE == 'CCD':
        return calculate_ik_ccd(current_angles, target_pos, joint_limits)
    else:
        raise ValueError(f"Unknown IK type: {IK_TYPE}")

# ======================== MAIN FUNCTIONS ========================
def main():
    global robot
    script_dir = Path(__file__).parent.resolve()
    world = WorldModel()
    world.readFile(str(script_dir / URDF_FILE))

    robot = world.robot(0)
    reset_robot(robot, CONTROLLABLE_JOINTS)

    update_thread = threading.Thread(target=update_ik)
    print_thread = threading.Thread(target=print_status)

    update_thread.start()
    print_thread.start()

    vis.add("world", world)
    vis.add("target", TARGETS[0], color=(1,0,0,1))
    vis.run()

    global simulate
    simulate = False
    update_thread.join()
    print_thread.join()

def reset_robot(robot, movable_joints):
    """Reset robot to home configuration"""
    config = robot.getConfig()
    for i in movable_joints:
        config[i] = 0.0
    robot.setConfig(config)

def update_ik():
    global current_angles, current_target, last_target_change, unreachable_target
    
    while simulate:
        if time.time() - last_target_change > 5:
            with angle_lock:
                current_target += 1
                last_target_change = time.time()
                unreachable_target = False
            vis.add("target", get_current_target(), color=(1,0,0,1))

        solution = calculate_ik(current_angles.copy(), get_current_target(), JOINT_LIMITS)
        
        if solution is not None:
            with angle_lock:
                # Increase step size for visible updates
                current_angles = [a + IK_CONFIG["angle_update_rate"]*(t - a) for a, t in zip(current_angles, solution)]
                unreachable_target = False
        else:
            with angle_lock:
                unreachable_target = True

        with angle_lock:
            config = robot.getConfig()
            for i, idx in enumerate(CONTROLLABLE_JOINTS):
                config[idx] = current_angles[i]
            robot.setConfig(config)

        time.sleep(DT)

def get_current_target():
    """Get current target position"""
    return TARGETS[current_target % len(TARGETS)]


def print_status():
    """Print end effector position and joint angles with reachability info"""
    global unreachable_target
    while simulate:
        try:
            base = robot.link(BASE_LINK).getTransform()
            ee = robot.link(END_EFFECTOR).getTransform()
            
            # Calculate relative position
            R_base = np.array(base[0]).reshape(3,3)
            t_base = np.array(base[1])
            t_ee = np.array(ee[1])
            rel_pos = np.dot(R_base.T, t_ee - t_base)
            target_pos = get_current_target()
            distance = np.linalg.norm(rel_pos - target_pos)
            
            # Build status message
            status = [
                f"Method: {JACOBIAN_TYPE} Jacobian + {IK_TYPE} IK",
                f"Target: {target_pos.round(2)}",
                f"Current Position: {rel_pos.round(3)}",
                f"Distance from target: {distance:.3f}",
                f"Joint Angles: {np.rad2deg(current_angles).round(1)}°"
            ]
            
            # Add reachability warning
            if unreachable_target:
                status.append("[WARNING] Target is unreachable!")
            
            print("\n".join(status) + "\n")
            time.sleep(1)
        except Exception as e:
            print("Status error:", e)
            break

if __name__ == "__main__":
    main()