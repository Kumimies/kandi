import time
import numpy as np
from klampt import WorldModel, vis
from pathlib import Path
from pynput import keyboard
import threading

# ======================== GLOBAL CONFIG ========================
JOINT_LIMITS = [(-2.618, 2.168), (0.1, 3.04), (-2.967, 0),
                (-1.745, 1.745), (-1.22, 1.22), (-2.0944, 2.0944)]
JOINT_INDICES = [1, 2, 3, 4, 5, 6]
BASE_LINK = "base_link"
END_EFFECTOR = "link6"
JOINT_SPEED = 0.5
DT = 0.01
FILENAME = 'pipermanual.urdf'

# ======================== GLOBAL STATE ========================
pressed_keys = set()
simulate = True
target_angles = [0.0] * len(JOINT_INDICES)
current_angles = [0.0] * len(JOINT_INDICES)
robot = None

# ======================== MAIN FUNCTION ========================
def main():
    global robot
    script_dir = Path(__file__).parent.resolve()
    world = WorldModel()
    world.readFile(str(script_dir / FILENAME))
    
    # Initialize robot
    robot = world.robot(0)
    config = robot.getConfig()
    for i in JOINT_INDICES:
        config[i] = 0.0
    robot.setConfig(config)

    # Start threads
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    update_thread = threading.Thread(target=update_kinematics)
    print_thread = threading.Thread(target=handle_end_effector)
    
    keyboard_listener.start()
    update_thread.start()
    print_thread.start()

    # Visualization
    vis.add("world", world)
    vis.run()
    
    # Cleanup
    global simulate
    simulate = False
    update_thread.join()
    print_thread.join()
    keyboard_listener.stop()

# ======================== CORE FUNCTIONS ========================
def handle_end_effector():
    while simulate:
        try:
            base = robot.link(BASE_LINK).getTransform()
            ee = robot.link(END_EFFECTOR).getTransform()
            
            R_base = np.array(base[0]).reshape(3,3)
            t_base = np.array(base[1])
            t_ee = np.array(ee[1])
            
            pos = np.dot(R_base.T, t_ee - t_base)
            print(f"End effector position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            time.sleep(2)
        except Exception as e:
            print("Position error:", e)
            break

def update_kinematics():
    key_bindings = {'1': (0,1), 'q': (0,-1), '2': (1,1), 'w': (1,-1),
                    '3': (2,1), 'e': (2,-1), '4': (3,1), 'r': (3,-1),
                    '5': (4,1), 't': (4,-1), '6': (5,1), 'y': (5,-1)}
    
    while simulate:
        # Update targets
        for key in pressed_keys:
            if key in key_bindings:
                idx, dir = key_bindings[key]
                delta = dir * JOINT_SPEED * DT
                new_angle = target_angles[idx] + delta
                target_angles[idx] = np.clip(new_angle, *JOINT_LIMITS[idx])
        
        # Smooth movement
        global current_angles
        current_angles = [a + 0.3*(t-a) for a,t in zip(current_angles, target_angles)]
        
        # Update robot
        config = robot.getConfig()
        for i, idx in enumerate(JOINT_INDICES):
            config[idx] = current_angles[i]
        robot.setConfig(config)
        
        time.sleep(DT)

# ======================== INPUT HANDLING ========================
def on_press(key):
    try:
        char = key.char.lower()
        if char in {'1','2','3','4','5','6','q','w','e','r','t','y'}:
            pressed_keys.add(char)
    except AttributeError:
        pass

def on_release(key):
    try:
        char = key.char.lower()
        if char in pressed_keys:
            pressed_keys.remove(char)
    except AttributeError:
        pass

if __name__ == "__main__":
    main()