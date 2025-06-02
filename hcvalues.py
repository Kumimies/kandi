"""
urdf_to_esp32_params.py
Lightweight URDF parser for ESP32 parameter extraction
"""
import numpy as np
from urdf_parser_py.urdf import URDF
from pathlib import Path

URDF_FILE = 'pipermanual.urdf'  # Same directory as this script

def create_transform_matrix(trans, rot):
    """Create 4x4 matrix from translation and RPY rotation"""
    tx, ty, tz = trans
    rx, ry, rz = rot

    # Rotation matrices
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rx), -np.sin(rx), 0],
                   [0, np.sin(rx), np.cos(rx), 0],
                   [0, 0, 0, 1]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                   [0, 1, 0, 0],
                   [-np.sin(ry), 0, np.cos(ry), 0],
                   [0, 0, 0, 1]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0],
                   [np.sin(rz), np.cos(rz), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Translation matrix
    Trans = np.array([[1, 0, 0, tx],
                      [0, 1, 0, ty],
                      [0, 0, 1, tz],
                      [0, 0, 0, 1]])

    return Trans @ Rz @ Ry @ Rx

def main():
    # Load URDF directly
    urdf_path = Path(__file__).parent / URDF_FILE
    robot = URDF.from_xml_file(urdf_path)
    
    # Extract only revolute joints
    joints = [j for j in robot.joints if j.type == 'revolute']
    
    # Collect parameters
    joint_limits = []
    transforms = []
    
    for j in joints:
        # Joint limits
        joint_limits.append((j.limit.lower, j.limit.upper))
        
        # Transformation matrix
        origin = j.origin.xyz if j.origin else [0,0,0]
        rpy = j.origin.rpy if j.origin else [0,0,0]
        transforms.append(create_transform_matrix(origin, rpy))
    
    # Generate C++ output
    print(f"// Auto-generated from {URDF_FILE}\n")
    print("#pragma once\n#include <stdint.h>\n")
    
    # Joint limits
    print("const float JOINT_LIMITS[][2] = {")
    for lower, upper in joint_limits:
        print(f"    {{{lower:.6f}f, {upper:.6f}f}},")
    print("};\n")
    
    # Transformation matrices
    print("const float PRECOMPUTED_TRANSFORMS[][4][4] = {")
    for T in transforms:
        print("    {")
        for row in T:
            print(f"        {{{row[0]:.6f}f, {row[1]:.6f}f, {row[2]:.6f}f, {row[3]:.6f}f}},")
        print("    },")
    print("};\n")
    
    # Joint indices (1-based to match original Python code)
    print(f"const uint8_t CONTROLLABLE_JOINTS[] = {{{', '.join(str(i+1) for i in range(len(joint_limits)))}}};")

if __name__ == "__main__":
    main()