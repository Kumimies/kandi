import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Define all joints and their transformations (parent, child, xyz, rpy)
joints = [
    ("world", "base_link", [0, 0, 0], [0, 0, 0]),          # Fixed base joint
    ("base_link", "link1", [0, 0, 0.123], [0, 0, 0]),     # Joint1
    ("link1", "link2", [0, 0, 0], [1.5708, -0.10095, -3.1416]),  # Joint2
    ("link2", "link3", [0.28503, 0, 0], [0, 0, -1.759]),  # Joint3
    ("link3", "link4", [-0.021984, -0.25075, 0], [1.5708, 0, 0]),  # Joint4
    ("link4", "link5", [0, 0, 0], [-1.5708, 0, 0]),       # Joint5
    ("link5", "link6", [8.8259e-5, -0.091, 0], [1.5708, 0, 0]),   # Joint6
    #link6 toimii pääte-efektorina ilman fyysistä pituutta.
]

def compute_transforms(joints):
    """Compute 4x4 transformation matrices for all joints."""
    transforms = {"world": np.eye(4)}  # World frame is identity
    
    for joint in joints:
        parent, child, xyz, rpy = joint
        
        # Get parent's transformation matrix
        parent_transform = transforms.get(parent, np.eye(4))
        
        # Create rotation matrix from RPY angles
        rotation = R.from_euler("xyz", rpy).as_matrix()
        
        # Build 4x4 transformation matrix for this joint
        joint_transform = np.eye(4)
        joint_transform[:3, :3] = rotation  # Apply rotation
        joint_transform[:3, 3] = xyz        # Apply translation
        
        # Compute child's transform: T_child = T_parent * T_joint
        child_transform = parent_transform @ joint_transform
        transforms[child] = child_transform
    
    return transforms

# Compute transformations for all joints
transforms = compute_transforms(joints)

# Extract positions of all joints
joints_order = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]
positions = [transforms[joint][:3, 3] for joint in joints_order]

colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'white', 'gray', 'orange']

# Plot the robot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Piper Robot Visualization")

# Plot joints as points and add small XYZ orientation lines (skip joint1)
for i in range(len(positions)):
    if i != len(positions) - 1:
        start = positions[i]
        end = positions[i + 1]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                marker='o', markersize=6, linewidth=4, color=colors[i])
    
    if i not in [1, 4]:  # Skip joint1 only
        # Add small XYZ orientation lines at each joint
        transform_matrix = transforms[joints_order[i]]
        origin = transform_matrix[:3, 3]
        
        x_axis = origin + transform_matrix[:3, 0] * 0.05
        y_axis = origin + transform_matrix[:3, 1] * 0.05
        z_axis = origin + transform_matrix[:3, 2] * 0.05
        
        ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r')
        ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g')
        ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b')

# Add labels for clarity
for i, pos in enumerate(positions):
    if i == 0:
        joint = "fixed_base_joint"
    else:
        joint = f"joint{i}"
    label = f"{joint}\n({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
    ax.text(pos[0], pos[1], pos[2], label, fontsize=8)
    print(label)

# Set axis limits
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_zlim(0, 0.3)

# Set axis scaling
ax.set_box_aspect([1, 1, 0.5])  # Equal aspect ratio
plt.show()