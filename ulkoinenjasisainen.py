import numpy as np
import matplotlib.pyplot as plt

def create_transform_matrix(trans, rot, intrinsic=False):
    """Create 4x4 homogenous transformation matrix from translation and Euler angles"""
    tx, ty, tz = trans
    rx, ry, rz = rot

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0,          1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rx = np.array([[1, 0,           0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])

    if intrinsic:
        rotation = Rx @ Ry @ Rz  # Intrinsic: rotate around local axes
    else:
        rotation = Rz @ Ry @ Rx  # Extrinsic: rotate around world axes

    translation = np.array([tx, ty, tz])
    return np.vstack([np.hstack([rotation, translation[:, np.newaxis]]), [0, 0, 0, 1]])

# Joint definitions
joints = [
    ("world", "base_link", [0, 0, 0], [0, 0, 0]),
    ("base_link", "link1", [0, 0, 0.123], [0, 0, -np.pi]),
    ("link1", "link2", [0.3, 0, 0], [-np.pi/2, -np.pi/3, 0]),
    ("link2", "link3", [-0.022, -0.25, 0], [0, 0, -0.3]),
    ("link3", "link4", [0.00009, -0.09, 0], [0, 0, 0]),
]

def compute_transforms(joints, intrinsic=False, x_offset=0.0):
    """Compute 4x4 transformation matrices for all joints with optional intrinsic rotation and offset"""
    transforms = {"world": np.eye(4)}
    transforms["world"][:3, 3] = [x_offset, 0, 0]  # Apply offset to world origin
    
    for parent, child, xyz, rpy in joints:
        parent_transform = transforms[parent]
        joint_transform = create_transform_matrix(xyz, rpy, intrinsic=intrinsic)
        transforms[child] = parent_transform @ joint_transform
    
    return transforms

def get_joint_order(joints):
    """Extract child joint order from list"""
    seen = set()
    order = []
    for _, child, _, _ in joints:
        if child not in seen:
            order.append(child)
            seen.add(child)
    return order

def plot_robot(ax, transforms, joints_order, color_map):
    """Plot the robot structure"""
    positions = [transforms[joint][:3, 3] for joint in joints_order]

    for i in range(len(positions)):
        if i < len(positions) - 1:
            start, end = positions[i], positions[i + 1]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    marker='o', markersize=5, linewidth=3, color=color_map[i % len(color_map)])
        
        transform = transforms[joints_order[i]]
        origin = transform[:3, 3]
        x_axis = origin + transform[:3, 0] * 0.03
        y_axis = origin + transform[:3, 1] * 0.03
        z_axis = origin + transform[:3, 2] * 0.03

        ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r')
        ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g')
        ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b')

        if i==2:
            joint = f"joint{i}"
            label = f"{joint}\n({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})"
            
            text_offset = [-0.08, 0.05, -0.08] # Adjust these values as needed
            ax.text(origin[0] + text_offset[0], origin[1] + text_offset[1], origin[2] + text_offset[2], label, fontsize=8)

            print(label)


# Prepare data
joints_order = get_joint_order(joints)

# Extrinsic chain at x = -0.4
transforms_extrinsic = compute_transforms(joints, intrinsic=False, x_offset=-0.4)

# Intrinsic chain at x = +0.4
transforms_intrinsic = compute_transforms(joints, intrinsic=True, x_offset=+0.4)

# Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Extrinsic vs Intrinsic Rotation Chains")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

colors = ['red', 'green', 'blue', 'orange', 'yellow', 'cyan', 'magenta']

# Plot both robots
plot_robot(ax, transforms_extrinsic, joints_order, colors)
plot_robot(ax, transforms_intrinsic, joints_order, colors)


# Axis limits and aspect
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.3, 0.3)
ax.set_zlim(0.0, 0.5)
ax.set_box_aspect([1.2, 0.6, 0.5])

plt.show()
