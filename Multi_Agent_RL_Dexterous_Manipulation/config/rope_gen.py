import xml.etree.ElementTree as ET
import pybullet as p
import pybullet_data
import time

def create_rope_urdf(length=1.0, radius=0.01, damping=0.01, stiffness=100.0, num_links=10, filename='rope_edge_connected.urdf'):
    """
    Generates a URDF file for a rope with links connected at their edges.

    :param length: Total length of the rope in meters.
    :param stiffness: Stiffness of the joints connecting the links.
    :param num_links: Number of links to model the rope.
    :param filename: Name of the output URDF file.
    """
    link_length = length / num_links

    robot = ET.Element('robot', name='rope')

    for i in range(num_links):
        # Define link
        link = ET.SubElement(robot, 'link', name=f'link_{i}')

        # Inertial properties
        inertial = ET.SubElement(link, 'inertial')
        mass = ET.SubElement(inertial, 'mass', value='0.05')
        inertia = ET.SubElement(inertial, 'inertia',
                                ixx='1e-5', ixy='0', ixz='0',
                                iyy='1e-5', iyz='0',
                                izz='1e-5')

        # Visual
        visual = ET.SubElement(link, 'visual')
        origin = ET.SubElement(visual, 'origin', xyz=f'0 0 {link_length / 2}', rpy='0 0 0')
        geometry = ET.SubElement(visual, 'geometry')
        cylinder = ET.SubElement(geometry, 'cylinder', length=str(link_length), radius=str(radius))

        # Collision
        collision = ET.SubElement(link, 'collision')
        origin = ET.SubElement(collision, 'origin', xyz=f'0 0 {link_length / 2}', rpy='0 0 0')
        geometry = ET.SubElement(collision, 'geometry')
        cylinder = ET.SubElement(geometry, 'cylinder', length=str(link_length), radius=str(radius))

        if i > 0:
            # Define joint
            joint = ET.SubElement(robot, 'joint', name=f'joint_{i}', type='revolute')
            parent = ET.SubElement(joint, 'parent', link=f'link_{i-1}')
            child = ET.SubElement(joint, 'child', link=f'link_{i}')
            origin = ET.SubElement(joint, 'origin', xyz=f'0 0 {link_length}', rpy='0 0 0')
            axis = ET.SubElement(joint, 'axis', xyz='0 1 0')
            limit = ET.SubElement(joint, 'limit', lower='-1.57', upper='1.57', effort='0', velocity='0')
            # Add joint dynamics for stiffness
            dynamics = ET.SubElement(joint, 'dynamics', damping=str(damping), spring_stiffness=str(stiffness))
        else:
            # For the first link, set its origin
            # This ensures the rope starts at the world origin
            origin = ET.SubElement(link, 'origin', xyz='0 0 0', rpy='0 0 0')

    # Write to file
    tree = ET.ElementTree(robot)
    tree.write(filename)

def create_rope_with_ball_joints(length=1.0, radius=0.0075, damping=0.001, stiffness=100.0, num_links=10, filename='rope_ball_joint.urdf'):
    """
    Generates a URDF file for a rope with simulated ball joints using three revolute joints per connection.
    All visual elements are set to green color.

    :param length: Total length of the rope in meters.
    :param radius: Radius of each rope segment.
    :param damping: Damping coefficient for the joints.
    :param stiffness: Stiffness of the joints connecting the links.
    :param num_links: Number of links to model the rope.
    :param filename: Name of the output URDF file.
    """
    link_length = length / num_links

    robot = ET.Element('robot', name='rope')

    for i in range(num_links):
        # Define link
        link = ET.SubElement(robot, 'link', name=f'link_{i}')

        # Inertial properties
        inertial = ET.SubElement(link, 'inertial')
        mass = ET.SubElement(inertial, 'mass', value='0.05')
        inertia = ET.SubElement(inertial, 'inertia',
                                ixx='1e-5', ixy='0', ixz='0',
                                iyy='1e-5', iyz='0',
                                izz='1e-5')

        # Visual
        visual = ET.SubElement(link, 'visual')
        origin = ET.SubElement(visual, 'origin', xyz=f'0 0 {link_length / 2:.6f}', rpy='0 0 0')
        geometry = ET.SubElement(visual, 'geometry')
        cylinder = ET.SubElement(geometry, 'cylinder', length=f'{link_length:.6f}', radius=f'{radius:.6f}')
        # Add material with green color
        material = ET.SubElement(visual, 'material', name='green')
        color = ET.SubElement(material, 'color', rgba='0 1 0 1')  # RGBA for green color

        # Collision
        collision = ET.SubElement(link, 'collision')
        origin = ET.SubElement(collision, 'origin', xyz=f'0 0 {link_length / 2:.6f}', rpy='0 0 0')
        geometry = ET.SubElement(collision, 'geometry')
        cylinder = ET.SubElement(geometry, 'cylinder', length=f'{link_length:.6f}', radius=f'{radius:.6f}')

        if i > 0:
            # First joint (roll)
            joint_roll = ET.SubElement(robot, 'joint', name=f'joint_{i}_roll', type='revolute')
            parent = ET.SubElement(joint_roll, 'parent', link=f'link_{i-1}')
            child = ET.SubElement(joint_roll, 'child', link=f'link_{i}_roll')
            origin = ET.SubElement(joint_roll, 'origin', xyz=f'0 0 {link_length:.6f}', rpy='0 0 0')
            axis = ET.SubElement(joint_roll, 'axis', xyz='1 0 0')
            limit = ET.SubElement(joint_roll, 'limit', lower='-3.14', upper='3.14', effort='0', velocity='0')
            dynamics = ET.SubElement(joint_roll, 'dynamics', damping=f'{damping:.6f}', spring_stiffness=f'{stiffness:.6f}')

            # Intermediate link for roll joint
            link_roll = ET.SubElement(robot, 'link', name=f'link_{i}_roll')
            # Empty inertial for intermediate link
            inertial = ET.SubElement(link_roll, 'inertial')
            mass = ET.SubElement(inertial, 'mass', value='0.0')
            inertia = ET.SubElement(inertial, 'inertia',
                                    ixx='0', ixy='0', ixz='0',
                                    iyy='0', iyz='0',
                                    izz='0')

            # Second joint (pitch)
            joint_pitch = ET.SubElement(robot, 'joint', name=f'joint_{i}_pitch', type='revolute')
            parent = ET.SubElement(joint_pitch, 'parent', link=f'link_{i}_roll')
            child = ET.SubElement(joint_pitch, 'child', link=f'link_{i}_pitch')
            origin = ET.SubElement(joint_pitch, 'origin', xyz='0 0 0', rpy='0 0 0')
            axis = ET.SubElement(joint_pitch, 'axis', xyz='0 1 0')
            limit = ET.SubElement(joint_pitch, 'limit', lower='-3.14', upper='3.14', effort='0', velocity='0')
            dynamics = ET.SubElement(joint_pitch, 'dynamics', damping=f'{damping:.6f}', spring_stiffness=f'{stiffness:.6f}')

            # Intermediate link for pitch joint
            link_pitch = ET.SubElement(robot, 'link', name=f'link_{i}_pitch')
            # Empty inertial for intermediate link
            inertial = ET.SubElement(link_pitch, 'inertial')
            mass = ET.SubElement(inertial, 'mass', value='0.0')
            inertia = ET.SubElement(inertial, 'inertia',
                                    ixx='0', ixy='0', ixz='0',
                                    iyy='0', iyz='0',
                                    izz='0')

            # Third joint (yaw)
            joint_yaw = ET.SubElement(robot, 'joint', name=f'joint_{i}_yaw', type='revolute')
            parent = ET.SubElement(joint_yaw, 'parent', link=f'link_{i}_pitch')
            child = ET.SubElement(joint_yaw, 'child', link=f'link_{i}')
            origin = ET.SubElement(joint_yaw, 'origin', xyz='0 0 0', rpy='0 0 0')
            axis = ET.SubElement(joint_yaw, 'axis', xyz='0 0 1')
            limit = ET.SubElement(joint_yaw, 'limit', lower='-3.14', upper='3.14', effort='0', velocity='0')
            dynamics = ET.SubElement(joint_yaw, 'dynamics', damping=f'{damping:.6f}', spring_stiffness=f'{stiffness:.6f}')
        else:
            # For the first link, set its origin
            origin = ET.SubElement(link, 'origin', xyz='0 0 0', rpy='0 0 0')

    # Write to file
    tree = ET.ElementTree(robot)
    tree.write(filename)

# Example usage
# create_rope_urdf(length=0.2, radius=0.0075, damping=0.001, stiffness=20.0, num_links=20, filename='rope.urdf')
create_rope_with_ball_joints(length=0.3, radius=0.01, damping=0.01, stiffness=20, num_links=1, filename='./assets/rope.urdf')


if __name__=="__main__":

    # Start the physics client
    physicsClient = p.connect(p.GUI)  # Use p.DIRECT for non-graphical version

    # Set the simulation parameters
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    p.setRealTimeSimulation(0)

    # Optionally, add search path for PyBullet to find plane.urdf
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load a plane to act as the ground
    planeId = p.loadURDF("plane.urdf")

    # Load the rope URDF at a certain position
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    ropeId = p.loadURDF("assets/rope.urdf", basePosition=startPos, baseOrientation=startOrientation, useFixedBase=False)

    # Optionally adjust camera view
    p.resetDebugVisualizerCamera(cameraDistance=0.2, cameraYaw=50, cameraPitch=-75, cameraTargetPosition=[0, 0, 1])

    # Run the simulation for 10 seconds
    simulation_duration = 10000  # seconds
    time_step = 1./240.
    num_steps = int(simulation_duration / time_step)

    for step in range(num_steps):
        p.stepSimulation()
        time.sleep(time_step)  # Sleep to match real time

    # Disconnect the physics client
    p.disconnect()