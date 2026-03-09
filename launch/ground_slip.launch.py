import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node



def generate_launch_description():
    # 1. Start Gazebo Harmonic
    # We include the standard gz_sim.launch.py provided by ros_gz_sim

    pkg_tracker_control = get_package_share_directory('tracker_control')

    world_file_path = os.path.join(pkg_tracker_control, 'worlds', 'test1.world')

    gz_sim_pkg = get_package_share_directory('ros_gz_sim')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        # You can pass a specific world file here: 'empty.sdf' or your custom world
        launch_arguments={'gz_args': f'-r {world_file_path}'}.items() 
    )

    

    sdf_file_path = os.path.join(pkg_tracker_control, 'models', 'sliping_robot', 'model.sdf')

    # 2. Spawn your Robot ('sliping_robot') into the simulation
    # Ensure your robot's SDF is somewhere in GZ_SIM_RESOURCE_PATH
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'sliping_robot',
            '-file', sdf_file_path, # Replace with the actual path or use GZ_SIM_RESOURCE_PATH
            '-x', '-4.0', '-y', '0.0', '-z', '0.5'
        ],
        output='screen'
    )

    # 3. The ROS-Gazebo Bridge
    # This replaces all the old `gazebo_ros` plugin topics. We map ROS 2 topics to Gazebo topics here.
    # Syntax: /topic@ROS_TYPE[GZ_TYPE (for ROS->GZ) or ]GZ_TYPE (for GZ->ROS)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Clock (Crucial for ROS 2 time synchronization)
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            
            # Camera and Ground Truth
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/ground_truth/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            
            # --- FRONT LEFT MOTOR ---
            # Command Voltage: ROS (Float64) -> Gazebo (Double)
            '/front_left_motor/voltage@std_msgs/msg/Float64]gz.msgs.Double',
            # Encoder Feedback: Gazebo (Double) -> ROS (Float64)
            '/front_left_motor/encoder@std_msgs/msg/Int32[gz.msgs.Int32',
            
            # --- FRONT RIGHT MOTOR ---
            '/front_right_motor/voltage@std_msgs/msg/Float64]gz.msgs.Double',
            '/front_right_motor/encoder@std_msgs/msg/Int32[gz.msgs.Int32',
            
            # Add rear motors here as needed...
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        spawn_robot,
        bridge
    ])