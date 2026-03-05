import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Paths and Folders
    pkg_tracker_control = get_package_share_directory('tracker_control')
    gazebo_ros_path = get_package_share_directory('gazebo_ros')
    
    world_path = os.path.join(pkg_tracker_control, 'worlds', 'test1.world')
    sdf_model_path = os.path.join(pkg_tracker_control, 'models', 'sliping_robot', 'model.sdf')

    # 2. Declare Launch Arguments (equivalent to <arg>)
    world_arg = DeclareLaunchArgument('world', default_value=world_path)
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # 3. Include Gazebo Launch
    # In ROS 2, gazebo.launch.py replaces empty_world.launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_path, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'gui': 'false',  # Set to 'true' if you want to see the Gazebo window
        }.items()
    )

    # 4. Spawn Robot Model
    # Replaces 'spawn_model' with 'spawn_entity.py'
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', sdf_model_path,
            '-entity', 'sliping_robot',
            '-x', '-4.0', '-y', '0.0', '-z', '0.3'
        ],
        output='screen'
    )

    # 5. Static Transform Publisher
    # ROS 2 Arg order: x y z yaw pitch roll frame_id child_frame_id
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['-4', '0', '0', '0', '0', '0', 'world', 'odom']
    )

    # 6. Image Publisher (replaces image_view/image_publisher)
    image_pub = Node(
        package='image_publisher',
        executable='image_publisher_node',
        name='image_pub',
        parameters=[{'filename': '../lab.jpg'}],
        remappings=[('/image', '/camera/image_raw')]
    )

    # 7. Our Ported EKF Node (The "Master Node")
    # Uncomment to launch automatically
    # tracker_node = Node(
    #     package='tracker_control',
    #     executable='image_analyzer.py', # Ensure this matches your CMakeLists entry
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time}]
    # )

    return LaunchDescription([
        world_arg,
        gazebo,
        spawn_robot,
        static_tf,
        image_pub,
        # tracker_node
    ])