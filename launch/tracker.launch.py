from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    package_name = 'rdk_pan_tilt_tracker'

    package_dir = get_package_share_directory(
        package_name
    )

    config_file = os.path.join(
        package_dir,
        'config',
        'tracker.yaml'
    )

    headless = LaunchConfiguration('headless')

    declare_headless = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Disable the OpenCV debug window'
    )

    tracker_node = Node(
        package=package_name,
        executable='tracker_node',
        name='pan_tilt_tracker',
        output='screen',
        parameters=[
            config_file,
            {
                'headless': ParameterValue(
                    headless,
                    value_type=bool
                )
            }
        ]
    )

    return LaunchDescription([
        declare_headless,
        tracker_node
    ])
