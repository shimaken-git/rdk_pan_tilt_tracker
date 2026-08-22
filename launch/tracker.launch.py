from launch import LaunchDescription

from launch_ros.actions import Node

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

    tracker_node = Node(
        package=package_name,
        executable='tracker_node',
        name='pan_tilt_tracker',
        output='screen',
        parameters=[
            config_file
        ]
    )

    return LaunchDescription([
        tracker_node
    ])
