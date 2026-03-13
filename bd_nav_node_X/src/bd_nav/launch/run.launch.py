from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bd_nav',
            executable='ui',
            name='ui_node',
            output='screen'
        ),
        Node(
            package='bd_nav',
            executable='map_generator',
            name='map_generator_node',
            output='screen'
        ),
        Node(
            package='bd_nav',
            executable='map_viewer',
            name='map_viewer_node',
            output='screen'
        ),
        Node(
            package='bd_nav',
            executable='intent_classifier',
            name='intent_classifier_node',
            output='screen'
        ),
        Node(
            package='bd_nav',
            executable='path_weighter',
            name='path_weighter_node',
            output='screen'
        ),
        Node(
            package='bd_nav',
            executable='path_generator',
            name='path_generator_node',
            output='screen'
        ),
         Node(
             package='bd_nav',
             executable='path_evaluator',
             name='path_evaluator_node',
             output='screen'
         ),
    ])
