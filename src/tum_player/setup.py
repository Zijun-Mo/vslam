from setuptools import setup
import os

package_name = 'tum_player'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'opencv-python'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TUM RGB-D dataset player for ROS2 Humble.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tum_player_node = tum_player.tum_player_node:main',
        ],
    },
)
