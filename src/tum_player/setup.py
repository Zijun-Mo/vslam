from setuptools import setup
from setuptools.command.develop import develop as _develop


class develop_cmd(_develop):
    """Ignore colcon's extra develop flags (uninstall/editable/build-directory/script-dir)."""

    user_options = _develop.user_options + [
        ('uninstall', None, "Ignore uninstall for develop installs"),
        ('editable', None, "Ignore editable flag"),
        ('build-directory=', None, "Ignore build directory"),
        ('script-dir=', None, "Ignore script dir"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.uninstall = None
        self.editable = None
        self.build_directory = None
        self.script_dir = None

    def run(self):
        if getattr(self, 'uninstall', None):
            return
        super().run()
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
    cmdclass={'develop': develop_cmd},
    entry_points={
        'console_scripts': [
            'tum_player_node = tum_player.tum_player_node:main',
        ],
    },
)
