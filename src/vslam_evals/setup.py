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
from glob import glob

package_name = 'vslam_evals'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Online ATE evaluator for TUM datasets.',
    license='TODO',
    tests_require=['pytest'],
    cmdclass={'develop': develop_cmd},
    entry_points={
        'console_scripts': [
            'eval_node = vslam_evals.eval_node:main',
            'seven_scenes_player = vslam_evals.seven_scenes_player_node:main',
        ],
    },
)
