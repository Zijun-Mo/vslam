from setuptools import setup
from setuptools.command.develop import develop as _develop


class develop_cmd(_develop):
    """Accept and ignore colcon's extra develop flags (uninstall/editable)."""

    user_options = _develop.user_options + [
        ('uninstall', None, "Ignore uninstall for develop installs"),
        ('editable', None, "Ignore editable flag (handled by setuptools)"),
        ('build-directory=', None, "Ignored build directory for colcon develop"),
        ('script-dir=', None, "Ignored script dir for legacy develop"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.uninstall = None
        self.editable = None
        self.build_directory = None
        self.script_dir = None

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        # colcon may invoke "setup.py develop --uninstall" as a cleanup step;
        # distutils doesn't know this option, so we ignore it to keep builds going.
        if getattr(self, 'uninstall', None):
            return
        super().run()

package_name = 'orb_slam3_driver'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Image publisher driver for ORB-SLAM3',
    license='TODO',
    tests_require=['pytest'],
    cmdclass={'develop': develop_cmd},
    entry_points={
        'console_scripts': [
            'mono_driver_node = orb_slam3_driver.mono_driver_node:main'
        ],
    },
)
