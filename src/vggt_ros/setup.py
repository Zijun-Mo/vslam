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

package_name = 'vggt_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    # 将关键 Python 依赖放入此处，确保 colcon 构建时使用的解释器安装它们。
    # 若希望只用系统 apt 的 ROS2 包，可删除 rclpy；OpenCV 若用系统库亦可去掉 opencv-python。
    install_requires=[
        'setuptools',
        'torch==2.3.1',
        'torchvision==0.18.1',
        'numpy>=1.26.1',
        'Pillow',
        'huggingface_hub',
        'einops',
        'safetensors',
        'opencv-python',
        'hydra-core',
        'omegaconf',
        'pydantic==2.10.6',
        'tqdm',
        'requests',
        'onnxruntime',
        'trimesh',
        'matplotlib',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS2 package for VGGT',
    license='GPLv3 + VGGT specific (see LICENSE.txt)',
    tests_require=['pytest'],
    cmdclass={'develop': develop_cmd},
    entry_points={
        'console_scripts': [
            'vggt_node = vggt_ros.vggt_node:main',
        ],
    },
)
