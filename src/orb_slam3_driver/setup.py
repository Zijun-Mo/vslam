from setuptools import setup

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
    entry_points={
        'console_scripts': [
            'mono_driver_node = orb_slam3_driver.mono_driver_node:main'
        ],
    },
)
