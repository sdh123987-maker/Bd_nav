from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'bd_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # add launch directory
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # add asset directory
        ('share/' + package_name + '/assets', glob('share/' + package_name + '/assets/*')),
        # ui directory
        (os.path.join('share', package_name, 'ui'), ['bd_nav/user_input.py'])
    ],

    install_requires=['setuptools', 'openai', 'pynput', 'osmnx', 'networkx', 'matplotlib', 'geopy', 'pandas', 'utm', 'geopandas', 'shapely', 'watchdog', 'streamlit', 'streamlit_autorefresh'],
    zip_safe=True,
    maintainer='donghwijung',
    maintainer_email='donghwijung@snu.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'testing': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'ui = bd_nav.cli:main',
            'user_input = bd_nav.user_input:main',
            'intent_classifier = bd_nav.intent_classifier:main',
            'path_weighter = bd_nav.path_weighter:main',
            'path_generator = bd_nav.path_generator:main',
            'map_generator = bd_nav.map_generator:main',
            'map_viewer = bd_nav.map_viewer:main',
            'path_evaluator = bd_nav.path_evaluator:main',
            'extractor = bd_nav.extractor:main',
            'profiler = bd_nav.profiler:main',
            'gt_generator = bd_nav.gt_generator:main',
        ],
    },
)