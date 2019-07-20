from setuptools import setup, find_packages

setup(
    name='tautil',
    version='1.0.0',
    url='https://github.com/taungyeon/tautil.git',
    license='Free',
    author='taungyeon',
    author_email='taungyeon.0103@gmail.com',
    description='preprocessing for ML/DL',
    # install_requires=['setuptools'],
    packages=find_packages(),
    # entry_points={
    #     'console_scripts': [
    #         'top = topDirectory.top:main',
    #         'bottom = topDirectory.bottomDirectory.bottom:sub'
    #     ]
    # }
)
