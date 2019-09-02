from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='abem',
      version='1.0a0',
      description='Boundary Element Method for Acoustic Simulations',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/fjargsto/AcousticBEM',
      author='Frank Jargstorff',
      license='GNU General Public License',
      packages=['abem'],
      zip_safe=False)