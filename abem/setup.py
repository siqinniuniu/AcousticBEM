from setuptools import setup, Extension

intops = Extension('intops',
                   sources=['intops/intops.c'],
                   depends=['intops/intops.h'],
                   include_dirs=['intops'],
                   language='c', )

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='abem',
    version='0.0a3',
    description='Boundary Element Method for Acoustic Simulations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/fjargsto/AcousticBEM',
    author='Frank Jargstorff',
    license='GNU General Public License',
    packages=['abem'],
    install_requires=['numpy', 'scipy'],
    zip_safe=False,
    ext_modules=[intops],
    test_suite='test',
    data_files=[(
        'notebooks', [
            'notebooks/exterior_helmholtz_solver_2d.ipynb',
            'notebooks/exterior_helmholtz_solver_3d.ipynb',
            'notebooks/exterior_helmholtz_solver_rad.ipynb',
            'notebooks/interior_helmholtz_solver_2d.ipynb',
            'notebooks/interior_helmholtz_solver_3d.ipynb',
            'notebooks/interior_helmholtz_solver_rad.ipynb',
            'notebooks/rayleigh_cavity_1.ipynb',
            'notebooks/rayleigh_cavity_2.ipynb',
            'notebooks/rayleigh_solver_3d_disk.ipynb',
            'notebooks/rayleigh_solver_square.ipynb']
    )],
)
