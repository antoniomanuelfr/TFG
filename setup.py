import setuptools

setuptools.setup(
                 name='tfg_utils',
                 version='1.0',
                 description='Utils for the development of TFG',
                 author='Antonio Manuel Fresneda Rodriguez',
                 author_email='antoniomanuelfr@gmail.com',
                 packages=setuptools.find_packages(),
                 install_requires=['numpy', 'scikit-learn', 'matplotlib', 'pandas', 'xgboost']
                 )
