from setuptools import find_packages,setup
from typing import List

VAR = '-e .'
def get_requirement(file_path:str)->List[str]:

    '''
    This function will return requirement list
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.strip() for req in requirements if req.strip()] 
        if VAR in requirements:
            requirements.remove(VAR)
    return requirements

setup(
name='Performance_Prediction',
version='0.0.1',
author='Sudeshna Saha',
author_email='sahasudeshna15@gmail.com',
install_requires=get_requirement('requirements.txt'),
packages=find_packages(),
)