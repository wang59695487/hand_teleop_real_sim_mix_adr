import setuptools

# Function to read requirements from a file
def read_requirements(filename):
    with open(filename) as f:
        # Filter out comments and empty lines
        return [line.strip() for line in f if line and not line.startswith('#')]

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name="hand_teleop",  # Replace with your username

    version="2.0.1",

    author="<authorname>",

    author_email="<authorname@templatepackage.com>",

    description="<Template Setup.py package>",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="<https://github.com/authorname/templatepackage>",

    packages=setuptools.find_packages(),

    classifiers=[

        "Programming Language :: Python :: 3",

        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",

    ],

    python_requires='>=3.6',
    install_requires=read_requirements('requirements.txt'), 
)
