# ReLearn

Introducing a dynamic package designed specifically for the development and implementation of reinforcement learning (RL) projects. This toolkit includes essential classes for modeling Markov Decision Processes (MDPs), agents, and environments, laying the foundation for RL systems. With our package, users have the flexibility to create custom environments, allowing for the exploration and testing of various RL algorithms. Whether you're a researcher, educator, or developer, this package provides the necessary tools to implement reinforcement learning applications.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

**Python**: This project requires a Python version >=3.9 and <3.13. You can download it from [python.org](https://www.python.org/downloads/).

This project can be installed using pip directly from PyPI or by cloning the repository from GitHub. Follow the instructions below based on your preferred method.

#### Installing from PyPI

First, consider creating a virtual environment:
```shell
python -m venv venv
source venv/bin/activate  # On Unix/macOS
.\venv\Scripts\activate   # On Windows
```

To install the package from PyPI, run the following command in your terminal. This is the simplest way to install the latest stable version of the project:

```shell
pip install relearn
```
Make sure you have pip installed and updated to the latest version to avoid any issues.

#### Installing from GitHub

If you prefer to install the latest development version or want to contribute to the project, you can clone the repository from GitHub and install it manually:

1. Clone the repository:
    ```shell
    git clone https://github.com/umilISLab/relearn.git
    ```

2. Navigate to the project directory:
    ```shell
    cd relearn
    ```

3. Consider creating a virtual environment:
    ```shell
    python -m venv venv
    source venv/bin/activate  # On Unix/macOS
    .\venv\Scripts\activate   # On Windows
    ```

4. Install the project and its dependencies using the preferred method:
    - **Dependency Management with Poetry**: This project uses [Poetry](https://python-poetry.org/) for dependency management and package handling. Ensure you have Poetry installed on your system. For installation instructions, visit the [official Poetry documentation](https://python-poetry.org/docs/#installation).

        To check if you have Poetry installed, run the following command in your terminal:
        ```shell
        poetry --version
        ```
        If Poetry is installed, you should see the version number in the output. If not, please follow the installation guide provided in the link above.
            - **Installing Dependencies**: With Poetry installed, you can install project dependencies by running:
            ```shell
            poetry install
            ```
    - If the project uses a `requirements.txt`:
        ```shell
        pip install -r requirements.txt
        ```

#### Verify the Installation

After installation, you can verify that the project is installed correctly by running:
```shell
python -c "import relearn; print(relearn.__version__)"
```

### Usage Example

You can find a _Recycling Robot_ example [here](https://github.com/umilISLab/relearn/tree/main/examples).

## Built With

* [poetry](https://maven.apache.org/) - Dependency Management

## Versioning

We use [SemVer](http://semver.org/) for versioning.

## Authors

* **Elisabetta Rocchetti** - *Initial work* - [ISLab](https://github.com/umilISLab)

## License

This project is licensed under the GNUv3 - see the [LICENSE.md](LICENSE.md) file for details