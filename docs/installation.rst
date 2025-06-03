Installation Guide
==================

This guide covers different methods to install the AWS GenAI Architecture Visualizer.

Requirements
------------

* Python 3.8 or higher
* 512MB RAM minimum (1GB+ recommended)
* 50MB disk space
* Modern web browser (for web interface)

Installation Methods
-------------------

From TestPyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -i https://test.pypi.org/simple/ aws-genai-architecture-viz

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/yourusername/aws-genai-architecture-viz.git
   cd aws-genai-architecture-viz
   pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

For development with additional tools:

.. code-block:: bash

   git clone https://github.com/yourusername/aws-genai-architecture-viz.git
   cd aws-genai-architecture-viz
   pip install -e ".[dev]"

This includes development dependencies like pytest, black, flake8, and mypy.

Virtual Environment Setup
-------------------------

It's recommended to use a virtual environment:

.. code-block:: bash

   python -m venv aws-genai-env
   source aws-genai-env/bin/activate  # On Windows: aws-genai-env\\Scripts\\activate
   pip install -i https://test.pypi.org/simple/ aws-genai-architecture-viz

Verification
------------

Verify your installation:

.. code-block:: bash

   aws-genai-viz --help

Or test the Python API:

.. code-block:: python

   from aws_genai_architecture import AWSGenAIArchitecture
   print("Installation successful!")

Docker Installation (Optional)
------------------------------

You can also run using Docker:

.. code-block:: bash

   docker run -p 8080:8080 aws-genai-architecture-viz:latest

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**

If you encounter import errors, ensure you're using Python 3.8+:

.. code-block:: bash

   python --version

**Permission Issues**

On some systems, you might need to use `--user`:

.. code-block:: bash

   pip install --user -i https://test.pypi.org/simple/ aws-genai-architecture-viz

**Dependency Conflicts**

Use a fresh virtual environment to avoid conflicts:

.. code-block:: bash

   python -m venv fresh-env
   source fresh-env/bin/activate
   pip install -i https://test.pypi.org/simple/ aws-genai-architecture-viz

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/yourusername/aws-genai-architecture-viz/issues>`_
2. Join our community discussions
3. Contact support at your.email@example.com
