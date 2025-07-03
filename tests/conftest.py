# tests/conftest.py
import os
import sys

# Ensure the project root is on the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

os.environ['HADOOP_HOME'] = "C:\\hadoop" 
os.environ['PATH'] = f"{os.environ['HADOOP_HOME']}\\bin;{os.environ['PATH']}"