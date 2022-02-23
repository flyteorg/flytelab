import re
import sys

PROJECT_NAME_REGEX = r'^[_a-zA-Z][_a-zA-Z0-9]+$'

project_name = '{{ cookiecutter.project_name }}'

if not re.match(PROJECT_NAME_REGEX, project_name):
    print(f"ERROR: Project name is invalid. Must match the expression {PROJECT_NAME_REGEX}")
    sys.exit(1)
