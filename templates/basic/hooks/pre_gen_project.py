import re
import sys

PROJECT_NAME_REGEX = r'^[_a-zA-Z][_a-zA-Z0-9]+$'
PROJECT_AUTHOR_REGEX = r'^[-_a-zA-Z0-9 ]+$'
FLYTE_PROJECT_REGEX = r'^[-a-z0-9]+$'

project_name = '{{ cookiecutter.project_name }}'
project_author = '{{ cookiecutter.project_author }}'
flyte_project = '{{ cookiecutter.flyte_project }}' or '{{ cookiecutter.project_author }}'.lower().replace("_", "-").replace(" ", "-")

if not re.match(PROJECT_NAME_REGEX, project_name):
    print(f"ERROR: project_name '{project_name}' is invalid. Must match the expression {PROJECT_NAME_REGEX}")
    sys.exit(1)

if not re.match(PROJECT_AUTHOR_REGEX, project_author):
    print(f"ERROR: project_author '{project_author}' is invalid. Must match the expression {PROJECT_AUTHOR_REGEX}")
    sys.exit(1)

if not re.match(FLYTE_PROJECT_REGEX, flyte_project):
    print(f"ERROR: flyte_project '{flyte_project}' is invalid. Must match the expression {FLYTE_PROJECT_REGEX}")
    sys.exit(1)
