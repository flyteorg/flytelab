import sys

project_name = '{{ cookiecutter.project_name }}'

if "-" in project_name:
    print("ERROR: Project name cannot contain '-'.")

    # exits with status 1 to indicate failure
    sys.exit(1)