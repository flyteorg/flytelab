"""Core utilities to be used in Machine Learning projects."""


def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Read the entire content of a text file.

    Args:
        file_path (str): Local file path.
        encoding (str, optional): Encoding of the file. Defaults to 'utf-8'.

    Returns:
        str: The contents of the file.
    """
    with open(file_path, 'r', encoding=encoding) as file_descriptor:
        content = file_descriptor.read()
    return content
