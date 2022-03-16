"""Utilities package for Machine Learning projects."""

# pylama: ignore=W0611 (ignore unused imports)

# Import submodules
from destinations_similarity.utils import core
from destinations_similarity.utils import google

# Make core utilities more easily available
read_file = core.read_file
