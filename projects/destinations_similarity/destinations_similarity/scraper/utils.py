# coding=utf-8
"""Utilities used by scraper drivers."""

import re

import requests
from bs4 import BeautifulSoup


def format_image_url(url: str) -> str:
    """Format URL to add the correct scheme."""
    return re.sub('^(https?)?:?/?/?', 'https://', url)


def get_html(
    url: str, session: requests.Session = None, n_retries: int = 3
) -> BeautifulSoup:
    """Get the Beautiful Soup object for a remote HTML page."""
    session = session or requests.Session()

    for _ in range(n_retries):
        try:
            request = session.get(url, timeout=30, allow_redirects=True)
            if request.status_code == 200:
                return BeautifulSoup(request.text, "html.parser")
            if request.status_code == 404:
                return None

            raise requests.RequestException(
                f"Unexpected status code returned from url '{url}'. Status "
                f"returned: {request.status_code}"
            )
        except requests.RequestException:
            continue

    return None
