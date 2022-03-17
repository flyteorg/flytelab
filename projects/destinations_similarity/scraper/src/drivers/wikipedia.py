# coding=utf-8
"""
--
"""
from typing import Dict, Union, Any
from requests import get
from bs4 import BeautifulSoup


def generate_wikipedia_url(citie_name: str) -> str:
    """
    Args:
    Returns:
    """
    return f"https://en.wikipedia.org/wiki/{citie_name}"


def format_image_url(url: str) -> str:
    if not url.startswith("http"):
        return "https:" + url


class WikipediaExtractor:

    def __init__(self):
        pass

    @staticmethod
    def _get_html(citie: str) -> BeautifulSoup:
        request = get(url=generate_wikipedia_url(citie))
        if request.status_code not in [200,404]:
            raise ValueError("Unexpected status code returned from url '%s'. Status returned: %s" %
                             (generate_wikipedia_url(citie), request.status_code))
        if request.status_code == 404:
            return None
        else:
            return BeautifulSoup(request.text, "html.parser")

    def extract_by_name(self, citie_name: str, seach_name: str) -> Union[Dict[str, Any], None]:
        html_page = self._get_html(seach_name)
        if not html_page:
            return {"name": citie_name, "description": None, "images": None, "do": None, "next": None}
        try:
            description = None
            citie_images = None
            paragraphs = html_page.findAll("p")
            if paragraphs and len(paragraphs) > 1:
                description = paragraphs[1].text
            images = html_page.findAll("img")
            if images:
                citie_images = [format_image_url(img.get("src")) for img in images]
            return {"name": citie_name, "description": description, "images": citie_images, "do": None, "next": None}
        except Exception as error:
            raise RuntimeError("Unexpected error occurred in extracting %s information" % citie_name) from error
