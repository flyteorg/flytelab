# coding=utf-8
"""
--
"""
from typing import Dict, Union, Any

import requests
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
        result = None
        for retry in range(3):
            request = get(url=generate_wikipedia_url(citie), timeout=30)
            try:
                if request.status_code not in [200, 404]:
                    raise ValueError("Unexpected status code returned from url '%s'. Status returned: %s" %
                                     (generate_wikipedia_url(citie), request.status_code))
                if request.status_code == 404:
                    break
                else:
                    result = BeautifulSoup(request.text, "html.parser")
                    break
            except requests.RequestException as error:
                continue
        return result

    def extract_by_name(self, citie_name: str, seach_name: str) -> Union[Dict[str, Any], None]:
        html_page = self._get_html(seach_name)
        if not html_page:
            return {"name": citie_name, "description": None, "images": None, "do": None, "next": None}
        try:
            main_page = html_page.find("div", attrs={"class": "mw-parser-output"})
            description = ""
            citie_images = []
            first_header_found = False
            for element in main_page.findAllNext(recursive=False):
                if not first_header_found:
                    if element.name == "p":
                        description += element.text + "\n"
                    if element.name == "h2":
                        first_header_found = True
                if element.name == "img":
                    citie_images.append(format_image_url(element.get("src")))
            return {"name": citie_name, "description": description.strip(), "images": citie_images, "do": None, "next": None}
        except Exception as error:
            raise RuntimeError("Unexpected error occurred in extracting %s information" % citie_name) from error
