# coding=utf-8
"""Driver for the Wikivoyage website."""

import re
from typing import Dict, Union

import requests

from destinations_similarity.scraper.utils import format_image_url, get_html


def generate_wikipedia_url(city_name: str) -> str:
    """Generate URL for the Wikipedia page."""
    return f"https://pt.wikipedia.org/wiki/{city_name}"


class WikipediaExtractor():
    """Driver for the Wikipedia page."""

    def __init__(self):
        """Initialize driver."""
        self.session = requests.Session()

    def extract_by_name(
        self, city_name: str, search_name: str
    ) -> Dict[str, Union[str, list, None]]:
        """Retrieve information of city from Wikipedia."""
        html_page = get_html(generate_wikipedia_url(search_name), self.session)

        if not html_page:
            return {
                "name": city_name,
                "description_wikipedia": None,
                "images_wikipedia": None,
            }

        main_div = html_page.find("div", attrs={"class": "mw-parser-output"})
        description = ""
        city_images = []
        first_header_found = False
        for element in main_div.find_all(["p", "h2", "img"]):
            # Get description
            if not first_header_found:
                if element.name == "p":
                    description += element.text + "\n"
                if element.name == "h2":
                    first_header_found = True

            # Get images
            if element.name == "img":
                city_images += [format_image_url(element.get("src"))]

        return {
            "name": city_name,
            "description_wikipedia":
                re.sub(r'\[.*?\]', '', description).strip() or None,
            "images_wikipedia": city_images or None
        }
