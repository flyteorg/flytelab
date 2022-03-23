# coding=utf-8
"""Driver for the Wikivoyage website."""

from typing import Dict, Union, List

import requests
from bs4 import BeautifulSoup

from destinations_similarity.scraper.utils import format_image_url, get_html


def generate_wikivoyage_url(city_name: str) -> str:
    """Generate URL for the Wikivoyage page."""
    return f"https://en.wikivoyage.org/wiki/{city_name}"


def get_to_do(content: BeautifulSoup) -> str:
    """Retrieve paragraphs on the 'Do' section of the page."""
    do_span = content.find('span', {'id': 'Do'})
    if do_span is None:
        return None

    current_tag = do_span.parent.find_next(['p', 'h2', 'li'])
    paragraphs = ""

    while not current_tag.name == 'h2':
        paragraphs += current_tag.text + '\n'
        current_tag = current_tag.find_next(['p', 'h2', 'li'])

    return paragraphs.strip()


def get_go_next(content: BeautifulSoup) -> List[str]:
    """Retrieve places to go next on 'Go Next' page."""
    places = []
    go_next_span = content.find('span', {'id': 'Go_next'})

    if go_next_span is None:
        return places

    current_tag = go_next_span.parent.find_next(['li', 'noscript'])
    while not current_tag.name == 'noscript':
        recommendations = current_tag.find_all('a')
        if recommendations is not None:
            places += [
                rec.get('title') for rec in recommendations if rec.get('title')
            ]
        current_tag = current_tag.find_next(['li', 'noscript'])

    return places


class WikivoyageExtractor():
    """Driver for the Wikivoyage page."""

    def __init__(self, cities_names: List[str]):
        """Initialize driver."""
        self._brazilian_cities = [x.strip().upper() for x in cities_names]
        self.session = requests.Session()

    def extract_by_name(
        self, city_name: str, search_name: str
    ) -> Dict[str, Union[str, list, None]]:
        """Retrieve information of city from Wikivoyage."""
        data = {
            "name": city_name,
            "description_wikivoyage": None,
            "images_wikivoyage": None,
            "do": None,
            "next": None,
            "poi": None
        }

        html_page = get_html(
            generate_wikivoyage_url(search_name), self.session)

        if not html_page:
            return data

        # If the city's page is empty, return None
        empty_page_text = "there is currently no text in this page"
        if html_page.find("p").text.lower().find(empty_page_text) >= 0:
            return data

        # Retrieve description
        main_div = html_page.find('div', attrs={'class': 'mw-parser-output'})
        description = ""
        for element in main_div.children:
            if element.name == 'mw:tocplace':
                break
            if element.name == 'p':
                description += element.text + '\n'
        data['description_wikivoyage'] = description.strip() or None

        # Retrieve images
        city_images = []
        for element in main_div.find_all(attrs="thumbinner"):
            images = element.find_all("img")
            city_images += [
                format_image_url(img.get("src"))
                for img in images if img.get("src") is not None
            ]
        data['images_wikivoyage'] = city_images or None

        # Get Do
        data['do'] = get_to_do(main_div) or None

        # Get Go Next (cities) and points of interest
        go_next = [x for x in get_go_next(main_div) if x is not None]
        data['poi'] = [
            elem for elem in go_next
            if elem.strip().upper() not in self._brazilian_cities
        ]
        data['next'] = list(set(go_next) - set(data['poi'])) or None
        data['poi'] = data['poi'] or None

        return data
