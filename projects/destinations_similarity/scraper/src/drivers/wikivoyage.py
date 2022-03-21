# coding=utf-8
"""
--
"""
from typing import Dict, Union, Any, List

import requests
from requests import get
from bs4 import BeautifulSoup


def generate_wikivoyage_url(citie_name: str) -> str:
    """
    Args:
    Returns:
    """
    return f"https://en.wikivoyage.org/wiki/{citie_name}"


def format_image_url(url: str) -> str:
    if not url.startswith("http") or not url.startswith("https"):
        return "https:" + url
    else:
        return url


def get_to_do(element, to_do: list, find_indicator: bool, stop_search: bool):
    finded = find_indicator
    stop = stop_search
    if not stop:
        if not finded:
            if element.name == "h2":
                if element.find("span") and element.find("span").get("id") == "Do":
                    finded = True
        else:
            if element.name == "p":
                to_do.append(element.text)
            if element.name == "ul":
                for li in element.findAll("li"):
                    to_do.append(li.text)
            if element.name == "h2":
                stop = True
        return finded, stop
    else:
        return finded, stop


def get_go_next(element, cities_next: list, find_indicator: bool, stop_search: bool):
    finded = find_indicator
    stop = stop_search
    if not stop:
        if not finded:
            if element.name == "h2":
                if element.find("span") and element.find("span").get("id") == "Go_next":
                    finded = True
        else:
            if element.name == "ul":
                for li in element.findAll("li"):
                    for city in li.findAll("a"):
                        cities_next.append(city.get("title"))
            if element.name == "p":
                for city in element.findAll("a"):
                    cities_next.append(city.get("title"))
            if element.name == "noscript":
                stop = True
        return finded, stop
    else:
        return finded, stop


def filter_none_from_list(elements: list):
    return list(filter(lambda x: True if x is not None else False, elements))


class WikivoyagerExtractor:

    def __init__(self, cities_name: List[str]):
        self._brazilian_cities = list(map(lambda x: x.strip().upper(),cities_name))

    @staticmethod
    def _get_html(citie) -> BeautifulSoup:
        result = None
        for retry in range(3):
            try:
                request = get(url=generate_wikivoyage_url(citie), timeout=30)
                if request.status_code not in [200, 404]:
                    raise ValueError("Unexpected status code returned from url '%s'. Status returned: %s" %
                                     (generate_wikivoyage_url(citie), request.status_code))
                else:
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
            return None
        try:
            if html_page.findAll("p")[0].text.lower().find("there is currently no text in this page") >= 0:
                return None
            else:
                main_div = html_page.find("div", attrs={"class": "mw-parser-output"})
                description = ""
                for element in main_div.findChildren("p", limit=3):
                    if element.text:
                        if element.text.find("citie_name"):
                            description += element.text + "\n"

                citie_images = []
                for element in main_div.findAll(attrs="thumbinner"):
                    images = element.findAll("img")
                    citie_images.extend([format_image_url(img.get("src")) for img in images if img.get("src") is not None])

                citie_to_do = []
                cities_next = []
                go_next_finded = False
                go_next_stop_search = False
                to_do_finded = False
                to_do_stop_search = False
                for element in main_div.findAllNext(recursive=False):
                    if not go_next_stop_search:
                        go_next_finded, go_next_stop_search = get_go_next(element, cities_next, go_next_finded,
                                                                          go_next_stop_search)
                    if not to_do_stop_search:
                        to_do_finded, to_do_stop_search = get_to_do(element, citie_to_do, to_do_finded, to_do_stop_search)

            citie_to_do = filter_none_from_list(citie_to_do)
            cities_next = filter_none_from_list(cities_next)

            poi = []
            for element in cities_next:
                if element.strip().upper() not in self._brazilian_cities:
                    poi.append(element)
            cities_next = list(filter(lambda x: True if x not in poi else False, cities_next))

            return {"name": citie_name, "description": description.strip(), "images": citie_images, "do": "\n".join(citie_to_do),
                    "next": cities_next if cities_next else None, "poi": poi if poi else None}
        except Exception as error:
            raise RuntimeError("Unexpected error occurred in extracting %s information" % citie_name) from error
