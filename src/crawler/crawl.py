import requests
from bs4 import BeautifulSoup

from .lecturer_crawler import SoictProfileCrawler


def get_lecturers_urls(base_url, headers=None):
    links = []
    for i in range(1, 7):
        request = requests.get(url=base_url.format(i=i), headers=headers)
        if request.status_code == 200:
            soup = BeautifulSoup(request.content, "html.parser")
            divs = soup.find_all("div", class_="col small-12 large-3")
            if divs:
                for div in divs:
                    link = div.find("a")["href"]
                    links.append(link)

    return links


if __name__ == "__main__":
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
    }
    base_url = "https://soict.hust.edu.vn/can-bo/page/{i}"

    crawler = SoictProfileCrawler()

    links = get_lecturers_urls(base_url, headers)
    # Crawl specific profiles
    crawler.crawl_profiles(
        urls=links, output_file="../data/lecturers/soict_lecturers.json"
    )
