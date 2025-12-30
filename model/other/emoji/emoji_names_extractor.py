from bs4 import BeautifulSoup

def extract_emoji_names(html_text, limit=256):
    soup = BeautifulSoup(html_text, "html.parser")

    names = []
    for td in soup.select("div.main table tbody tr td.name"):
        name = td.get_text(strip=True)
        if not name:
            continue

        kebab = name.lower().replace(" ", "-")
        names.append(kebab)

        if len(names) == limit:
            break

    return names


# Example usage
if __name__ == "__main__":
    with open("full-emoji-list.html", "r", encoding="utf-8") as f:
        html = f.read()

    emoji_names = extract_emoji_names(html)

    print(emoji_names)
