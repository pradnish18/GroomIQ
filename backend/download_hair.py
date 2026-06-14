import os, requests

os.makedirs("hairstyles", exist_ok=True)

images = {
    "straight_long.png": "https://www.freepngimg.com/download/hair/24-woman-hair-png-image.png",
    "wavy_long.png":      "https://www.freepngimg.com/download/hair/7-hair-png-image.png",
    "curly.png":          "https://www.freepngimg.com/download/hair/22-curly-hair-png-image.png",
    "afro.png":           "https://www.freepngimg.com/download/hair/36-hair-png-image.png",
    "bob.png":            "https://www.freepngimg.com/download/hair/18-hair-png-image.png",
}

for name, url in images.items():
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            with open("hairstyles/" + name, "wb") as f:
                f.write(r.content)
            print("OK " + name)
        else:
            print("FAIL " + name + " " + str(r.status_code))
    except Exception as e:
        print("ERR " + name + " " + str(e))