from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import time


options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

try:
    api_url = "https://nobinobi.vn/shop-service/api/app/v1/product?offset=0&limit=18&warehouse_code=tbn1"
    driver.get(api_url)

    time.sleep(5)

    page_source = driver.find_element(By.TAG_NAME, "pre").text
    data = json.loads(page_source)

    print(data)

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

finally:
    driver.quit()