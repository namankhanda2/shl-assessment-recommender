from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time

# Setup Chrome
options = webdriver.ChromeOptions()
options.add_argument("--headless")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

url = "https://www.shl.com/solutions/products/product-catalog/"

driver.get(url)

# Wait for JS content
time.sleep(5)

print("TITLE:", driver.title)

# Get page source
html = driver.page_source

print(html[:2000])

driver.quit()