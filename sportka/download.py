from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1200x3600')
prefs = {"download.default_directory" : "/tmp"}
options.add_experimental_option("prefs",prefs)

driver = webdriver.Chrome(chrome_options=options)
driver.implicitly_wait(10)

driver.get('https://www.sazka.cz/loterie/sportka/statistiky')
element = driver.find_element_by_id('p_lt_ctl09_wPL_p_lt_ctl03_wS_csvHistory_btnGetCSV')
driver.get_screenshot_as_file('main.png')
element.click()
driver.get_screenshot_as_file('click.png')

