import logging
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.chrome.options import Options
from time import sleep


class WebScraper:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/web_scraper.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        self.chrome_options = Options()
        self.chrome_options.add_argument('window-size=400,800')
        self.chrome_options.add_argument('--headless')
        self.driver_url = 'https://www.vivareal.com.br/aluguel/sp/sao-paulo/apartamento_residencial/'
        self.driver = webdriver.Chrome(options=self.chrome_options)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 100
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.is_finished = False
        self.apts_df = pd.DataFrame()

    def start(self):
        self.logger.debug('Starting Class')
        self.open_browser()
        self.scrape_site()
        self.close_browser()
        # self.create_dataframe_csv()
        self.logger.debug('Ending Class')

    def open_browser(self):
        self.logger.debug('Opening browser')
        self.driver.get(self.driver_url)
        sleep(2.5)

    def scrape_apt_cards(self):
        self.logger.debug('Scraping cards on new page')
        # Scrapes the html using the class related to the apartment card
        self.apts = self.driver.find_elements_by_css_selector('[class="property-card__container js-property-card"]')

    def scrape_apt_info(self, apt):
        # Scrape the html looking for the specific class of each apartment feature
        self.apt_titl = apt.find_element_by_css_selector(
            '[class="property-card__title js-cardLink js-card-title"]')
        self.apt_addr = apt.find_element_by_css_selector(
            '[class="property-card__address"]')
        self.apt_area = apt.find_element_by_css_selector(
            '[class="property-card__detail-item property-card__detail-area"]')
        self.apt_room = apt.find_element_by_css_selector(
            '[class="property-card__detail-item property-card__detail-room js-property-detail-rooms"]')
        self.apt_brom = apt.find_element_by_css_selector(
            '[class="property-card__detail-item property-card__detail-bathroom js-property-detail-bathroom"]')
        self.apt_gara = apt.find_element_by_css_selector(
            '[class="property-card__detail-item property-card__detail-garage js-property-detail-garages"]')
        # Scrape the html looking for a specific style, since rent didn't have a specific class to it
        self.apt_rent = apt.find_element_by_css_selector(
            '[style="display: block;"]')
        try:
            # In the case Condominium Fee was R$ 0, the condominium field wasn't present in the html
            # So, if the element isn't found, its value is 0
            self.apt_cond = apt.find_element_by_css_selector(
                '[class="js-condo-price"]')
        except NoSuchElementException:
            self.apt_cond = 'R$ 0'

    def create_apt_info_list(self):
        try:
            self.apts_list = pd.Series([self.apt_titl.text, self.apt_addr.text, self.apt_area.text, self.apt_room.text,
                                        self.apt_brom.text, self.apt_gara.text, self.apt_rent.text, self.apt_cond.text])
        except AttributeError:
            # In the case apt.cond element wasn't found in the html, hard coding apt_cond = 'R$ 0' made it so
            # it was no longer a WebElement, but a string. As such, it doesn't have a .text attribute
            # So, if apt_cond turns out as a string, we can just use self.apt_cond instead
            self.apts_list = pd.Series([self.apt_titl.text, self.apt_addr.text, self.apt_area.text, self.apt_room.text,
                                        self.apt_brom.text, self.apt_gara.text, self.apt_rent.text, self.apt_cond])

    def append_list_to_apt_dataframe(self):
        self.apts_df = self.apts_df.append(self.apts_list, ignore_index=True)

    def print_df_tail(self):
        self.logger.debug(f"Printing new rows\n{self.apts_df.tail(36)}")

    def go_to_next_page(self):
        # Looks for the element in the page that has 'Próxima página' written on it
        # indicating the next page button
        next_page = self.driver.find_element_by_xpath("//*[contains(text(), 'Próxima página')]")
        # The common .click attribute wasn't working in this situation
        # So the following script had to be used
        self.driver.execute_script("arguments[0].click();", next_page)
        self.logger.debug(f'Moving to page: {self.driver.current_url}')
        sleep(2.5)

    def check_if_finished(self):
        # At the last page, the next page button was still present, but it was disabled
        next_page_disabled = self.driver.find_elements_by_css_selector(
            '[data-disabled][class="js-change-page"][title="Próxima página"]')
        if len(next_page_disabled) != 0:
            # is_finished is later used as a flag to determine whether the last page has been reached
            self.is_finished = True
            self.logger.debug('Arrived at last page')

    def scrape_site(self):
        # An infinite loop that is broken when the last page is reached
        while True:
            self.scrape_apt_cards()
            for apt in self.apts:
                try:
                    self.scrape_apt_info(apt)
                    self.create_apt_info_list()
                    self.append_list_to_apt_dataframe()
                except StaleElementReferenceException:
                    # In some rare occasions (around 0.3%), a stale element exception is triggered
                    # while scraping the apartment cards
                    # Since the loss of data is really small, it was decided to just skip the stale data
                    self.logger.warning('A Stale Element Exception has been caught')
            self.print_df_tail()
            self.go_to_next_page()
            self.check_if_finished()
            if self.is_finished is True:
                break

    def close_browser(self):
        self.driver.quit()
        self.logger.debug('Closing browser')

    def create_dataframe_csv(self):
        self.logger.debug('Creating csv file')
        self.apts_df = self.apts_df.rename(columns={0: 'Title', 1: 'Address', 2: 'Area', 3: 'Bedrooms',
                                                    4: 'Bathrooms', 5: 'Garage Cars', 6: 'Rent', 7: 'Condominium'})
        self.apts_df.to_csv(r'.csv files/apts_dataframe.csv')
        self.logger.debug('apts_dataframe.csv file created')
