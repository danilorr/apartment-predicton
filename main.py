# from web_scraper import WebScraper
from data_processor import DataProcessor
from exploratory_data_analyser import DataAnalyser
from model_selector import ModelSelector
from final_model_trainer import ModelTrainer

# web_scraper = WebScraper()
# web_scraper.start()

data_processor = DataProcessor()
data_processor.start()

data_analyser = DataAnalyser()
data_analyser.start()

model_selector = ModelSelector()
model_selector.start()

model_trainer = ModelTrainer()
model_trainer.start()
