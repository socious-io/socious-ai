import os
from configparser import ConfigParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    def __init__(self):
        # Load sensitive information from environment variables
        self.debug = os.environ.get('DEBUG', True)
        self.api_port = os.environ.get('API_PORT', 5000)
        self.idealist_token = os.environ.get('IDEALIST_TOKEN')
        self.nats_url = os.environ.get('NATS_URL')
        self.database_url = os.environ.get('DATABASE_URL')
        self.admin_token = os.environ.get('ADMIN_TOKEN')
        self.sql_dir = os.path.join(os.getcwd(), 'src/core/sql')
        self.http_proxy = dict(
            http=os.environ.get('HTTP_PROXY'),
            https=os.environ.get('HTTP_PROXY')
        )
        self.datadog = dict(
            api_key=os.environ.get('DD_API_KEY'),
            app_key=os.environ.get('DD_APP_KEY'),
            api_host='ap1.datadoghq.com'
        )
        # Load settings from the configuration file
        config = ConfigParser()
        config.read("config.ini")
        self.apps = config['core']['apps'].split(',')


# Initialize a global configuration object
config = Config()
