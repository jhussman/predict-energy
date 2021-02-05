import pandas as pd
import os
import ssl
import requests
import zipfile
import glob
import time
import datetime


if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

downloads = '/Users/jhussman/Downloads/'


def download_file(url, file_name):
    r = requests.get(url, allow_redirects=True)
    open(file_name, 'wb').write(r.content)


def unzip_and_delete(file_name):
    zip_file = zipfile.ZipFile(file_name, 'r')
    zip_file.extractall(downloads + 'Temp Folder/')
    zip_file.close()
    os.remove(file_name)


def delete_files(directory):
    files = glob.glob(directory + '/*')
    for f in files:
        os.remove(f)


def index_to_datetime(df, col_name):
    df.index = pd.to_datetime(df.index) + (df.Hour - 1).astype('timedelta64[h]')
    return df[col_name].copy()


class APIWrapper:

    URL_G = 'http://oasis.caiso.com/oasisapi/GroupZip?groupid='
    URL_S = 'http://oasis.caiso.com/oasisapi/SingleZip?queryname='

    def __init__(self, node='GARNET_7_N001', trans='SCE-TAC', zone='SP15'):
        """
        :param node: str, name of node
        :param zone: str, demand zone of CA-ISO - either ISO-TAC, PGE, PGE-TAC, SCE-TAC, SDE-TAC
        """

        self.NODE = node
        self.TRANS = trans
        self.ZONE = zone

    def parse_date(self, date):
        """
        :param date: datetime obj, example datetime.date(2019, 1, 1)
        :return: tuple of strings
        """

        d_p1 = date + datetime.timedelta(days=1)

        date_start = str(date.year) + (str(date.month) if date.month > 9 else '0' + str(date.month)) + \
            (str(date.day) if date.day > 9 else '0' + str(date.day))

        date_end = str(d_p1.year) + (str(d_p1.month) if d_p1.month > 9 else '0' + str(d_p1.month)) + \
            (str(d_p1.day) if d_p1.day > 9 else '0' + str(d_p1.day))

        return date_start, date_end

    def lmp_by_date(self, day):
        """
        :param day: datetime obj, example datetime.date(2019, 1, 1)
        :return: pandas DataFrame with 24 hour profile of prices
        """

        # Create URL
        date, date_p1 = self.parse_date(day)
        date_p1 += 'T00:00-0000'

        url = self.URL_G + 'DAM_LMP_GRP&startdatetime=' + date_p1 + '&version=1&resultformat=6'

        # Download zip file, extract and delete
        zip_path = downloads + 'Temp Folder/' + date + '_' + date + '_DAM_LMP_GRP_N_N_v1_csv.zip'
        download_file(url, zip_path)

        path_base = downloads + 'Temp Folder/' + date + '_' + date + '_PRC_LMP_DAM_'
        unzip_and_delete(zip_path)

        # Load file into pandas DataFrame object and extract Node of interest
        df = pd.read_csv(path_base + 'LMP_v1.csv', index_col='OPR_DT')[['OPR_HR', 'NODE', 'MW']].copy()
        df.columns = ['Hour', 'Node', 'Price']
        df = df.sort_values(by=['Node', 'Hour'])
        df = df.loc[df.Node == self.NODE]

        delete_files(downloads + 'Temp Folder')
        return df  # index_to_datetime(df, 'Price')

    def demand_by_date(self, day, forecast='DAM'):
        """
        :param day: datetime obj, example datetime.date(2019, 1, 1)
        :param forecast: str, default DAM, can also be ACTUAL, RTM, 2DA, 7DA
        :return: pandas DataFrame with 24 hour profile of demand
        """

        date, date_p1 = self.parse_date(day)

        url = self.URL_S + 'SLD_FCST&startdatetime=' + date + 'T07:00-0000&market_run_id=' + forecast\
            + '&enddatetime=' + date_p1 + 'T08:00-0000' + '&version=1&resultformat=6'

        # Download zip file, extract and delete
        zip_path = downloads + date + '_' + date + '_' + 'SLD_FCST_N_N_v1_csv.zip'
        download_file(url, zip_path)
        unzip_and_delete(zip_path)
        path_base = [i for i in glob.glob(downloads + 'Temp Folder/*')]

        # Load file into pandas DataFrame object and extract Zone of interest
        df = pd.read_csv(path_base[0], index_col='OPR_DT')[['OPR_HR', 'TAC_AREA_NAME', 'MW']].copy()

        df.columns = ['Hour', 'Tac Area', 'Load']
        df = df.sort_values(by=['Tac Area', 'Hour'])
        df = df.loc[(df.index == str(day.year) + '-' + (str(day.month) if day.month > 9 else '0' + str(day.month))
                    + '-' + (str(day.day) if day.day > 9 else '0' + str(day.day))) & (df['Tac Area'] == self.TRANS)]

        delete_files(downloads + 'Temp Folder')
        return df  # index_to_datetime(df, 'Load')

    def vre_by_date(self, day, forecast='DAM', vre='Wind'):
        """
        :param day: datetime obj, example datetime.date(2019, 1, 1)
        :param forecast: str, default DAM, can also be RTD (5min), RTPD (15min), HASP (updated hourly) or ACTUAL
        :param vre: str, default wind, can also be Solar
        :return: pandas DataFrame with 24 hour profile of demand
        """

        date, date_p1 = self.parse_date(day)

        url = self.URL_S + 'SLD_REN_FCST&startdatetime=' + date + 'T07:00-0000&enddatetime=' + date_p1 + 'T08:00-0000'\
            + '&version=1&resultformat=6'

        # Download zip file, extract and delete
        zip_path = downloads + date + '_' + date + '_' + 'SLD_REN_FCST_N_N_v1_csv.zip'
        download_file(url, zip_path)
        unzip_and_delete(zip_path)
        path_base = [i for i in glob.glob(downloads + 'Temp Folder/*')]

        # Load file into pandas DataFrame object and extract Zone of interest
        df = pd.read_csv(path_base[0], index_col='OPR_DT')[['OPR_HR', 'TRADING_HUB', 'RENEWABLE_TYPE', 'MW',
                                                            'MARKET_RUN_ID']].copy()

        df.columns = ['Hour', 'Trading Hub', 'Renewable Type', 'Generation', 'Market']
        df = df.sort_values(by=['Trading Hub', 'Renewable Type', 'Hour'])
        df = df.loc[(df['Trading Hub'] == self.ZONE) & (df['Market'] == forecast) &
                    (df.index == str(day.year) + '-' + (str(day.month) if day.month > 9 else '0' + str(day.month))
                    + '-' + (str(day.day) if day.day > 9 else '0' + str(day.day)))]  # & (df['Renewable Type'] == vre)]
        df = df.loc[df['Renewable Type'] == vre] if vre != 'all' else df
        delete_files(downloads + 'Temp Folder')
        return df

    def fuel_price_by_day(self, day):
        """
        :param day:
        :return:
        """
        date, date_p1 = self.parse_date(day)

        url = 'http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_FUEL&fuel_region_id=ALL&startdatetime=' + date +\
              'T07:00-0000&enddatetime=' + date_p1 + 'T08:00-0000&version=1&resultformat=6'

        # Download zip file, extract and delete
        zip_path = downloads + date + '_' + date + '_' + 'PRC_FUEL_N_N_v1_csv.zip'
        download_file(url, zip_path)
        unzip_and_delete(zip_path)
        path_base = [i for i in glob.glob(downloads + 'Temp Folder/*')]

        # Load file into pandas DataFrame object and extract Zone of interest
        df = pd.read_csv(path_base[0], index_col='OPR_DT')[['OPR_HR', 'FUEL_REGION_ID', 'GROUP', 'PRC']].copy()

        df.columns = ['Hour', 'Fuel Region', 'Group', 'Price']
        df = df.sort_values(by=['Fuel Region', 'Hour'])
        delete_files(downloads + 'Temp Folder')

        return df

    def outages_by_day(self, day):
        # Create URL
        date, date_p1 = self.parse_date(day)
        date_p1 += 'T00:00-0000'

        url = self.URL_G + 'AGGR_OUTAGE_SCH_GRP&startdatetime=' + date_p1 + '&version=1&resultformat=6'

        # Download zip file, extract and delete
        zip_path = downloads + 'Temp Folder/' + date + '_' + date + '_AGGR_OUTAGE_SCH_GRP_N_N_v1_csv.zip'
        download_file(url, zip_path)

        path_base = downloads + 'Temp Folder/' + date + '_' + date + '_AGGR_OUTAGE_SCH_N_v1.csv'
        unzip_and_delete(zip_path)

        # Load file into pandas DataFrame object and extract Node of interest
        df = pd.read_csv(path_base, index_col='OUTAGE_DATE')[['OUTAGE_HOUR', 'FUEL_CATEGORY', 'TRADING_HUB',
                                                              'MW', 'REPORT_DATE']].copy()
        df.columns = ['Hour', 'Type', 'Trading Hub', 'Outage', 'Report Date']
        df = df.sort_values(by=['Type', 'Hour'])
        df = df.loc[df['Trading Hub'] == self.ZONE]
        df = df.loc[df.index == str(day + datetime.timedelta(days=1))]

        delete_files(downloads + 'Temp Folder')
        return df  # index_to_datetime(df, 'Price')

    def loop_historical(self, day_i=datetime.date(2017, 10, 1), day_f=datetime.date.today(), param='demand'):
        i = day_i
        df = pd.DataFrame([])
        while i <= day_f:
            print(str(i))
            if param == 'demand':
                df_i = self.demand_by_date(i)
            elif param == 'price':
                df_i = self.lmp_by_date(i)
            elif param == 'fuel':
                df_i = self.fuel_price_by_day(i)
            elif param == 'outages':
                df_i = self.outages_by_day(i)
            else:
                df_i = self.vre_by_date(i, vre=param.capitalize())
            df = df.append(df_i)
            i += datetime.timedelta(days=1)
            df.to_csv(param + '_data.csv')
            time.sleep(5)
        # df.index = pd.date_range(day_i, day_f+datetime.timedelta(days=1), freq='H')
        # df.to_csv(param + '_data.csv')
        return df

    def extract_data(self, day=(datetime.date.today() + datetime.timedelta(days=1))):
        data = dict()

        # If DAM is not available, go back to 2-day forecast
        try:
            load = self.demand_by_date(day, 'DAM')['Load'].to_list()
            time.sleep(5)
            print('Successfully extracted load data.')

            if len(load) == 0:
                load = self.demand_by_date(day, '2DA')['Load'].to_list()
                time.sleep(5)
                print('Successfully extracted load data - DAM was unavailable so extracted 2DA.')

            vre = self.vre_by_date(day, vre='all')
            time.sleep(5)
            print('Successfully extracted VRE data.')

            wind = vre.loc[vre['Renewable Type'] == 'Wind']['Generation'].to_list()
            solar = vre.loc[vre['Renewable Type'] == 'Solar']['Generation'].to_list()

            price_minus_1 = self.lmp_by_date(day - datetime.timedelta(days=1))['Price'].to_list()
            time.sleep(5)
            print('Successfully extracted LMP data from the previous day.')

            # Outages
            outages = self.outages_by_day(day - datetime.timedelta(days=1))
            time.sleep(5)
            print('Successfully extracted outage data.')

            outages_hydro = outages.loc[outages['Type'] == 'Hydro']['Outage'].to_list()
            outages_vre = outages.loc[outages['Type'] == 'Renewable']['Outage'].to_list()
            outages_thermal = outages.loc[outages['Type'] == 'Thermal']['Outage'].to_list()

            # Fuel price
            fuel_price = self.fuel_price_by_day(day)
            time.sleep(5)
            print('Successfully extracted fuel price data.')

            fuel_price = 24 * fuel_price.loc[fuel_price['Fuel Region'] == 'FRSCE1']['Price'].to_list()

            for w, s, o_h, o_v, o_t, f_p, lmp, lo, h in zip(wind, solar, outages_hydro, outages_vre, outages_thermal,
                                                            fuel_price, price_minus_1, load, range(1, 25)):
                data.update({h: {'wind': w, 'solar': s, 'outages_hydro': o_h, 'outages_vre': o_v, 'load': lo,
                                 'outages_thermal': o_t, 'fuel_price': f_p, 'price_minus_1': lmp,
                                 'day_of_year': (day - datetime.date(day.year, 1, 1)).days + 1, 'hour': h,
                                 'day_of_week': day.weekday()
                                 }})
            return data
        except ConnectionError:
            print('Unable to extract data. Please try again.')


a = APIWrapper()

