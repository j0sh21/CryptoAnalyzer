import mariadb
import requests
from price import config
from price import mail
import time
import datetime
from price import insert1_12 as insert

headers = {
    'X-CMC_PRO_API_KEY': config.key_coinmarketcap,
    'Accepts': 'application/json.json',
}

params = {
    'start': '1',
    'limit': '50',
    'convert': 'USD'
}
# see docsc https://coinmarketcap.com/api/documentation/v1/#tag/cryptocurrency
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'



if 1 == 1 :
    def run(fetch_id):
        try:
            json_data = requests.get(url, params=params, headers=headers).json()
            coins = json_data['data']
            a=""
            insert.delete('coins')
            for i in coins:
                if i['symbol'] in ('BTC', 'ETH', 'WBTC', 'BNB', 'MATIC', 'ATOM', 'DOGE'):
                    a = str.lower(str(i['symbol']))
                    insert.html(str(i['id']), a)
                else:
                    insert.html(str(i['id']), str(i['symbol']))
                coinid = i['id']
                for userid in insert.getAlertsUsid():
                    try:
                        if i['quote']['USD']['price'] > insert.getconf('upper', userid[0], coinid) and int(
                                insert.getconf('new', userid[0], coinid)) == 1:
                            print('ueber')
                            mail.mail(
                                f"Hallo {insert.getusname(userid[0])}, \n\nDer Preis von {i['symbol']} ist ueber {insert.getconf('upper', userid[0], coinid)} gestiegen!\n\nDein Alert wurde deaktiviert, du kannst auf unserer Webseite einen neuen Alert einrichten oder diesen erneut aktivieren.\n\nViele Gruesse \ndein CryptoAlert Team :)",
                                config.subject, config.mailfrom, insert.getmail(userid[0]),
                                config.smtp, config.mail_usr, config.mail_pwd)
                            insert.updateconf(userid[0], coinid)
                        elif i['quote']['USD']['price'] < insert.getconf('lower', userid[0], coinid) and int(
                                insert.getconf('new', userid[0], coinid)) == 1:
                            print('unter')
                            mail.mail(
                                f"Hallo {insert.getusname(userid[0])}, \n\nDer Preis von {i['symbol']} ist unter {insert.getconf('lower', userid[0], coinid)} USD gefallen!\n\nDein Alert wurde deaktiviert, du kannst auf unserer Webseite einen neuen Alert einrichten oder diesen erneut aktivieren.\n\nViele Gruesse \ndein CryptoAlert Team :)",
                                config.subject, config.mailfrom, insert.getmail(userid[0]),
                                config.smtp, config.mail_usr, config.mail_pwd)
                            insert.updateconf(userid[0], coinid)
                        else:
                            print(f"\nKeine E-Mail zu Alert von {insert.getusname(userid[0])} bei {i['symbol']} versendet.\n")
                    except:
                        None

                try:
                    insert.insert(i['quote']['USD']['price'], str(i['quote']['USD']['last_updated']), str(i['symbol']).upper(), str(i['quote']['USD']['volume_24h']), str(i['quote']['USD']['volume_change_24h']), str(i['quote']['USD']['percent_change_1h']), str(i['quote']['USD']['percent_change_24h']), str(i['quote']['USD']['percent_change_7d']), str(i['quote']['USD']['percent_change_30d']), str(i['quote']['USD']['percent_change_60d']), str(i['quote']['USD']['percent_change_90d']), str(i['quote']['USD']['market_cap']), str(i['quote']['USD']['market_cap_dominance']), str(i['cmc_rank']), str(fetch_id))

                except mariadb.ProgrammingError:

                    insert.create(str(i['symbol']))
                    insert.insert(i['quote']['USD']['price'], str(i['quote']['USD']['last_updated']), str(i['symbol']), str(i['quote']['USD']['volume_24h']), str(i['quote']['USD']['volume_change_24h']), str(i['quote']['USD']['percent_change_1h']), str(i['quote']['USD']['percent_change_24h']), str(i['quote']['USD']['percent_change_7d']), str(i['quote']['USD']['percent_change_30d']), str(i['quote']['USD']['percent_change_60d']), str(i['quote']['USD']['percent_change_90d']), str(i['quote']['USD']['market_cap']), str(i['quote']['USD']['market_cap_dominance']), str(i['cmc_rank']), str(fetch_id))
            now = datetime.datetime.now()
            now = now.strftime('%Y-%m-%d %H:%M:%S')
            print(now)

            #insert.log(now)

        except requests.exceptions.ConnectionError:
            time.sleep(3)
