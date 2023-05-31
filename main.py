import time
from twitter import twitterapi
from price import api2
from price import insert1_12 as insert
from price import config
import mariadb
def dbcon():
    # Verbindung zur Datenbank herstellen
    conn = mariadb.connect(user=config.duser, password=config.dpassword, host=config.dhost, database=config.ddatabase)
    cur = conn.cursor()
    return cur, conn

def start():
    cur, conn = dbcon()

    fetch_id = twitterapi.insertfetch(cur, conn)
    twitterapi.run(fetch_id)
    api2.run(fetch_id)
    insert.insertfetch(fetch_id)


while __name__ == "__main__":
    start()
    time.sleep(300)

