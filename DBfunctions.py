import sys
import mariadb
import mysql.connector
import json

# takes mysql text and a tuple of the user input values and returns the results in cursor format. 
def sql_execute(text, user_input, result_type):
    try:
        # conn = mariadb.connect(
        #     user="carruffsite",
        #     host="127.0.0.1",
        #     port=3306,
        #     database="carruffdb"
        # )
        conn = mariadb.connect(
            user="nate",
            host="23.254.144.125",
            password='hiatus32',
            port=3306,
            database="tlom"
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB platform: {e}")
        sys.exit(1)

    # Get cursor
    cur = conn.cursor()

    if user_input and result_type:
        cur.execute(text, user_input)
    elif not user_input:
        cur.execute(text)

    if result_type == 'table':
        result_fetched = cur.fetchall()
        columns = [c[0] for c in cur.description]
        cur.close()
        conn.close()
        return result_fetched, columns
    elif result_type == 'json':
        result_fetched = cur.fetchall()
        columns = [c[0] for c in cur.description]
        result_json = []
        for result in result_fetched:
            result_json.append(dict(zip(columns,result)))
        cur.close()
        conn.close()
        return result_json
    elif result_type == 'updatedb':
        conn.commit()
        cur.close()
        conn.close()
        return




def sql_execute_pd(text, user_input):
    try:
        conn = mysql.connector.connect(
            user="nate",
            host="23.254.144.125",
            port=3306,
            database="tlom"
        )
    except mysql.connector.Error as e:
        print(f"Error connecting to MariaDB platform: {e}")
        sys.exit(1)

    # Get cursor
    cur = conn.cursor()

    if user_input:
        cur.execute(text, user_input)
    else:
        cur.execute(text)

    conn.close()

    return cur