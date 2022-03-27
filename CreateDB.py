import sqlite3

conn = sqlite3.connect('image2.db')
cursor = conn.cursor()

cursor.execute("""
        CREATE TABLE IF NOT EXISTS my_table 
        (Time TEXT,Image BLOB,Objects INTEGER)""")

conn.commit()
cursor.close()
conn.close()

