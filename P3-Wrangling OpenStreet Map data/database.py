import csv, sqlite3

c = sqlite3.connect("clean_ipswich.db")
cur = c.cursor()
cur.execute("CREATE TABLE nodes (id, lat, lon, user, uid, version, changeset, timestamp);") 

with open('nodes.csv','rb') as fin: 
    reader = csv.DictReader(fin) 
    to_db = [(i['id'].decode("utf-8"), i['lat'].decode("utf-8"), i['lon'].decode("utf-8"), i['user'].decode("utf-8"), i['uid'].decode("utf-8"), i['version'].decode("utf-8"), i['changeset'].decode("utf-8"), i['timestamp'].decode("utf-8")) for i in reader]

cur.executemany("INSERT INTO nodes (id, lat, lon, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", to_db)
c.commit()


#### TABLE nodes_tags ############

cur.execute("CREATE TABLE nodes_tags (id, key, value, type);")

with open('nodes_tags.csv','rb') as fin: 
    reader = csv.DictReader(fin) 
    to_db = [(i['id'].decode("utf-8"), i['key'].decode("utf-8"), i['value'].decode("utf-8"), i['type'].decode("utf-8")) for i in reader]

cur.executemany("INSERT INTO nodes_tags (id, key, value, type) VALUES (?, ?, ?, ?);", to_db)
c.commit()


#### TABLE ways ############


cur.execute("CREATE TABLE ways (id, user, uid, version, changeset, timestamp);")

with open('ways.csv','rb') as fin: 
    reader = csv.DictReader(fin) 
    to_db = [(i['id'].decode("utf-8"), i['user'].decode("utf-8"), i['uid'].decode("utf-8"), i['version'].decode("utf-8"), i['changeset'].decode("utf-8"), i['timestamp'].decode("utf-8")) for i in reader]

cur.executemany("INSERT INTO ways (id, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?);", to_db)
c.commit()

######## Table ways_nodes ###

cur.execute("CREATE TABLE ways_nodes (id, node_id, position);")
with open('ways_nodes.csv','rb') as fin: 
    reader = csv.DictReader(fin) 
    to_db = [(i['id'].decode("utf-8"), i['node_id'].decode("utf-8"), i['position'].decode("utf-8")) for i in reader]

cur.executemany("INSERT INTO ways_nodes (id, node_id, position) VALUES (?, ?, ?);", to_db)
c.commit()

############ Table ways_tags #######
cur.execute("CREATE TABLE ways_tags (id, key, value, type);") 

with open('ways_tags.csv','rb') as fin: 
    reader = csv.DictReader(fin) 
    to_db = [(i['id'].decode("utf-8"), i['key'].decode("utf-8"), i['value'].decode("utf-8"), i['type'].decode("utf-8")) for i in reader]

cur.executemany("INSERT INTO ways_tags (id, key, value, type) VALUES (?, ?, ?, ?);", to_db)
c.commit()
c.close()

