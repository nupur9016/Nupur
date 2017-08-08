import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "ipswich.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)





mapping = { "Ln": "Lane",
            "Rd":"Road",
            "Rd.":"Road",
            "Ave":"Avenue",
           
            }




def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")



def update_name(name, mapping):
    m=street_type_re.search(name)
    if m:
        street_type=m.group()
        if street_type in mapping.keys():
            name=re.sub(street_type,mapping[street_type],name)
    return name

def clean_file(osmfile,newfile):
    osm_file = open(osmfile, "r")
    clean_file= open(newfile,"w")
    clean_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    clean_file.write('<osm>\n  ')
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    street_name=tag.attrib["v"]
                    clean_name=update_name(street_name,mapping)
                    tag.attrib["v"]=clean_name
        clean_file.write(ET.tostring(elem, encoding='utf-8'))
        
    clean_file.write('</osm>')     
    

    
clean_file("ipswich.osm","clean_ipswich.osm")       






