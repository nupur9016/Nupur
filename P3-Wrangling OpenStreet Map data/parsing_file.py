import xml.etree.cElementTree as ET
import pprint

def count_tags(filename):
    tags={}
    
    for event,elem in ET.iterparse(filename):
       if elem.tag not in tags.keys():
           tags[elem.tag]=1
       else:
         
           tags[elem.tag] +=1
    return tags
tags=count_tags("ipswich.osm")
pprint.pprint(tags)
