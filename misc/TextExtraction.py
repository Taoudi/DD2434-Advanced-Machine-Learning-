from bs4 import BeautifulSoup
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
current_path = dir_path + '/reuters21578'
directory = os.fsencode(current_path)


def extract_text(file_name):
    path = 'reuters21578/' + file_name
    f = open(path, 'r')
    data = f.read()
    soup = BeautifulSoup(data)
    contents = soup.findAll('body')
    for content in contents:
        print(content.text)


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".sgm"):
        extract_text(filename)
