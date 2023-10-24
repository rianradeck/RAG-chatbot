from langchain.document_loaders import WebBaseLoader
import urllib.request
from pprint import pprint
from parser import HTMLTableParser

import sys

orig_stdout = sys.stdout
f = open('out.txt', 'w', encoding='utf-8')
sys.stdout = f

def url_get_contents(url: str) -> bytes:
    """ Opens a website and read its binary contents (HTTP Response Body) """
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)
    return f.read()

url = 'https://www.pg.unicamp.br/norma/31594/0'
xhtml = url_get_contents(url).decode('utf-8')

p = HTMLTableParser()
p.feed(xhtml)

# Get all tables
final_tables = []
table_idx = 0
for table in p.tables:
    if len(table[0]) == 1:
        continue

    for row in table:
        if table_idx == 0 and len(row) == 7:
            prep = "No" if row[0] == "Total" else "Em"
            print(f"{prep} {row[0]} existem {row[1]} vagas regulares e {row[2]} vagas pelo vestibular Unicamp (VU)")
            print(f"{prep} {row[0]} as {row[2]} vagas do VU são divididas da seguinte maneira:",
                  f" para ampla concorrência algo entre {row[3]} (valor mínimo) e {row[4]} (valor máximo) e",
                  f" para vagas reservadas para cotas étnico-raciais (pessoas pretas e pardas) algo entre {row[5]} (valor mínimo) e {row[6]} (valor máximo).", sep='')

    table_idx += 1

    
# print(final_tables)

loader = WebBaseLoader(url)
loader.default_parser = "xml"
docs = loader.load()[0].page_content

# index1 = docs[0].index('ANEXO I')
index1 = docs.index("ANEXO I")
index2 = docs.index("ANEXO II")

docs = docs[:index1 + len("ANEXO I")] + docs[index2:]

# index3 = docs.index("RELAÇÃO DE LIVROS")
# index4 = docs.index("dominiopublico.gov.br")

# docs = docs[:index3 + len("RELAÇÃO DE LIVROS")] + docs[index4:]

# index5 = docs.index("ANEXO III")
# index6 = docs.index("ANEXO IV")

# docs = docs[:index5 + len("ANEXO III")] + docs[index6:]

for i in range(len(docs)):
    if i > 0 and docs[i] == '\n' and docs[i-1] == '\n':
        continue
    print(docs[i], end='')

sys.stdout = orig_stdout
f.close()
