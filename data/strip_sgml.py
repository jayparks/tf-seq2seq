import sys
import re

'''
This code comes from strip_sgml.py of
nematus proejct (https://github.com/rsennrich/nematus)
'''

def main():
    fin = sys.stdin
    fout = sys.stdout
    for l in fin:
        line = l.strip()
        text = re.sub('<[^<]+>', "", line).strip()
        if len(text) == 0:
            continue
        print >>fout, text
                

if __name__ == "__main__":
    main()

