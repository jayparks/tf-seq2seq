
import sys
import numpy as np

def main(argv):
    for input_file in argv:
        lengths = []
        with open(input_file, 'r') as corpus:
            for line in corpus:
                lengths.append(len(line.split()))
            print("%s: size=%d, avg_length=%.2f, std=%.2f, min=%d, max=%d" 
              % (input_file, len(lengths), np.mean(lengths), np.std(lengths), np.min(lengths), np.max(lengths)))


if __name__ == "__main__":
    main(sys.argv[1:])
