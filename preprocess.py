import sys
import numpy as np

if __name__ == "__main__":
    word2idx = {}
    word2idx['SOS'] = 1
    word2idx['EOS'] = 2
    vocab_size = 3
    for line in sys.stdin:
        fs = line.strip().split()
        sequence = []
        sequence.append("1")
        for word in fs:
            id = word2idx.get(word, None)
            if id is None:
                word2idx[word] = vocab_size
                vocab_size += 1
            sequence.append(str(word2idx[word]))
        sequence.append("2")
        print ' '.join(sequence)
        
