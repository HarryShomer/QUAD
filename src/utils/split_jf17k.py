"""
Split JF17K into 3 splits based on Qualifier% like WD50K.

Output JF17K - 33%, 66%, 100%
"""
import os
import random
from pathlib import Path

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARSED_DATA_DIR = Path(os.path.join(FILE_DIR, "..", "..", 'data/parsed_data/'))


def load_jf17K():
    """
    Load complete dataset
    """
    train_statements = []
    test_statements = []

    with open(PARSED_DATA_DIR / 'jf17k/train.txt', 'r') as train_file:
        for line in train_file:
            train_statements.append(line.strip("\n").split(","))

    with open(PARSED_DATA_DIR / 'jf17k/test.txt', 'r') as test_file:
        for line in test_file:
            test_statements.append(line.strip("\n").split(","))

    return train_statements + test_statements


def save_split(stmts, sample_perc):
    """
    Save data in data/parsed_data/jf17k_{sample_perc}

    Split about same as standard JF17K which is a 75/25 train/test split
    """
    # Randomly split into train and test set
    random.shuffle(stmts)
    num_train = int(len(stmts) * .75)
    train_data = stmts[:num_train]
    test_data = stmts[num_train:]

    # Make cleaner name
    if sample_perc == 1: 
        data_dir = PARSED_DATA_DIR / 'jf17k_100'
    elif sample_perc == .66:
        data_dir = PARSED_DATA_DIR / 'jf17k_66'
    elif sample_perc == .33:
        data_dir = PARSED_DATA_DIR / 'jf17k_33'

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    with open(data_dir / 'train.txt', 'w') as f:
        train_data = [",".join(s) + "\n" for s in train_data]
        f.writelines(train_data) 

    with open(data_dir / 'test.txt', 'w') as f:
        test_data = [",".join(s) + "\n" for s in test_data]
        f.writelines(test_data) 



def sample_jf17k(stmts, sample_perc):
    """
    Use either qualifier or non-qialifier statement as anchor and adjust other accordingly
    """
    qual_stmts = [t for t in stmts if len(t) > 3]
    non_qual_stmts = [t for t in stmts if len(t) <= 3]
    qual_perc = len(qual_stmts) / len(stmts)

    # No need to continue here
    if sample_perc == 1:
        return qual_stmts
    
    # When greater we reduce number of non-qualifier stmts
    # otherwise reduce number of qualifiers
    if sample_perc >= qual_perc:
        num_non_qual_sample = int(len(qual_stmts) / sample_perc) - len(qual_stmts)
        non_qual_stmts = random.sample(non_qual_stmts, num_non_qual_sample)
    else:
        num_qual_sample = int(len(non_qual_stmts) * sample_perc / (1 - sample_perc))
        qual_stmts = random.sample(qual_stmts, num_qual_sample)   

    return qual_stmts + non_qual_stmts



def main():
    stmts = load_jf17K()
    random.shuffle(stmts)

    for p in [.33, .66, 1]:
        sampled_stmts = sample_jf17k(stmts, p)
        save_split(sampled_stmts, p)





if __name__ == "__main__":
    main()
