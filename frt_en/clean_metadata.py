import os
from tacotron_cleaner.cleaners import custom_english_cleaners
from g2p_en import G2p
from tqdm import tqdm
import random


def init_dict():
    char2index = {}
    phn2index = {}
    dict_path = os.path.dirname(__file__)

    with open(os.path.join(dict_path, "char_dict.txt"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            char, index = line.split(" ")
            char2index[char] = index

    with open(os.path.join(dict_path, "phn_dict.txt"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            phn, index = line.split(" ")
            phn2index[phn] = index
    return char2index, phn2index


def clean_ljspeech(metadata, trans_type, char2index, phn2index):
    g2p = G2p()
    cur_dir = os.path.dirname(__file__)
    filelists_path = os.path.join(cur_dir, "../filelists")

    f_read = open(metadata, "r", encoding="utf-8")
    f_write = open(os.path.join(filelists_path, "data.csv"), "w", encoding="utf-8")

    for line in tqdm(f_read, desc="cleaning and nomalizing: "):
        utterence_id, _, content = line.split("|")
        content = content.strip()
        clean_char = custom_english_cleaners(content)

        if trans_type == "char":
            normalized_char = []
            token_id = []
            for char in clean_char:
                if char in char2index.keys():
                    normalized_char.append(char)
                    token_id.append(char2index[char])
                elif char == " ":
                    normalized_char.append("<space>")
                    token_id.append(char2index["<space>"])
                else:
                    normalized_char.append("<unk>")
                    token_id.append(char2index["<unk>"])
            normalized_char.append("<eos>")
            token_id.append(char2index["<eos>"])

            normalized_char = " ".join(normalized_char)
            token_id = " ".join(token_id)
            f_write.write(
                utterence_id
                + "|"
                + content
                + "|"
                + normalized_char
                + "|"
                + token_id
                + "\n"
            )
        elif trans_type == "phn":
            clean_char = clean_char.lower()
            clean_phn = g2p(clean_char)
            normalized_phn = []
            token_id = []
            for phn in clean_phn:
                if phn in phn2index:
                    normalized_phn.append(phn)
                    token_id.append(phn2index[phn])
                elif phn == " ":
                    normalized_phn.append("<space>")
                    token_id.append(phn2index["<space>"])
                else:
                    normalized_phn.append("<unk>")
                    token_id.append(phn2index["<unk>"])
            normalized_phn.append("<eos>")
            token_id.append(phn2index["<eos>"])

            normalized_phn = " ".join(normalized_phn)
            token_id = " ".join(token_id)
            f_write.write(
                utterence_id
                + "|"
                + content
                + "|"
                + normalized_phn
                + "|"
                + token_id
                + "\n"
            )
        else:
            raise Exception("Wrong Type")

    f_read.close()
    f_write.close()


def clean_blizzard17(metadata, trans_type, char2index, phn2index):
    g2p = G2p()
    cur_dir = os.path.dirname(__file__)
    filelists_path = os.path.join(cur_dir, "../filelists")

    f_read = open(metadata, "r", encoding="utf-8")
    f_write = open(os.path.join(filelists_path, "data.csv"), "w", encoding="utf-8")

    for line in tqdm(f_read, desc="cleaning and nomalizing: "):
        line = line.strip("(")
        line = line.strip(")\n")
        utterence_id, content, _ = line.split("\"")
        utterence_id = utterence_id.strip()
        content = content.strip()
        clean_char = custom_english_cleaners(content)

        if trans_type == "char":
            normalized_char = []
            token_id = []
            for char in clean_char:
                if char in char2index.keys():
                    normalized_char.append(char)
                    token_id.append(char2index[char])
                elif char == " ":
                    normalized_char.append("<space>")
                    token_id.append(char2index["<space>"])
                else:
                    normalized_char.append("<unk>")
                    token_id.append(char2index["<unk>"])
            normalized_char.append("<eos>")
            token_id.append(char2index["<eos>"])

            normalized_char = " ".join(normalized_char)
            token_id = " ".join(token_id)
            f_write.write(
                utterence_id
                + "|"
                + content
                + "|"
                + normalized_char
                + "|"
                + token_id
                + "\n"
            )
        elif trans_type == "phn":
            clean_char = clean_char.lower()
            clean_phn = g2p(clean_char)
            normalized_phn = []
            token_id = []
            for phn in clean_phn:
                if phn in phn2index:
                    normalized_phn.append(phn)
                    token_id.append(phn2index[phn])
                elif phn == " ":
                    normalized_phn.append("<space>")
                    token_id.append(phn2index["<space>"])
                else:
                    normalized_phn.append("<unk>")
                    token_id.append(phn2index["<unk>"])
            normalized_phn.append("<eos>")
            token_id.append(phn2index["<eos>"])

            normalized_phn = " ".join(normalized_phn)
            token_id = " ".join(token_id)
            f_write.write(
                utterence_id
                + "|"
                + content
                + "|"
                + normalized_phn
                + "|"
                + token_id
                + "\n"
            )
        else:
            raise Exception("Wrong Type")

    f_read.close()
    f_write.close()


def make_subsets():
    cur_dir = os.path.dirname(__file__)
    filelists_path = os.path.join(cur_dir, "../filelists")
    data_path = os.path.join(filelists_path, "data.csv")
    train_path = os.path.join(filelists_path, "train_set.csv")
    dev_path = os.path.join(filelists_path, "dev_set.csv")
    test_path = os.path.join(filelists_path, "test_set.csv")
    long_path = os.path.join(filelists_path, "long_set.csv")

    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line for line in f]

    lines = sorted(lines, key=lambda x: len(x.split("|")[-1]))
    lines_domain = lines[:-100]
    lines_outdom = lines[-100:]

    random.seed(0)
    random.shuffle(lines_domain)
    train_lines = lines_domain[:-500]
    dev_lines = lines_domain[-500:-250]
    test_lines = lines_domain[-250:]
    long_lines = lines_outdom

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)

    with open(dev_path, "w", encoding="utf-8") as f:
        for line in dev_lines:
            f.write(line)

    with open(test_path, "w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line)

    with open(long_path, "w", encoding="utf-8") as f:
        for line in long_lines:
            f.write(line)
    print("All the subsets have been prepared")


if __name__ == "__main__":
    # metadata path and transcript type
    metadata = "/home/server/disk1/DATA/LJS/LJSpeech-1.1/metadata.csv"
    trans_type = "phn"

    # create the char and phn dict
    char2index, phn2index = init_dict()

    # create the data.csv in filelists
    clean_ljspeech(metadata, trans_type, char2index, phn2index)

    # split the data.csv into train, dev, and test
    make_subsets()
