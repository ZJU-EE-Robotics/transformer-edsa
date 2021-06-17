import os
from tacotron_cleaner.cleaners import custom_english_cleaners
from g2p_en import G2p
from tqdm import tqdm


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


def clean_custom(dummy_id, trans_type, char2index, phn2index):
    g2p = G2p()
    cur_dir = os.path.dirname(__file__)
    filelists_path = os.path.join(cur_dir, "../filelists")

    f_read = open(os.path.join(cur_dir, "custom_data.txt"), "r", encoding="utf-8")
    f_write = open(
        os.path.join(filelists_path, "custom_set.csv"), "w", encoding="utf-8"
    )

    for line in tqdm(f_read, desc="cleaning and nomalizing: "):
        content = line.strip()
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
                dummy_id
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
                dummy_id
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


if __name__ == "__main__":
    # give a dummpy utterence id for uniform dataloader
    dummpy_id = "LJ050-0037"
    trans_type = "phn"

    # create the char and phn dict
    char2index, phn2index = init_dict()

    # create the custom_set.csv in filelists
    clean_custom(dummpy_id, trans_type, char2index, phn2index)
