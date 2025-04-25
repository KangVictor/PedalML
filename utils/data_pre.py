from os import listdir
from os.path import isfile, join

def match_list_lengths(list1, list2):
    len1, len2 = len(list1), len(list2)
    if len1 == len2:
        return list1, list2
    elif len1 > len2:
        repeated_list2 = (list2 * (len1 // len2 + 1))[:len1]
        return list1, repeated_list2
    else:
        repeated_list1 = (list1 * (len2 // len1 + 1))[:len2]
        return repeated_list1, list2

def get_wav_file_list(dir):
    return [
        join(dir, f)
        for f in listdir(dir)
        if isfile(join(dir, f)) and f.lower().endswith(".wav")
    ]