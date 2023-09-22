import random
import itertools

import numpy as np
from scipy.spatial.distance import hamming

# def hamming_set(num_permutations, num_crops, selection):
def hamming_set():
    perm_list = list(itertools.permutations(range(9)))
    perm_arr = np.array(perm_list)
    perm_len = len(perm_list)


    set_of_taken = set()
    itr = 100
    count_itr = 0
    while True:
        count_itr += 1
        x = random.randint(1, perm_len-1)
        y = random.randint(1, perm_len-1)
        perm_1 = perm_arr[x]
        perm_2 = perm_arr[y]
        hd = hamming(perm_1, perm_2)

        # hd > 0.9 is not legit na
        if (hd > 0.9) and (not x in set_of_taken) and (not y in set_of_taken):
            set_of_taken.add(x)
            set_of_taken.add(y)

            # print("Itr: {}; perm_1: {} and perm_2: {}".format(count_itr,perm_1,perm_2))
            # print("hamming distance:", hd)

            if len(set_of_taken) == itr:
                break
        
    # build the array for selected permutation indices
    selected_perm = []
    for idx, perm_id in enumerate(set_of_taken):
        selected_perm.append(perm_arr[perm_id])
    selected_perm = np.array(selected_perm)

    return selected_perm


    # perm_list = list(itertools.permutations(list(range(num_crops)), num_crops))
    # perm_arr = np.array(perm_list)
    # perm_len = perm_arr.shape[0]
    #
    # for i in range(num_permutations):
    #     if i == 0:
    #         j = np.random.randint(perm_len)
    #         perm = np.array(perm_arr[j])
    #     else:
    #         perm = np.concatenate([perm, perm_arr[j]])
    #
    #     perm_arr = np.delete(perm_arr, j)
    #     hd = hamming(perm, perm_arr)
    #
    #     if selection == "max":
    #         j = hd.argmax()
    #     elif selection == "mean":
    #         m = int(hd.shape[0]/2)
    #         s = hd.argsort()
    #         j = s[np.random.randint(m-10, m+10)]
    #
    # print(j)
    # return j

    # for item in len(perm_arr):
    #     ran_perm = random.choice(perm_arr) and not ran_perm == perm_arr[item]
    #     hd = hamming(perm_arr[item], ran_perm)
    #     pass

    # if __name__ == "__main__":
    #     hamming_set(num_crops=9,
    #                 num_permutations=100,
    #                 selection="max")