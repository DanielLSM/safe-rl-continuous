import pickle as pkl


def save_pickle(obj, name):
    with open("data/" + name + ".pkl", 'wb') as handle:
        pkl.dump(obj, handle, protocol=pkl.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open("data/" + name, 'rb') as handle:
        return pkl.load(handle)


if __name__ == "__main__":
    a = [1, 2, 3, 3, 4]
    b = {"data": [1, 2, 3, 4]}
    save_pickle(a, "test_list")
    save_pickle(b, "test_dict")

    test_list = load_pickle("test_list.pkl")
    test_dict = load_pickle("test_dict.pkl")

    print(test_list)
    print(test_dict)
    print(load_pickle("default.pkl"))
