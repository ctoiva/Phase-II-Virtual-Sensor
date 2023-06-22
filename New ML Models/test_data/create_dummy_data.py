import pandas as pd
import numpy as np
import os


def gen_data(base):
    """
    Function to create data for 7 sensors using a sample array.
    Calling this function also prints the minimum, maximum and the number of NaN's in each sensor array.
    :param base: Sample array based on which data is generated
    :return: A dictionary of 7 arrays
    :rtype: dict
    """
    base_2 = (3 * base ** 2 + 2 * base) / (4 * base)
    sen_dict = {}
    sen_dict["Sensor A"] = ((7 * base) / 3)**1.2
    sen_dict["Sensor B"] = ((4 * base) / 2)**1.5
    sen_dict["Sensor C"] = (12 + 3 * base) / 5
    sen_dict["Sensor D"] = ((20 + 4 * base * base_2 ** 2) / (2 * base + 3) ** 2)
    sen_dict["Sensor E"] = ((12 / (base ** 0.8 + base_2)) * (base + base_2)) ** (base**0.1) / 2
    sen_dict["Sensor F"] = ((2 * base ** 1.5 + 4) / (5 * base + 3)) ** 3.2

    for sen in sen_dict.keys():
        print(f"{sen} ------> Min = {min(sen_dict[sen]):0,.4f}\t&\t"
              f"Max = {max(sen_dict[sen]):0,.4f}\t&\t"
              f"NaN = {sum(np.isnan(sen_dict[sen]))}")

    return sen_dict


# Generating the data for the datasets
# ang = (np.linspace(0, 7200000, 7200000).transpose()) * (1/10)
ang = (np.linspace(0, 3600, 3600).transpose()) * (1 / 5)

print("<--------- NH & NL details --------->")
NH = 50 + 50 * np.cos(ang * 2 * 3.14 / 180)
NL = 45 + 45 * np.cos((ang * 2 + 18) * 3.14 / 180)
print(f"NH ------> Min = {min(NH):0,.4f}\t&\tMax = {max(NH):0,.4f}")
print(f"NL ------> Min = {min(NL):0,.4f}\t&\tMax = {max(NL):0,.4f}")

print("\n<--------- Train data details --------->")
train_base = 350 + 300 * np.sin((ang + 18 * 2) * 3.14 / 180)
print(f"train_base ------> Min = {min(train_base):0,.4f}\t&\tMax = {max(train_base):0,.4f}")
train_data = gen_data(train_base)
train_data["NH"] = NH
train_data["NL"] = NL

print("\n<--------- Test data details --------->")
test_base = 300 + 260 * np.sin((ang + 18 * 2) * 3.14 / 180)
print(f"test_base ------> Min = {min(test_base):0,.4f}\t&\tMax = {max(test_base):0,.4f}")
test_data = gen_data(test_base)
test_data["NH"] = NH
test_data["NL"] = NL

print("\n<--------- Inference data details --------->")
inf_base = 600 + 501 * np.sin((ang + 18 * 2) * 3.14 / 180)
print(f"inf_base ------> Min = {min(inf_base):0,.4f}\t&\tMax = {max(inf_base):0,.4f}")
inf_data = gen_data(inf_base)
inf_data["NH"] = NH
inf_data["NL"] = NL

out_path = r"C:\Users\USER\Desktop\Work\Virtual Sensor Enhancement\New ML models\test_data"
train_csv = "HF_train_data_1.csv"
test_csv = "HF_test_data_1.csv"
inf_csv = "HF_inf_data_1.csv"

# base relations
# base_1 = 3 * np.sin((ang + 18 * 2) * 3.14 / 180)
# base_2 = 5.2 * np.sin((ang + 18 * 2) * 3.14 / 180)
# base_3 = 7 * np.sin((ang + 18 * 2) * 3.14 / 180)
# base_4 = 9 * np.sin((ang + 18 * 2) * 3.14 / 180)

# Writing the data in csv files
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
inf_df = pd.DataFrame(inf_data)

train_df.to_csv(os.path.join(out_path, train_csv), index=False)
test_df.to_csv(os.path.join(out_path, test_csv), index=False)
inf_df.to_csv(os.path.join(out_path, inf_csv), index=False)
