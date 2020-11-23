from util.seeg_utils import *

a = [1, 2, 3]
b = [2, 3, 4]
c = [3, 4, 5]
for x, y, z in zip(a, b, c):
    print(x, y, z)

# path = "/data/yh/Python/SleepState/dataset/insomnia/ins1.edf"
# data = read_edf_raw(path)
# channels = get_channels_names(data)
# print(channels)

# data_cho = select_channel_data_mne(data, "")
