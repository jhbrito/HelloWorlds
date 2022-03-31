# get("https://v3.football.api-sports.io/fixtures?date=2019-10-22");
import os
import pickle
import http.client

data_file_name = 'football_api_data_2022-04-12.pickle'
data_txt_file_name = 'football_api_data_2022-04-12.txt'

decoded_data_file_name = 'football_api_decoded_data_2022-04-12.pickle'
decoded_data_txt_file_name = 'football_api_decoded_data_2022-04-12.txt'

if os.path.exists(data_file_name):
    with open(data_file_name, 'rb') as f:
        data = pickle.load(f)
    with open(data_txt_file_name, "r") as f:
        data_text = f.read()
else:
    with open("key.txt", "r") as data_file:
        api_key = data_file.read()
    print(api_key)

    headers = {
        'x-rapidapi-host': "v3.football.api-sports.io",
        'x-rapidapi-key': api_key}
    print(headers)

    import requests
    api_url = 'https://v3.football.api-sports.io/fixtures?date=2022-04-12'
    data = requests.get(api_url, headers=headers)

    # conn = http.client.HTTPSConnection("v3.football.api-sports.io")
    # conn.request("GET", "/fixtures?live=all", headers=headers)
    # conn.request("GET", "/fixtures?date=2022-04-05", headers=headers)
    # res = conn.getresponse()
    # data = res.read()

    print(data)
    with open(data_file_name, 'wb') as f:
        pickle.dump(data, f)
    with open(data_txt_file_name, "w") as f:
        f.write(data.text)

    # decoded_data = data.decode("utf-8")
    # with open(decoded_data_file_name, 'wb') as f:
    #     pickle.dump(decoded_data, f)
    #
    # with open(decoded_data_txt_file_name, "w") as data_file:
    #     data_file.write(decoded_data)

print(data)
print(data.text)
print(data_text)
