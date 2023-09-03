def read_file(filename):
    with open(filename, 'r') as txt_file:
        data = txt_file.read()
    return data

def save_file(data, filename):
    with open(filename, 'a') as txt_file:
        txt_file.write("\n")
        if isinstance(data, dict):
            for key, value in data.items():
                txt_file.write(f"{key}: {value}\n")
        else:
            txt_file.write(str(data))
