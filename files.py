def read_file(filename):
    with open(filename, 'r') as txt_file:
        data = txt_file.read()
    return data

def save_file(data, filename, mode='w'):
    with open(filename, mode) as txt_file:
        if isinstance(data, dict):
            for key, value in data.items():
                txt_file.write(f"{key}: {value}\n")
        else:
            txt_file.write(str(data))
