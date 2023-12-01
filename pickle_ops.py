import pickle

def pickle_write(file_path, data_to_save):
    # Writing to a file using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(data_to_save, file)

    print(f'Data has been saved to {file_path}')

def pickle_load(file_path):
    # Reading from the file using pickle
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    print('Loaded Data:', loaded_data)
    return loaded_data