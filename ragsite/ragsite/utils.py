import os


def create_file_storage_dir(given_path=None):
    current_directory = given_path if given_path else os.getcwd()
    file_directory = os.path.join(current_directory, r'new_folder')
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    return current_directory
