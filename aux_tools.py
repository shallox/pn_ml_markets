import yaml
import logging
from datetime import datetime
import pickle
import os
import zipfile
import platform


def logging_config(message, level):
    """
    Log a message to file and prints it to console if required.
    Key:
        0: INFO
        1: DEBUG
        2: WARNING
        3: ERROR
        4: CRITICAL

    :param message: str
    :param level: int
    :return: none
    """

    config_cache = config_parser(['settings', 'dev_mode'],
                                 os.path.join(os.path.realpath(__file__).rsplit("\\", 1)[0].rsplit("/", 1)[0],
                                              'settings.yaml'))
    print_out = config_cache['console_out']['data']
    print_out_level = config_cache['console_print_out']['data']
    log_level = config_cache['log_level']['data']
    if level == 0:
        logger.info(message)
        level_str = 'INFO'
    elif level == 1:
        logger.debug(message)
        level_str = 'DEBUG'
    elif level == 2:
        logger.warning(message)
        level_str = 'WARNING'
    elif level == 3:
        logger.error(message)
        level_str = 'ERROR'
    elif level == 4:
        logger.critical(message)
        level_str = 'CRITICAL'
    if print_out == 'True' and int(print_out_level) <= level:
        if level >= log_level:
            print_log(message, level_str)


def config_parser(input_data, file_path, **kwargs):
    """
    This parser seeks to provide a single data set from within any subset of a yaml structure,
    passing a list in series (top down) in order to navigate to the data point you require.

    Alternatively, you can extract top level data structures by providing a string.

    Note:
        Use 'base_of_yaml=True' in order to get the base of any config you pass.

    :param file_path: File path to Yml file.
    :type file_path: str
    :param input_data: str, list
    :return: any
    """
    try:
        with open(file_path, 'r') as yaml_data:
            config_dict = dict(
                yaml.safe_load(
                    yaml_data
                )
            )
            if 'base_of_yaml' in kwargs.keys():
                if kwargs['base_of_yaml']:
                    return config_dict
            if type(input_data) is list:
                cache_dict = {}
                tik = 0
                for key in input_data:
                    if tik == 0:
                        cache_dict.update(
                            {
                                'return_data': config_dict[key]
                            }
                        )
                        tik += 1
                    else:
                        cache_dict.update(
                            {
                                'return_data': cache_dict['return_data'][key]
                            }
                        )
                        tik += 1
                return cache_dict['return_data']
            else:
                return config_dict[input_data]
    except KeyError as key_error:
        logging_config(
            f'Key missing: {key_error} | Definition: config_parser | File: {file_path}',
            3
        )
    except FileNotFoundError as file_notfound_error:
        logging_config(
            file_notfound_error,
            3
        )


def print_log(data, level):
    """
    Prints to std out, log entries and other info.


    :param data: information relating to log.
    :type data: str
    :param level: Log level, takes any string info.
    :type level: any
    :return: None
    :rtype: None
    """
    print(f'@{datetime.now()} - {level} - {data}')


def pickler(mode, file_path, byte_data):
    """
    Creates a pickle file in the desired destination and or reads one.

    :param mode: 0 = write pickle file, 1 = read pickle file
    :type mode: int
    :param file_path: file path of pickle file to save or load
    :type file_path: str
    :param byte_data: data to be pickled if any (use None if mode 1 is selected)
    :type byte_data: any
    :return: either done as string or
    :rtype: str, json
    """
    if mode == 0:
        with open(file_path, 'wb') as pickle_meta:
            # Use pickle.dump() to pickle the dictionary and write it to the file
            pickle.dump(byte_data, pickle_meta)
            return 'done'
    if mode == 1:
        with open(file_path, 'rb') as pickle_meta:
            # Use pickle.load() to unpickle the dictionary and read it from the file
            return pickle.load(pickle_meta)


def move_file(target_file, destination):
    """
    In this definition we take a target file as well as a destination dir.
    With this information, the target file is moved into the destination dir.

    :param target_file: Complete file path
    :type target_file: str
    :param destination: Complete folder path
    :type destination: str
    :return: Nothing
    :rtype: None
    """

    file_name = target_file.split('/')
    file_name = file_name[(len(file_name) - 1)]
    logging_config(f'Moving {file_name} into {destination}', 0)
    target_file_path = os.path.join(destination, file_name)
    try:
        os.renames(target_file, target_file_path)
    except FileExistsError as fee:
        logging_config(fee, 3)
    except FileNotFoundError as fnfe:
        logging_config(fnfe, 3)


def zippy(mode, **kwargs):
    """
    Zip and unzip files/folders.
    Modes: 0 = unzip, 1 = zip

    :param mode:
    :type mode:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """

    if mode == 0:
        file_to_un_zip = kwargs['file_to_un_zip']
        logging_config(f'Starting extraction of {file_to_un_zip}', 0)
        zip_ref = zipfile.ZipFile(file_to_un_zip, 'r')
        zip_ref.extractall(kwargs['extraction_location'])
        zip_ref.close()
        logging_config(f'Finished extraction of {file_to_un_zip}', 0)


def check_os_type():
    """
    Checkes OS type and returns list with 2 objects, type of os at position 0 and file path seperator at position 1

    :return: Checkes OS type and returns list with 2 objects, type of os at position 0 and file path seperator at position 1
    :rtype: list
    """
    os_type_check = platform.system()
    if os_type_check == 'Windows':
        split_check = '\\'
    else:
        split_check = '/'
    return [os_type_check, split_check]


### Logging config ###
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
    filename=os.path.join(str(os.path.realpath(__file__)).rsplit(check_os_type()[1], 1)[0],
                          f"logs{check_os_type()[1]}runtime.log"))
logger = logging.getLogger()
