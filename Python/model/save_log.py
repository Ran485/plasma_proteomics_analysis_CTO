#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File     : save_log.py
@Time     : 2024/07/03 14:06:07
@Author   : RanPeng 
@Version  : python 3.8.6
@Contact  : 21112030023@m.fudan.edu.cn
@License  : (C)Copyright 2022-2023, DingLab-CHINA-SHANGHAI
@Function : None
"""

import sys
import os
import datetime

def save_print_to_file(path='./'):
    """
    Redirects print statements to a log file.
    
    Args:
        path (str): The directory path where the log file will be saved.
    
    Example:
        To use this function, call it with the desired path and all the subsequent
        print statements will be written to a log file.
        
        save_print_to_file(path='/path/to/log/directory/')
        print("This message will be logged.")
    """
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            """
            Initializes the Logger object.
            
            Args:
                filename (str): The name of the log file.
                path (str): The directory path where the log file will be saved.
            """
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')

        def write(self, message):
            """
            Writes a message to both the terminal and the log file.
            
            Args:
                message (str): The message to be logged.
            """
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            """ Flush method for the logger. """
            pass

    fileName = datetime.datetime.now().strftime('day_%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    # All subsequent print statements will be written to the log file
    print(fileName.center(60, '*'))

# Example usage:
if __name__ == '__main__':
    save_print_to_file(path='/Users/ranpeng/Desktop/Desktop/项目文件/对外服务/罗俊一/script/results/XGBoost/clinical_output_20220224/')
    print("This message will be logged.")
