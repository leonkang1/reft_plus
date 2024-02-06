from colorama import Style
import logging

class nlogger:
    def __init__(self, filename=None):
        logging.basicConfig(filename=filename, level=logging.DEBUG)
        self.output_type_list = [
            "UNCATE"
        ]

    def nprint(self, text, color=Style.RESET_ALL, print_type="UNCATE"):
        if print_type in self.output_type_list:
            logging.debug(color + text + Style.RESET_ALL)
