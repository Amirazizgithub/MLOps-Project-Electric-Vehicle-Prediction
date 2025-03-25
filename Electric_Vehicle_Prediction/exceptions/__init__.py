# Path: Electric_Vehicle_Prediction/exceptions/__init__.py

import os
import sys


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    file = os.path.basename(file_name)
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file, exc_tb.tb_lineno, str(error)
    )

    return error_message


class EV_Exception(Exception):
    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message
