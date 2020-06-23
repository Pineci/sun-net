'''
This file contains useful helper functions that could be used in multiple different package files.

This file contains the functions:
    * pad_string - adds characters to a string to a certain length
    * get_date_strings - converts integer date and time to string format
'''


def pad_string(s, pad='0', length=2):
    '''Adds a padding string to the front of an input string until a desired length is reached.
    The padding string is added in complete segments, so it is possible that the output is larger
    than the desired length. For example, pad_string('abc', '12', 6) returns '1212abc'.
    
    Parameters
    ----------
    s : str
        The string to which the pad string will be added
    pad : str, optional
        The string which will be used for padding (default = '0')
    length : int, optional
        The minimum length of the output string (default = 2)
        
    Returns
    -------
    s : str
        The formatted output string
    '''
    while len(s) < length:
        s = pad + s
    return s

def get_date_strings(year=2000, month=1, day=1, hour=0, minute=0):
    '''Converts integer values of year, month, day, hour, minute to standard
    string format. Each resultant string must have length 2 (with leading 0's
    where necessary) except for year which has length 4. The program does not
    check whether the specified date is valid.
    
    Parameters
    ----------
    year : int, optional
        The year as an integer. Should be 4 digits long (not necessary right 
        now to have years below 1000 or above 9999...), (default = 2000)
    month : int, optional
        The month as an integer. Should be a value between 1 and 12 inclusive (default = 1)
    day : int, optional
        The day as an integer. Should be a value between 1 and 31 inclusive (default = 1)
    hour : int, optional
        The hour as an integer. Should be a value between 0 and 23 inclusive (default = 0)
    minute : int, optional
        The minute as an integer. Should be a value between 0 and 59 inclusive (default = 0)
        
    Returns
    -------
    yr_str : str
        The year as a 4 digit string
    month_str : str
        The month as a 2 digit string, padded with leading 0's
    day_str : str
        The day as a 2 digit string, padded with leading 0's
    hour_str : str
        The hour as a 2 digit string, padded with leading 0's
    minute_str : str
        The minute as a 2 digit string, padded with leading 0's
    '''
    yr_str = str(year)
    month_str = pad_string(str(month), length=2)
    day_str = pad_string(str(day), length=2)
    hour_str = pad_string(str(hour), length=2)
    minute_str = pad_string(str(minute), length=2)
    return yr_str, month_str, day_str, hour_str, minute_str