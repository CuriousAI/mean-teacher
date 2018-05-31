# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import collections
import re


class DictFormatter:
    """Format dictionaries into strings

    The (key, value) pairs in dictionary are ordered and formatted
    based on the regex patterns of the key. Use format_dict method
    to actually return the string for a dict.

    Args:
        order (array of Regexes): The order of the keys in the resulting string.
            All the pairs whose key match the first regex are output first
            (in arbitrary order), then all the pairs whose key matches the second regex
            (but not the first), and so on. If a key of a pair does not match any of the
            regexes, the pair is not included in the output.
        default_format (string): How each of the pairs will be formatted in the output.
            Can be overriden based on key with add_format method.
        separator (string): How the pairs are combined in the output.
    """

    def __init__(self, order=None, default_format='{name}: {value}', separator=", "):
        self.default_format = default_format
        self.formats = []
        self.order = order or [".+"]
        self.separator = separator

    def add_format(self, name_regex, string_format):
        """Add format for all the keys that match name_regex.

        When format_dict is called, if multiple formats match a key,
        the first one is used. If no format matches a key,
        the default format is used.
        """
        self.formats.append((name_regex, string_format))

    def format_dict(self, dictionary):
        """Return formatted string presentation of the dictionary"""
        name_order = [
            name
            for order_regex in self.order
            for name in dictionary.keys()
            if re.search(order_regex, name)
            ]
        name_order = uniq(name_order)
        strings = [self._format_single(name, dictionary[name]) for name in name_order]
        return self.separator.join(strings)

    def _format_single(self, name, value):
        for name_regex, string_format in self.formats:
            if re.search(name_regex, name):
                return string_format.format(name=name, value=value)
        return self.default_format.format(name=name, value=value)


def uniq(lst):
    return collections.OrderedDict(zip(lst, lst)).keys()
