import pandas as pd
import numpy as np
import streamlit as st

class InputData:
    def __init__(self, text: str, pr: float, size: int):
        self.text = text
        self._pr = pr
        self.size = size
        self.content = None
        self.sim_sum = None

    @property
    def pr(self):
        return self._pr

    @pr.setter
    def pr(self, value):
        if value in np.arange(0, 1, .01):
            self._pr = value

        raise ValueError("Page rank should be between 0 and 1")
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
