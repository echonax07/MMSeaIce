#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilization resources."""

# -- File info -- #
__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

# Charts in the dataset
CHARTS = ['SIC', 'SOD', 'FLOE']

# Variables in the ASID3 challenge ready-to-train dataset
SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangle',

    # -- Geographical variables -- #
    'distance_map',

    # -- AMSR2 channels -- #
    'btemp_6_9h', 'btemp_6_9v',
    'btemp_7_3h', 'btemp_7_3v',
    'btemp_10_7h', 'btemp_10_7v',
    'btemp_18_7h', 'btemp_18_7v',
    'btemp_23_8h', 'btemp_23_8v',
    'btemp_36_5h', 'btemp_36_5v',
    'btemp_89_0h', 'btemp_89_0v',

    # -- Environmental variables -- #
    'u10m_rotated', 'v10m_rotated',
    't2m', 'skt', 'tcwv', 'tclw'

]

# Sea Ice Concentration (SIC) code to class conversion lookup table.
SIC_LOOKUP = {
    'polygon_idx': 0,  # Index of polygon number.
    'total_sic_idx': 1,  # Total Sea Ice Concentration Index, CT.
    # Partial SIC polygon code index. CA, CB, CC.
    'sic_partial_idx': [2, 5, 8],
    0: 0,
    1: 0,
    2: 0,
    55: 0,
    10: 1,  # 10 %
    20: 2,  # 20 %
    30: 3,  # 30 %
    40: 4,  # 40 %
    50: 5,  # 50 %
    60: 6,  # 60 %
    70: 7,  # 70 %
    80: 8,  # 80 %
    90: 9,  # 90 %
    91: 10,  # 100 %
    92: 10,  # Fast ice
    'mask': 255,
    'n_classes': 12
}

# Names of the SIC classes.
SIC_GROUPS = {
    0: 0,
    1: 10,
    2: 20,
    3: 30,
    4: 40,
    5: 50,
    6: 60,
    7: 70,
    8: 80,
    9: 90,
    10: 100
}

# Stage of Development code to class conversion lookup table.
SOD_LOOKUP = {
    # Partial SIC polygon code index. SA, SB, SC.
    'sod_partial_idx': [3, 6, 9],
    # < 1. Minimum partial percentage SIC of total SIC to select SOD. Otherwise ambiguous polygon.
    'threshold': 0.7,
    # larger than threshold.
    # Value for polygons where the SOD is ambiguous or not filled.
    'invalid': -9,
    'water': 0,
    0: 0,
    80: 0,  # No stage of development
    81: 1,  # New ice
    82: 1,  # Nilas, ring ice
    83: 2,  # Young ice
    84: 2,  # Grey ice
    85: 2,  # White ice
    86: 4,  # First-year ice, overall categary
    87: 3,  # Thin first-year ice
    88: 3,  # Thin first-year ice, stage 1
    89: 3,  # Thin first-year ice, stage 2
    91: 4,  # Medium first-year ice
    93: 4,  # Thick first-year ice
    95: 5,  # Old ice
    96: 5,  # Second year ice
    97: 5,  # Multi-year ice
    98: 255,  # Glacier ice
    99: 255,
    'mask': 255,
    'n_classes': 7
}

# Names of the SOD classes.
SOD_GROUPS = {
    0: 'Open water',
    1: 'New Ice',
    2: 'Young ice',
    3: 'Thin FYI',
    4: 'Thick FYI',
    5: 'Old ice',
}

# Ice floe/form code to class conversion lookup table.
FLOE_LOOKUP = {
    # Partial SIC polygon code index. FA, FB, FC.
    'floe_partial_idx': [4, 7, 10],
    # < 1. Minimum partial concentration to select floe. Otherwise polygon may be ambiguous.
    'threshold': 0.5,
    # Value for polygons where the floe is ambiguous or not filled.
    'invalid': -9,
    'water': 0,
    0: 0,
    22: 255,  # Pancake ice
    1: 255,  # Shuga / small ice cake
    2: 1,  # Ice cake
    3: 2,  # Small floe
    4: 3,  # Medium floe
    5: 4,  # Big floe
    6: 5,  # Vast fæpe
    7: 5,  # Gian floe
    8: 255,  # Fast ice
    9: 6,  # Growlers, floebergs or floebits
    10: 6,  # Icebergs
    21: 255,  # Level ice
    'fastice_class': 255,
    'mask': 255,
    'n_classes': 8
}

# Names of the FLOE classes.
FLOE_GROUPS = {
    0: 'Open water',
    1: 'Cake Ice',
    2: 'Small floe',
    3: 'Medium floe',
    4: 'Big floe',
    5: 'Vast floe',
    6: 'Bergs'
}

# Used in converting polygon codes into the preprocessed ice charts.
ICECHART_NOT_FILLED_VALUE = -9
# Used in converting polygon codes into the preprocessed ice charts.
ICECHART_UNKNOWN = 99

# Strings for each chart. Useful for plotting.
ICE_STRINGS = {
    'SIC': 'Sea Ice Concentration [%]',
    'SOD': 'Stage of Development',
    'FLOE': 'Floe Size'
}

# Wraps the lookup tables together.
LOOKUP_NAMES = {
    'SIC': SIC_LOOKUP,
    'SOD': SOD_LOOKUP,
    'FLOE': FLOE_LOOKUP
}

# Wraps the group names together.
GROUP_NAMES = {
    'SIC': SIC_GROUPS,
    'SOD': SOD_GROUPS,
    'FLOE': FLOE_GROUPS
}

# Colour dictionary
COLOURS = {'red': '\033[0;31m',
           'black': '\033[0m',
           'green': '\033[0;32m',
           'orange': '\033[0;33m',
           'purple': '\033[0;35m',
           'blue': '\033[0;34m',
           'cyan': '\033[0;36m'}


def colour_str(word, colour: str):
    """Function to colour strings."""
    return COLOURS[colour.lower()] + str(word) + COLOURS['black']


run_names = [
    'Goku', 'Naruto', 'Luffy', 'Ichigo', 'Light', 'Edward', 'Saitama', 'Levi', 'Gon', 'Eren',
    'Vegeta', 'Zoro', 'Sasuke', 'Alucard', 'Kakashi', 'Guts', 'Tetsuo', 'Kenshin', 'Monkey',
    'Gintoki', 'Spike', 'Kagome', 'Rei', 'Naruto', 'Shinya', 'Kira', 'Sesshomaru', 'Asuna',
    'Simon', 'Holo', 'Rukia', 'Yato', 'Misato', 'Mikasa', 'Haruhi', 'Roy', 'Vash', 'Mugen',
    'Jotaro', 'Vegeta', 'Killua', 'Kyon', 'Hiei', 'Sosuke', 'Frieza', 'Joseph', 'Zeref',
    'Saber', 'Yugi', 'Vivi', 'Usagi', 'Nami', 'Sakura', 'Jiraiya', 'Itachi', 'Madara', 'Rias',
    'Zoro', 'Tatsumi', 'Yuno', 'Shiro', 'Kirito', 'C.C.', 'Orihime', 'Kenshiro', 'Maka',
    'Maka', 'Gon', 'Momo', 'Natsu', 'Koro-sensei', 'Katsuki', 'Riza', 'Akira', 'Makise',
    'Kenshiro', 'Kagura', 'Faye', 'Ryuk', 'Suzaku', 'Touka', 'Lelouch', 'Homura', 'Rukako',
    'Zero', 'Inuyasha', 'Sebastian', 'Homura', 'Shinichi', 'Sinon', 'Koyomi', 'Ryuko', 'Tamaki',
    'Clare', 'Rito', 'Takumi', 'Midoriya', 'Yoh', 'Rukia', 'Kaori', 'Soma', 'Kaneki', 'Edward',
    'Gray', 'Allen', 'Sasuke', 'Kaname', 'Shizuo', 'Kagami', 'Akame', 'Ken', 'Yukino', 'Taiga',
    'Sakura', 'Rin', 'Miyuki', 'Mikoto', 'Kuroko', 'Haruka', 'Mikasa', 'Alois', 'Misa', 'Tomoya',
    'Lelouch', 'Hikaru', 'Nagisa', 'Revy', 'Haruka', 'Mugen', 'Kotomine', 'Takashi', 'Ling', 'Shinji',
    'Hikigaya', 'Allen', 'Kyo', 'Ayase', 'Rider', 'Kurama', 'Shiki', 'Shiroe', 'Asuka', 'Kaga', 'Nico',
    'Riko', 'Mirai', 'Nanoha', 'Hinata', 'Tsubasa', 'Shinra', 'Homura', 'Takanashi', 'Kenshin', 'Tsubaki',
    'Rei', 'Lan', 'Makoto', 'Kraft', 'Miyamura', 'Kaito', ]
