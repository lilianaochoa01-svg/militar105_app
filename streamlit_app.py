import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np 

# --- Datos base ---
tabla_letras = [
    {"min": 2, "max": 8, "S-1": "A", "S-2": "A","S-3": "A", "S-4": "A","I": "A","II": "A", "III": "B"},
    {"min": 9, "max": 15, "S-1": "A", "S-2": "A","S-3": "A", "S-4": "A","I": "A", "II": "B", "III": "C"},
    {"min": 16, "max": 25, "S-1": "A", "S-2": "A","S-3": "B", "S-4": "B","I": "B", "II": "C", "III": "D"},
    {"min": 26, "max": 50,"S-1": "A", "S-2": "B","S-3": "B", "S-4": "C", "I": "C", "II": "D", "III": "E"},
    {"min": 51, "max": 90, "S-1": "B", "S-2": "B","S-3": "C", "S-4": "A","I": "C", "II": "E", "III": "F"},
    {"min": 91, "max": 150, "S-1": "B", "S-2": "B","S-3": "C", "S-4": "D","I": "D", "II": "F", "III": "G"},
    {"min": 151, "max": 280, "S-1": "B", "S-2": "C","S-3": "D", "S-4": "E","I": "E", "II": "G", "III": "H"},
    {"min": 281, "max": 500, "S-1": "B", "S-2": "C","S-3": "D", "S-4": "E","I": "F", "II": "H", "III": "J"},
    {"min": 501, "max": 1200, "S-1": "C", "S-2": "C","S-3": "E", "S-4": "F","I": "G", "II": "J", "III": "K"},
    {"min": 1201, "max": 3200, "S-1": "C", "S-2": "D","S-3": "E", "S-4": "G","I": "H", "II": "K", "III": "L"},
    {"min": 3201, "max": 10000, "S-1": "C", "S-2": "D","S-3": "F", "S-4": "G","I": "J", "II": "L", "III": "M"},
    {"min": 10001, "max": 35000, "S-1": "C", "S-2": "D","S-3": "F", "S-4": "H","I": "K", "II": "M", "III": "N"},
    {"min": 35001, "max": 150000, "S-1": "D", "S-2": "E","S-3": "G", "S-4": "J","I": "L", "II": "N", "III": "P"},
    {"min": 150001, "max": 500000, "S-1": "D", "S-2": "E","S-3": "G", "S-4": "K","J": "M", "II": "P", "III": "Q"},
    {"min": 500001,  "S-1": "D", "S-2": "E","S-3": "H", "S-4": "K","I": "N", "II": "Q", "III": "R"},
]
  
tabla_normal={
    "A": {"0.010": {"n": 2, "Ac": 0, "Re": 1}, "0.015": {"n": 2, "Ac": 0, "Re": 1}, "0.025": {"n": 2, "Ac": 0, "Re": 1}, "0.040": {"n": 2, "Ac": 0, "Re": 1}, "0.065": {"n": 2, "Ac": 0, "Re": 1},"0.10": {"n": 2, "Ac": 0, "Re": 1},"0.15": {"n": 2, "Ac": 0, "Re": 1},"0.25": {"n": 2, "Ac": 0, "Re": 1},"0.40": {"n": 2, "Ac": 0, "Re": 1},"0.65": {"n": 2, "Ac": 0, "Re": 1}},
    "B": {"0.010": {"n": 3, "Ac": 0, "Re": 1}, "0.015": {"n": 3, "Ac": 0, "Re": 1}, "0.025": {"n": 3, "Ac": 0, "Re": 1}, "0.040": {"n": 3, "Ac": 0, "Re": 1}, "0.065": {"n": 3, "Ac": 0, "Re": 1},"0.10": {"n": 3, "Ac": 0, "Re": 1},"0.15": {"n": 3, "Ac": 0, "Re": 1},"0.25": {"n": 3, "Ac": 0, "Re": 1},"0.40": {"n": 3, "Ac": 0, "Re": 1},"0.65": {"n": 3, "Ac": 0, "Re": 1}},
    "C": {"0.010": {"n": 5, "Ac": 0, "Re": 1}, "0.015": {"n": 5, "Ac": 0, "Re": 1}, "0.025": {"n": 5, "Ac": 0, "Re": 1}, "0.040": {"n": 5, "Ac": 0, "Re": 1}, "0.065": {"n": 5, "Ac": 0, "Re": 1},"0.10": {"n": 5, "Ac": 0, "Re": 1},"0.15": {"n": 5, "Ac": 0, "Re": 1},"0.25": {"n": 5, "Ac": 0, "Re": 1},"0.40": {"n": 5, "Ac": 0, "Re": 1},"0.65": {"n": 5, "Ac": 0, "Re": 1}},
    "D": {"0.010": {"n": 8, "Ac": 0, "Re": 1}, "0.015": {"n": 8, "Ac": 0, "Re": 1}, "0.025": {"n": 8, "Ac": 0, "Re": 1}, "0.040": {"n": 8, "Ac": 0, "Re": 1}, "0.065": {"n": 8, "Ac": 0, "Re": 1},"0.10": {"n": 8, "Ac": 0, "Re": 1},"0.15": {"n": 8, "Ac": 0, "Re": 1},"0.25": {"n": 8, "Ac": 0, "Re": 1},"0.40": {"n": 8, "Ac": 0, "Re": 1},"0.65": {"n": 8, "Ac": 0, "Re": 1}},
    "E": {"0.010": {"n": 13, "Ac": 0, "Re": 1}, "0.015": {"n":13, "Ac": 0, "Re": 1}, "0.025": {"n": 13, "Ac": 0, "Re": 1}, "0.040": {"n": 13, "Ac": 0, "Re": 1}, "0.065": {"n": 13, "Ac": 0, "Re": 1},"0.10": {"n": 13, "Ac": 0, "Re": 1},"0.15": {"n": 13, "Ac": 0, "Re": 1},"0.25": {"n": 13, "Ac": 0, "Re": 1},"0.40": {"n": 13, "Ac": 0, "Re": 1},"0.65": {"n": 13, "Ac": 0, "Re": 1}},
    "F": {"0.010": {"n": 20, "Ac": 0, "Re": 1}, "0.015": {"n":20, "Ac": 0, "Re": 1}, "0.025": {"n": 20, "Ac": 0, "Re": 1}, "0.040": {"n": 20, "Ac": 0, "Re": 1}, "0.065": {"n": 20, "Ac": 0, "Re": 1},"0.10": {"n": 20, "Ac": 0, "Re": 1},"0.15": {"n": 20, "Ac": 0, "Re": 1},"0.25": {"n": 20, "Ac": 0, "Re": 1},"0.40": {"n": 20, "Ac": 0, "Re": 1},"0.65": {"n": 20, "Ac": 0, "Re": 1}},
    "G": {"0.010": {"n": 32, "Ac": 0, "Re": 1}, "0.015": {"n": 32, "Ac": 0, "Re": 1}, "0.025": {"n": 32, "Ac": 0, "Re": 1}, "0.040": {"n": 32, "Ac": 0, "Re": 1}, "0.065": {"n": 32, "Ac": 0, "Re": 1},"0.10": {"n": 32, "Ac": 0, "Re": 1},"0.15": {"n": 32, "Ac": 0, "Re": 1},"0.25": {"n": 32, "Ac": 0, "Re": 1},"0.40": {"n": 32, "Ac": 0, "Re": 1},"0.65": {"n": 32, "Ac": 0, "Re": 1}},
    "H": {"0.010": {"n": 50, "Ac": 0, "Re": 1}, "0.015": {"n": 50, "Ac": 0, "Re": 1}, "0.025": {"n": 50, "Ac": 0, "Re": 1}, "0.040": {"n": 50, "Ac": 0, "Re": 1}, "0.065": {"n": 50, "Ac": 0, "Re": 1},"0.10": {"n": 50, "Ac": 0, "Re": 1},"0.15": {"n": 50, "Ac": 0, "Re": 1},"0.25": {"n": 50, "Ac": 0, "Re": 1},"0.40": {"n": 50, "Ac": 0, "Re": 1},"0.65": {"n": 50, "Ac": 1, "Re": 2}},
    "J": {"0.010": {"n": 80, "Ac": 0, "Re": 1}, "0.015": {"n": 80, "Ac": 0, "Re": 1}, "0.025": {"n": 80, "Ac": 0, "Re": 1}, "0.040": {"n": 80, "Ac": 0, "Re": 1}, "0.065": {"n": 80, "Ac": 0, "Re": 1},"0.10": {"n": 80, "Ac": 0, "Re": 1},"0.15": {"n": 80, "Ac": 0, "Re": 1},"0.25": {"n": 80, "Ac": 0, "Re": 1},"0.40": {"n": 80, "Ac": 1, "Re": 2}, "0.65": {"n": 80, "Ac": 1, "Re": 2}},
    "K": {"0.010": {"n": 150, "Ac": 0, "Re": 1}, "0.015": {"n": 150, "Ac": 0, "Re": 1}, "0.025": {"n": 150, "Ac": 0, "Re": 1}, "0.040": {"n": 150, "Ac": 0, "Re": 1}, "0.065": {"n": 150, "Ac": 0, "Re": 1},"0.10": {"n": 150, "Ac": 0, "Re": 1},"0.15": {"n": 150, "Ac": 0, "Re": 1},"0.25": {"n": 150, "Ac": 1, "Re": 2}, "0.40": {"n": 150, "Ac": 1, "Re": 2},"0.65": {"n": 150, "Ac": 2, "Re": 3}},
    "L": {"0.010": {"n": 200, "Ac": 0, "Re": 1}, "0.015": {"n": 200, "Ac": 0, "Re": 1}, "0.025": {"n": 200, "Ac": 0, "Re": 1}, "0.040": {"n": 200, "Ac": 0, "Re": 1}, "0.065": {"n": 200, "Ac": 0, "Re": 1},"0.10": {"n": 200, "Ac": 0, "Re": 1},"0.15": {"n": 200, "Ac": 1, "Re": 2}, "0.25": {"n": 200, "Ac": 1, "Re": 2},"0.40": {"n": 200, "Ac": 2, "Re": 3},"0.65": {"n": 200, "Ac": 3, "Re": 4}},
    "M": {"0.010": {"n": 315, "Ac": 0, "Re": 1}, "0.015": {"n": 315, "Ac": 0, "Re": 1}, "0.025": {"n": 315, "Ac": 0, "Re": 1}, "0.040": {"n": 315, "Ac": 0, "Re": 1}, "0.065": {"n": 315, "Ac": 0, "Re": 1},"0.10": {"n":315,"Ac": 1, "Re": 2}, "0.15": {"n": 315, "Ac": 1, "Re": 2},"0.25": {"n": 315, "Ac": 2, "Re": 3},"0.40": {"n": 315, "Ac": 3, "Re": 4},"0.65": {"n": 315, "Ac": 5, "Re": 6}},
    "N": {"0.010": {"n": 500, "Ac": 0, "Re": 1}, "0.015": {"n": 500, "Ac": 0, "Re": 1}, "0.025": {"n": 500, "Ac": 0, "Re": 1}, "0.040": {"n": 500, "Ac": 0, "Re": 1}, "0.065": {"n": 500, "Ac": 1, "Re": 2}, "0.10": {"n": 500, "Ac": 1, "Re": 2},"0.15": {"n": 500, "Ac": 2, "Re": 3},"0.25": {"n": 500, "Ac": 3, "Re": 4},"0.40": {"n": 500, "Ac": 5, "Re": 6},"0.65": {"n": 500, "Ac": 7, "Re": 8}},
    "P": {"0.010": {"n": 800, "Ac": 0, "Re": 1}, "0.015": {"n": 800, "Ac": 0, "Re": 1}, "0.025": {"n": 800, "Ac": 0, "Re": 1}, "0.040": {"n": 800, "Ac": 1, "Re": 2}, "0.065": {"n": 800, "Ac": 1, "Re": 2},"0.10": {"n": 800, "Ac": 2, "Re": 3},"0.15": {"n": 800, "Ac": 3, "Re": 4},"0.25": {"n": 800, "Ac": 5, "Re": 6},"0.40": {"n": 800, "Ac": 7, "Re": 8},"0.65": {"n": 800, "Ac": 10, "Re": 11}},
    "Q": {"0.010": {"n": 1250, "Ac": 0, "Re": 1}, "0.015": {"n": 1250, "Ac": 0, "Re": 1}, "0.025": {"n": 1250, "Ac": 1, "Re": 2}, "0.040": {"n": 1250, "Ac": 1, "Re": 2}, "0.065": {"n": 1250, "Ac": 2, "Re": 3},"0.10": {"n": 1250, "Ac": 3, "Re": 4},"0.15": {"n": 1250, "Ac": 5, "Re": 6},"0.25": {"n": 1250, "Ac": 7, "Re": 8},"0.40": {"n": 1250, "Ac": 10, "Re": 11},"0.65": {"n": 1250, "Ac": 14, "Re": 15}},
    "R": {"0.010": {"n": 2000, "Ac": 0, "Re": 1}, "0.015": {"n": 2000, "Ac": 0, "Re": 1}, "0.025": {"n": 2000, "Ac": 1, "Re": 2}, "0.040": {"n": 2000, "Ac": 2, "Re": 3}, "0.065": {"n": 2000, "Ac": 3, "Re": 4},"0.10": {"n": 2000, "Ac": 5, "Re": 6},"0.15": {"n": 2000, "Ac": 7, "Re": 8},"0.25": {"n": 2000, "Ac": 10, "Re": 11},"0.40": {"n": 2000, "Ac": 14, "Re": 15},"0.65": {"n": 2000, "Ac": 21, "Re": 22}},


}

tabla_severa ={
     "A": {"0.010": {"n": 2, "Ac": 0, "Re": 1}, "0.015": {"n": 2, "Ac": 0, "Re": 1}, "0.025": {"n": 2, "Ac": 0, "Re": 1}, "0.040": {"n": 2, "Ac": 0, "Re": 1}, "0.065": {"n": 2, "Ac": 0, "Re": 1},"0.10": {"n": 2, "Ac": 0, "Re": 1},"0.15": {"n": 2, "Ac": 0, "Re": 1},"0.25": {"n": 2, "Ac": 0, "Re": 1},"0.40": {"n": 2, "Ac": 0, "Re": 1},"0.65": {"n": 2, "Ac": 0, "Re": 1}},
    "B": {"0.010": {"n": 3, "Ac": 0, "Re": 1}, "0.015": {"n": 3, "Ac": 0, "Re": 1}, "0.025": {"n": 3, "Ac": 0, "Re": 1}, "0.040": {"n": 3, "Ac": 0, "Re": 1}, "0.065": {"n": 3, "Ac": 0, "Re": 1},"0.10": {"n": 3, "Ac": 0, "Re": 1},"0.15": {"n": 3, "Ac": 0, "Re": 1},"0.25": {"n": 3, "Ac": 0, "Re": 1},"0.40": {"n": 3, "Ac": 0, "Re": 1},"0.65": {"n": 3, "Ac": 0, "Re": 1}},
    "C": {"0.010": {"n": 5, "Ac": 0, "Re": 1}, "0.015": {"n": 5, "Ac": 0, "Re": 1}, "0.025": {"n": 5, "Ac": 0, "Re": 1}, "0.040": {"n": 5, "Ac": 0, "Re": 1}, "0.065": {"n": 5, "Ac": 0, "Re": 1},"0.10": {"n": 5, "Ac": 0, "Re": 1},"0.15": {"n": 5, "Ac": 0, "Re": 1},"0.25": {"n": 5, "Ac": 0, "Re": 1},"0.40": {"n": 5, "Ac": 0, "Re": 1},"0.65": {"n": 5, "Ac": 0, "Re": 1}},
    "D": {"0.010": {"n": 8, "Ac": 0, "Re": 1}, "0.015": {"n": 8, "Ac": 0, "Re": 1}, "0.025": {"n": 8, "Ac": 0, "Re": 1}, "0.040": {"n": 8, "Ac": 0, "Re": 1}, "0.065": {"n": 8, "Ac": 0, "Re": 1},"0.10": {"n": 8, "Ac": 0, "Re": 1},"0.15": {"n": 8, "Ac": 0, "Re": 1},"0.25": {"n": 8, "Ac": 0, "Re": 1},"0.40": {"n": 8, "Ac": 0, "Re": 1},"0.65": {"n": 8, "Ac": 0, "Re": 1}},
    "E": {"0.010": {"n": 13, "Ac": 0, "Re": 1}, "0.015": {"n":13, "Ac": 0, "Re": 1}, "0.025": {"n": 13, "Ac": 0, "Re": 1}, "0.040": {"n": 13, "Ac": 0, "Re": 1}, "0.065": {"n": 13, "Ac": 0, "Re": 1},"0.10": {"n": 13, "Ac": 0, "Re": 1},"0.15": {"n": 13, "Ac": 0, "Re": 1},"0.25": {"n": 13, "Ac": 0, "Re": 1},"0.40": {"n": 13, "Ac": 0, "Re": 1},"0.65": {"n": 13, "Ac": 0, "Re": 1}},
    "F": {"0.010": {"n": 20, "Ac": 0, "Re": 1}, "0.015": {"n":20, "Ac": 0, "Re": 1}, "0.025": {"n": 20, "Ac": 0, "Re": 1}, "0.040": {"n": 20, "Ac": 0, "Re": 1}, "0.065": {"n": 20, "Ac": 0, "Re": 1},"0.10": {"n": 20, "Ac": 0, "Re": 1},"0.15": {"n": 20, "Ac": 0, "Re": 1},"0.25": {"n": 20, "Ac": 0, "Re": 1},"0.40": {"n": 20, "Ac": 0, "Re": 1},"0.65": {"n": 20, "Ac": 0, "Re": 1}},
    "G": {"0.010": {"n": 32, "Ac": 0, "Re": 1}, "0.015": {"n": 32, "Ac": 0, "Re": 1}, "0.025": {"n": 32, "Ac": 0, "Re": 1}, "0.040": {"n": 32, "Ac": 0, "Re": 1}, "0.065": {"n": 32, "Ac": 0, "Re": 1},"0.10": {"n": 32, "Ac": 0, "Re": 1},"0.15": {"n": 32, "Ac": 0, "Re": 1},"0.25": {"n": 32, "Ac": 0, "Re": 1},"0.40": {"n": 32, "Ac": 0, "Re": 1},"0.65": {"n": 32, "Ac": 1, "Re": 2}},
    "H": {"0.010": {"n": 50, "Ac": 0, "Re": 1}, "0.015": {"n": 50, "Ac": 0, "Re": 1}, "0.025": {"n": 50, "Ac": 0, "Re": 1}, "0.040": {"n": 50, "Ac": 0, "Re": 1}, "0.065": {"n": 50, "Ac": 0, "Re": 1},"0.10": {"n": 50, "Ac": 0, "Re": 1},"0.15": {"n": 50, "Ac": 0, "Re": 1},"0.25": {"n": 50, "Ac": 0, "Re": 1},"0.40": {"n": 50, "Ac": 1, "Re": 2},"0.65": {"n": 50, "Ac": 1, "Re": 2}},
    "J": {"0.010": {"n": 80, "Ac": 0, "Re": 1}, "0.015": {"n": 80, "Ac": 0, "Re": 1}, "0.025": {"n": 80, "Ac": 0, "Re": 1}, "0.040": {"n": 80, "Ac": 0, "Re": 1}, "0.065": {"n": 80, "Ac": 0, "Re": 1},"0.10": {"n": 80, "Ac": 0, "Re": 1},"0.15": {"n": 80, "Ac": 0, "Re": 1},"0.25": {"n": 80, "Ac": 1, "Re": 2},"0.40": {"n": 80, "Ac": 1, "Re": 2}, "0.65": {"n": 80, "Ac": 1, "Re": 2}},
    "K": {"0.010": {"n": 150, "Ac": 0, "Re": 1}, "0.015": {"n": 150, "Ac": 0, "Re": 1}, "0.025": {"n": 150, "Ac": 0, "Re": 1}, "0.040": {"n": 150, "Ac": 0, "Re": 1}, "0.065": {"n": 150, "Ac": 0, "Re": 1},"0.10": {"n": 150, "Ac": 0, "Re": 1},"0.15": {"n": 150, "Ac": 1, "Re": 2},"0.25": {"n": 150, "Ac": 1, "Re": 2}, "0.40": {"n": 150, "Ac": 1, "Re": 2},"0.65": {"n": 150, "Ac": 2, "Re": 3}},
    "L": {"0.010": {"n": 200, "Ac": 0, "Re": 1}, "0.015": {"n": 200, "Ac": 0, "Re": 1}, "0.025": {"n": 200, "Ac": 0, "Re": 1}, "0.040": {"n": 200, "Ac": 0, "Re": 1}, "0.065": {"n": 200, "Ac": 0, "Re": 1},"0.10": {"n": 200, "Ac": 1, "Re": 2},"0.15": {"n": 200, "Ac": 1, "Re": 2}, "0.25": {"n": 200, "Ac": 1, "Re": 2},"0.40": {"n": 200, "Ac": 2, "Re": 3},"0.65": {"n": 200, "Ac": 3, "Re": 4}},
    "M": {"0.010": {"n": 315, "Ac": 0, "Re": 1}, "0.015": {"n": 315, "Ac": 0, "Re": 1}, "0.025": {"n": 315, "Ac": 0, "Re": 1}, "0.040": {"n": 315, "Ac": 0, "Re": 1}, "0.065": {"n": 315, "Ac": 0, "Re": 1},"0.10": {"n":315,"Ac": 1, "Re": 2}, "0.15": {"n": 315, "Ac": 1, "Re": 2},"0.25": {"n": 315, "Ac": 2, "Re": 3},"0.40": {"n": 315, "Ac": 3, "Re": 4},"0.65": {"n": 315, "Ac": 5, "Re": 6}},
    "N": {"0.010": {"n": 500, "Ac": 0, "Re": 1}, "0.015": {"n": 500, "Ac": 0, "Re": 1}, "0.025": {"n": 500, "Ac": 0, "Re": 1}, "0.040": {"n": 500, "Ac": 0, "Re": 1}, "0.065": {"n": 500, "Ac": 1, "Re": 2}, "0.10": {"n": 500, "Ac": 1, "Re": 2},"0.15": {"n": 500, "Ac": 1, "Re": 2},"0.25": {"n":500,"Ac": 2, "Re": 3},"0.40": {"n": 500, "Ac": 3, "Re": 4},"0.65": {"n": 500, "Ac": 5, "Re": 6}},
    "P": {"0.010": {"n": 800, "Ac": 0, "Re": 1}, "0.015": {"n": 800, "Ac": 0, "Re": 1}, "0.025": {"n": 800, "Ac": 0, "Re": 1}, "0.040": {"n": 800, "Ac": 1, "Re": 2}, "0.065": {"n": 800, "Ac": 1, "Re": 2},"0.10": {"n": 800, "Ac": 1, "Re": 2},"0.15": {"n": 800,"Ac": 2, "Re": 3},"0.25": {"n": 800, "Ac": 3, "Re": 4},"0.40": {"n": 800, "Ac": 5, "Re": 6},"0.65": {"n": 800, "Ac": 8, "Re": 9}},
    "Q": {"0.010": {"n": 1250, "Ac": 0, "Re": 1}, "0.015": {"n": 1250, "Ac": 0, "Re": 1}, "0.025": {"n": 1250, "Ac": 1, "Re": 2}, "0.040": {"n": 1250, "Ac": 1, "Re": 2}, "0.065": {"n": 1250, "Ac": 1, "Re": 2}, "0.10": {"n": 1250, "Ac": 2, "Re": 3},"0.15": {"n": 1250, "Ac": 3, "Re": 4},"0.25": {"n":1250, "Ac": 5, "Re": 6},"0.40": {"n": 1250, "Ac": 8, "Re": 9},"0.65": {"n": 1250, "Ac": 12, "Re":13 }},
    "R": {"0.010": {"n": 2000, "Ac": 0, "Re": 1}, "0.015": {"n": 2000, "Ac": 0, "Re": 1}, "0.025": {"n": 2000, "Ac": 1, "Re": 2}, "0.040": {"n": 2000, "Ac": 1, "Re": 2}, "0.065": {"n": 2000, "Ac": 2, "Re": 3},"0.10": {"n": 2000, "Ac": 3, "Re": 4},"0.15": {"n": 2000, "Ac": 5, "Re": 6},"0.25": {"n": 2000, "Ac": 8, "Re": 9},"0.40": {"n": 2000, "Ac": 12, "Re":13 },"0.65": {"n": 2000, "Ac": 18, "Re": 19}},
    "S": {"0.010": {"n": 3150, "Ac": 0, "Re": 0}, "0.015": {"n": 3150, "Ac": 0, "Re": 0}, "0.025": {"n": 3150, "Ac": 1, "Re": 2}, "0.040": {"n": 3150, "Ac": 0, "Re": 0}, "0.065": {"n": 3150, "Ac":0, "Re": 0},"0.10": {"n": 3150, "Ac": 0, "Re": 0},"0.15": {"n": 3150, "Ac":0 , "Re": 0},"0.25": {"n": 3150, "Ac": 0, "Re": 0},"0.40": {"n": 3150, "Ac": 0, "Re": 0},"0.65": {"n": 3150, "Ac": 0, "Re": 0}},
}

tabla_reducida = {
     "A": {"0.010": {"n":2, "Ac": 0, "Re": 1}, "0.015": {"n": 2, "Ac": 0, "Re": 1}, "0.025": {"n": 2, "Ac": 0, "Re": 1}, "0.040": {"n": 2, "Ac": 0, "Re": 1}, "0.065": {"n": 2, "Ac": 0, "Re": 1},"0.10": {"n": 2, "Ac": 0, "Re": 1},"0.15": {"n": 2, "Ac": 0, "Re": 1},"0.25": {"n": 2, "Ac": 0, "Re": 1},"0.40": {"n": 2, "Ac": 0, "Re": 1},"0.65": {"n": 2, "Ac": 0, "Re": 1}},
    "B": {"0.010": {"n": 2, "Ac": 0, "Re": 1}, "0.015": {"n": 2, "Ac": 0, "Re": 1}, "0.025": {"n": 2, "Ac": 0, "Re": 1}, "0.040": {"n": 2, "Ac": 0, "Re": 1}, "0.065": {"n": 2, "Ac": 0, "Re": 1},"0.10": {"n": 2, "Ac": 0, "Re": 1},"0.15": {"n": 2, "Ac": 0, "Re": 1},"0.25": {"n": 2, "Ac": 0, "Re": 1},"0.40": {"n": 2, "Ac": 0, "Re": 1},"0.65": {"n": 2, "Ac": 0, "Re": 1}},
    "C": {"0.010": {"n": 2, "Ac": 0, "Re": 1}, "0.015": {"n": 2, "Ac": 0, "Re": 1}, "0.025": {"n": 2, "Ac": 0, "Re": 1}, "0.040": {"n": 2, "Ac": 0, "Re": 1}, "0.065": {"n": 2, "Ac": 0, "Re": 1},"0.10": {"n": 2, "Ac": 0, "Re": 1},"0.15": {"n": 2, "Ac": 0, "Re": 1},"0.25": {"n": 2, "Ac": 0, "Re": 1},"0.40": {"n": 2, "Ac": 0, "Re": 1},"0.65": {"n": 2, "Ac": 0, "Re": 1}},
    "D": {"0.010": {"n": 3, "Ac": 0, "Re": 1}, "0.015": {"n": 3, "Ac": 0, "Re": 1}, "0.025": {"n": 3, "Ac": 0, "Re": 1}, "0.040": {"n": 3, "Ac": 0, "Re": 1}, "0.065": {"n": 3, "Ac": 0, "Re": 1},"0.10": {"n": 3, "Ac": 0, "Re": 1},"0.15": {"n": 3, "Ac": 0, "Re": 1},"0.25": {"n": 3, "Ac": 0, "Re": 1},"0.40": {"n": 3, "Ac": 0, "Re": 1},"0.65": {"n": 3, "Ac": 0, "Re": 1}},
    "E": {"0.010": {"n": 5, "Ac": 0, "Re": 1}, "0.015": {"n":5, "Ac": 0, "Re": 1}, "0.025": {"n": 5, "Ac": 0, "Re": 1}, "0.040": {"n": 5, "Ac": 0, "Re": 1}, "0.065": {"n": 5, "Ac": 0, "Re": 1},"0.10": {"n": 5, "Ac": 0, "Re": 1},"0.15": {"n":5, "Ac": 0, "Re": 1},"0.25": {"n": 5, "Ac": 0, "Re": 1},"0.40": {"n": 5, "Ac": 0, "Re": 1},"0.65": {"n": 5, "Ac": 0, "Re": 1}},
    "F": {"0.010": {"n": 8, "Ac": 0, "Re": 1}, "0.015": {"n":8, "Ac": 0, "Re": 1}, "0.025": {"n": 8, "Ac": 0, "Re": 1}, "0.040": {"n": 8, "Ac": 0, "Re": 1}, "0.065": {"n": 8, "Ac": 0, "Re": 1},"0.10": {"n": 8, "Ac": 0, "Re": 1},"0.15": {"n": 8, "Ac": 0, "Re": 1},"0.25": {"n": 8, "Ac": 0, "Re": 1},"0.40": {"n": 8, "Ac": 0, "Re": 1},"0.65": {"n": 8, "Ac": 0, "Re": 1}},
    "G": {"0.010": {"n": 13, "Ac": 0, "Re": 1}, "0.015": {"n": 13, "Ac": 0, "Re": 1}, "0.025": {"n": 13, "Ac": 0, "Re": 1}, "0.040": {"n": 13, "Ac": 0, "Re": 1}, "0.065": {"n": 13, "Ac": 0, "Re": 1},"0.10": {"n": 13, "Ac": 0, "Re": 1},"0.15": {"n": 13, "Ac": 0, "Re": 1},"0.25": {"n": 13, "Ac": 0, "Re": 1},"0.40": {"n": 13, "Ac": 0, "Re": 1},"0.65": {"n": 13, "Ac": 0, "Re": 1}},
    "H": {"0.010": {"n": 20, "Ac": 0, "Re": 1}, "0.015": {"n": 20, "Ac": 0, "Re": 1}, "0.025": {"n": 20, "Ac": 0, "Re": 1}, "0.040": {"n": 20, "Ac": 0, "Re": 1}, "0.065": {"n": 20, "Ac": 0, "Re": 1},"0.10": {"n":20, "Ac": 0, "Re": 1},"0.15": {"n": 20, "Ac": 0, "Re": 1},"0.25": {"n": 20, "Ac": 0, "Re": 1},"0.40": {"n": 20, "Ac": 0, "Re": 1},"0.65": {"n": 20, "Ac": 0, "Re": 2}},
    "J": {"0.010": {"n": 32, "Ac": 0, "Re": 1}, "0.015": {"n": 32, "Ac": 0, "Re": 1}, "0.025": {"n": 32, "Ac": 0, "Re": 1}, "0.040": {"n": 32, "Ac": 0, "Re": 1}, "0.065": {"n": 32, "Ac": 0, "Re": 1},"0.10": {"n": 32, "Ac": 0, "Re": 1},"0.15": {"n":32, "Ac": 0, "Re": 1},"0.25": {"n": 32, "Ac": 0, "Re": 1},"0.40": {"n": 32, "Ac":0, "Re": 2}, "0.040": {"n": 32, "Ac": 0, "Re": 2}},
    "K": {"0.010": {"n": 50, "Ac": 0, "Re": 1}, "0.015": {"n": 50, "Ac": 0, "Re": 1}, "0.025": {"n": 50, "Ac": 0, "Re": 1}, "0.040": {"n":50, "Ac": 0, "Re": 1}, "0.065": {"n": 50, "Ac": 0, "Re": 1},"0.10": {"n":50, "Ac": 0, "Re": 1},"0.15": {"n": 50, "Ac": 0, "Re": 1},"0.25": {"n": 50, "Ac":0, "Re": 2}, "0.040": {"n": 50, "Ac": 0, "Re": 2}, "0.040": {"n":50, "Ac": 1, "Re": 3}},
    "L": {"0.010": {"n": 80, "Ac": 0, "Re": 1}, "0.015": {"n": 80, "Ac": 0, "Re": 1}, "0.025": {"n": 80, "Ac": 0, "Re": 1}, "0.040": {"n": 200, "Ac": 0, "Re": 1}, "0.065": {"n": 200, "Ac": 0, "Re": 1},"0.10": {"n": 200, "Ac": 0, "Re": 1},"0.15": {"n": 200, "Ac":0, "Re": 2}, "0.040": {"n": 1250, "Ac": 0, "Re": 2}, "0.040": {"n": 2000, "Ac": 1, "Re": 3}, "0.065": {"n": 2000, "Ac": 1, "Re": 4}},
    "M": {"0.010": {"n": 125, "Ac": 0, "Re": 1}, "0.015": {"n": 125, "Ac": 0, "Re": 1}, "0.025": {"n": 125, "Ac": 0, "Re": 1}, "0.040": {"n": 315, "Ac": 0, "Re": 1}, "0.065": {"n": 315, "Ac": 0, "Re": 1},"0.10": {"n":315,"Ac":0, "Re": 2}, "0.040": {"n": 1250, "Ac": 0, "Re": 2}, "0.040": {"n": 2000, "Ac": 1, "Re": 3}, "0.065": {"n": 2000, "Ac": 1, "Re": 4},"0.10": {"n": 2000, "Ac": 2, "Re": 5}},
    "N": {"0.010": {"n": 200, "Ac": 0, "Re": 1}, "0.015": {"n": 200, "Ac": 0, "Re": 1}, "0.025": {"n": 200, "Ac": 0, "Re": 1}, "0.040": {"n": 500, "Ac": 0, "Re": 1}, "0.065": {"n": 500,  "Ac":0, "Re": 2}, "0.040": {"n": 1250, "Ac": 0, "Re": 2}, "0.040": {"n": 2000, "Ac": 1, "Re": 3}, "0.065": {"n": 2000, "Ac": 1, "Re": 4},"0.10": {"n": 2000, "Ac": 2, "Re": 5},"0.15": {"n": 2000, "Ac": 3, "Re": 6}},
    "P": {"0.010": {"n": 315, "Ac": 0, "Re": 1}, "0.015": {"n": 315, "Ac": 0, "Re": 1}, "0.025": {"n": 315, "Ac": 0, "Re": 1}, "0.040": {"n": 800, "Ac":0, "Re": 2}, "0.065": {"n": 1250, "Ac": 0, "Re": 2}, "0.040": {"n": 2000, "Ac": 1, "Re": 3}, "0.065": {"n": 2000, "Ac": 1, "Re": 4},"0.10": {"n": 2000, "Ac": 2, "Re": 5},"0.15": {"n": 2000, "Ac": 3, "Re": 6},"0.25": {"n": 2000, "Ac": 5, "Re": 8}},
    "Q": {"0.010": {"n": 500, "Ac": 0, "Re": 1}, "0.015": {"n": 500, "Ac": 0, "Re": 1}, "0.025": {"n": 500, "Ac":0, "Re": 2}, "0.040": {"n": 1250, "Ac": 0, "Re": 2}, "0.065": {"n": 2000, "Ac": 1, "Re": 3}, "0.065": {"n": 2000, "Ac": 1, "Re": 4},"0.10": {"n": 2000, "Ac": 2, "Re": 5},"0.15": {"n": 2000, "Ac": 3, "Re": 6},"0.25": {"n": 2000, "Ac": 5, "Re": 8},"0.40": {"n": 2000, "Ac": 7, "Re": 10}},
    "R": {"0.010": {"n": 800, "Ac": 0, "Re": 1}, "0.015": {"n": 800, "Ac": 0, "Re": 1}, "0.025": {"n": 800, "Ac": 0, "Re": 2}, "0.040": {"n": 2000, "Ac": 1, "Re": 3}, "0.065": {"n": 2000, "Ac": 1, "Re": 4},"0.10": {"n": 2000, "Ac": 2, "Re": 5},"0.15": {"n": 2000, "Ac": 3, "Re": 6},"0.25": {"n": 2000, "Ac": 5, "Re": 8},"0.40": {"n": 2000, "Ac": 7, "Re": 10},"0.65": {"n": 2000, "Ac": 10, "Re": 13}},

} 

# --- Interfaz de usuario ---
st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background-color: #fff3e0; /* fondo naranja muy claro */
        color: #1a1a1a; /* texto negro */
    }

    /* Título principal */
    h1 {
        color: #f57c00; /* naranja principal */
        text-align: center;
        font-weight: 800;
    }

    /* Subtítulos */
    h2, h3 {
        color: #ffffff;
        border-bottom: 2px solid #f57c00;
        padding-bottom: 4px;
    }

    /* Texto y etiquetas */
    p, label, span {
        color:#1a1a1a !important;
        font-weight: 500;
    }

    /* Barra lateral */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border-right: 3px solid #f57c00;
    }
            
/* Texto dentro de la barra lateral */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 500;
}

    /* --- Campos de entrada --- */
    /* Ajusta color de fondo de inputs, selectbox y sliders */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div, 
    .stNumberInput input {
        background-color: #ffb74d !important;  /* naranja medio */
        color: #1a1a1a !important;              /* texto negro */
        border: 1px solid #f57c00 !important;
        border-radius: 6px;
    }

    /* Texto dentro de selectbox */
    div[data-baseweb="select"] span {
        color: #1a1a1a !important;
    }

    /* Hover de campos */
    div[data-baseweb="input"]:hover > div,
    div[data-baseweb="select"]:hover > div {
        background-color: #ffa726 !important; /* naranja un poco más oscuro */
        border-color: #ef6c00 !important;
    }

    /* Botones */
    .stButton>button {
        background-color: #f57c00;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #e65100;
        color: #ffffff;
    }

    /* Cuadro de éxito (resultados) */
    .stSuccess {
        background-color: #ffe0b2 !important;
        color: #1a1a1a !important;
        border-left: 5px solid #f57c00;
        font-weight: 600;
    }

    /* Advertencias y errores */
    .stWarning {
        background-color: #fff59d !important;
        color: #1a1a1a !important;
    }
    .stError {
        background-color: #d32f2f !important;
        color: #ffffff !important;
    }

    /* Scroll y barra lateral */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #f57c00;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

#-- Título y descripción ---
st.title("📊APLICACION DE PLAN DE MUESTREO MIL-STD-105E")
st.markdown("*Desarrollado por:* Liliana Ochoa, Julieth Achury, Daniela Pérez")
st.markdown("Para la asignatura de Aseguramiento de la calidad ")
st.markdown("Ingeniero Wilmar Guillermo Rodriguez ")
from PIL import Image
logo = Image.open("logo_uptc.png")
st.image(logo, width=200, caption="Universidad Pedagógica y Tecnológica de Colombia - UPTC")

# --- Entrada de datos ---
st.sidebar.header("📥 PARAMETROS DE ENTRADA ")
lote = st.sidebar.number_input("Tamaño del lote", min_value=2, value=500)
nivel = st.sidebar.selectbox("Nivel de inspección", ["S-1", "S-2", "S-3", "S-4", "I", "II", "III"])
tipo_inspeccion = st.sidebar.selectbox("Tipo de inspección", 
    options=["normal", "severa", "reducida"], 
    key="tipo_inspeccion")
aql = st.sidebar.selectbox("Nivel de calidad aceptable (AQL)", ["0.010", "0.015", "0.025", "0.040","0.065", "0.10","0.15", "0.25", "0.40", "0.65"])
rango_defectuosos = st.sidebar.selectbox(
    "Rango de porcentaje de defectuosos para la curva OC",
    ["0% - 10%", "0% - 20%", "0% - 30%", "0% - 50%"]
)

# --- Procesamiento ---
codigo_letra = None # none nos indica que no tiene valor asignado , esta linea inicializa la variable codigo_letra con un valor nulo.
for fila in tabla_letras: # recorremos cada fila de la tabla_letras
    if fila["min"] <= lote <= fila["max"]: # verificamos si el tamaño del lote esta dentro del rango min y max de la fila actual
        codigo_letra = fila[nivel] # si es asi, asignamos el codigo de letra correspondiente al nivel seleccionado
        break # salimos del bucle una vez que encontramos el codigo de letra adecuado

if codigo_letra: # si se encontro un codigo de letra valido
    if tipo_inspeccion == "normal": # si el tipo de inspeccion es normal
        datos = tabla_normal.get(codigo_letra, {}).get(aql) # obtenemos los datos de la tabla normal usando el codigo de letra y el AQL seleccionado
    elif tipo_inspeccion == "severa": # si el tipo de inspeccion es severa
        datos = tabla_severa.get(codigo_letra, {}).get(aql) # obtenemos los datos de la tabla severa
    else:
        datos = tabla_reducida.get(codigo_letra, {}).get(aql)# obtenemos los datos de la tabla reducida

    if datos:# si se encontraron datos para el AQL seleccionado
        st.success(f"Código de letra: *{codigo_letra}*")# mostramos el codigo de letra
        st.write(f"Tamaño de muestra: *{datos['n']}* unidades")# mostramos el tamaño de muestra
        st.write(f"Número de aceptación (Ac): *{datos['Ac']}*")# mostramos el numero de aceptacion
        st.write(f"Número de rechazo (Re): *{datos['Re']}*")   # mostramos el numero de rechazo
    else:
        st.warning("No hay datos disponibles para este AQL en la tabla seleccionada.") # si no se encontraron datos para el AQL seleccionado
else:
    st.error("El tamaño de lote no está dentro del rango de la tabla.") # si no se encontro un codigo de letra valido
    

        # --- NUEVA SECCIÓN: Evaluación de la muestra con mediciones ---
st.subheader("📌 Evaluación del lote con datos reales de la muestra")

st.markdown("Ingrese los valores medidos en la muestra y los límites de especificación para evaluar automáticamente los defectos.")

# --- LÍMITES SUPERIOR E INFERIOR ---
LSL = st.number_input("Ingrese el Límite Inferior de Especificación (LSL):", value=0.0, format="%.4f")
USL = st.number_input("Ingrese el Límite Superior de Especificación (USL):", value=1.0, format="%.4f")

# --- ENTRADA DE MUESTRA COMPLETA ---
st.write(f"Ingrese **{datos['n']} valores** correspondientes a la muestra:")

muestra_input = st.text_area(
    "Escriba los valores separados por comas (Ejemplo: 3.1, 2.9, 3.0, 3.2 ...)",
    ""
)

if muestra_input:
    try:
        # Convertir texto a lista de números
        muestra = [float(x.strip()) for x in muestra_input.split(",")]

        if len(muestra) != datos["n"]:
            st.error(f"Debe ingresar exactamente **{datos['n']} valores**.")
        else:
            # Contar defectos fuera de los límites
            defectos_fuera = sum((x < LSL or x > USL) for x in muestra)

            st.write(f"🔎 Número de unidades fuera de especificación: **{defectos_fuera}**")

            # --- Evaluación según Ac y Re ---
            if defectos_fuera <= datos["Ac"]:
                st.success("✔️ EL LOTE SE ACEPTA")
                st.write(
                    f"Con base en los valores medidos, solo se encontraron {defectos_fuera} defectos, "
                    f"lo cual está dentro del límite permitido (Ac = {datos['Ac']})."
                )

            elif defectos_fuera >= datos["Re"]:
                st.error("❌ EL LOTE SE RECHAZA")
                st.write(
                    f"El lote excede los límites permitidos por el plan de muestreo. "
                    f"Los {defectos_fuera} defectos superan el valor permitido (Re = {datos['Re']})."
                )
            else:
                st.warning(
                    "⚠️ La cantidad de defectos se encuentra entre Ac y Re. "
                    "Según la MIL-STD-105E, se requiere una decisión adicional o reinspección."
                )

    except:
        st.error("Formato incorrecto. Asegúrese de ingresar solo números separados por comas.")

            # --- Evaluación de la muestra ingresada por el usuario ---
        st.subheader("📌 Evaluación del lote según los defectos encontrados")

        defectos_encontrados = st.number_input(
            "Ingrese el número de unidades defectuosas encontradas en la muestra:",
        min_value=0, value=0
        )

        if defectos_encontrados <= datos["Ac"]:
            st.success("✔️ EL LOTE SE ACEPTA")

            st.write(
            f"El lote cumple con la calidad requerida. "
            f"Con {defectos_encontrados} defectos encontrados, no se supera el límite de aceptación (Ac = {datos['Ac']})."
            )

        elif defectos_encontrados >= datos["Re"]:
            st.error("❌ EL LOTE SE RECHAZA")

            st.write(
            f"El lote NO cumple con el estándar MIL-STD-105E. "
            f"Los {defectos_encontrados} defectos encontrados superan el límite permitido "
            f"para este plan de muestreo (Re = {datos['Re']})."
            )

        else:
             st.warning(
            "⚠️ El número de defectos está entre Ac y Re. "
            "Se requiere aplicar resolución específica del estándar o repetir la muestra."
            )

    


# --- Generar Curva OC ---
# Solo se genera si existen datos de n y Ac
if datos:
    n = datos["n"]
    Ac = datos["Ac"]

    # Rango de probabilidad de defectuosos (de 0% a 20%)
    # --- Rango de probabilidad de defectuosos según selección ---
    if rango_defectuosos == "0% - 10%":
        p = np.linspace(0, 0.1, 100)
    elif rango_defectuosos == "0% - 20%":
        p = np.linspace(0, 0.2, 100)
    elif rango_defectuosos == "0% - 30%":
        p = np.linspace(0, 0.3, 100)
    else:
        p = np.linspace(0, 0.5, 100)


    # Probabilidad de aceptar el lote (suma binomial acumulada)
    P_aceptar = binom.cdf(Ac, n, p)

    # Crear la gráfica
    fig, ax = plt.subplots()
    ax.plot(p * 100, P_aceptar, color="#f57c00", linewidth=2)
    ax.set_title("Curva Característica Operativa (O.C.)", fontsize=14, color="#1a1a1a")
    ax.set_xlabel("Porcentaje real de defectuosos (%)", fontsize=12)
    ax.set_ylabel("Probabilidad de aceptación", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.patch.set_facecolor("#fff3e0")
    ax.set_facecolor("#ffffff")
    ax.spines["bottom"].set_color("#1a1a1a")
    ax.spines["left"].set_color("#1a1a1a")
    ax.tick_params(colors="#1a1a1a")
    ax.title.set_color("#f57c00")
    # Mostrar en la app
    st.pyplot(fig)







