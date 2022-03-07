import numpy as np
import random
import networkx as nx
import copy as cp
import gurobipy as gb


class Environment(object):
    def __init__(self):
        self.num_agents = 6
        self.controllable_sd_list = ["AD", "AE", "BD", "BF", "CE", "CF"]
        self.num_links = 40
        self.link_names = ["A1", "A2", "B2", "B3", "C3", "C4",
                           "18", "15", "25", "26", "36", "37", "47", "411",
                           "58", "59", "69", "610", "710", "711",
                           "815", "812", "912", "913", "1013", "1014", "1114", "1118",
                           "1215", "1216", "1316", "1317", "1417", "1418",
                           "15D", "16D", "16E", "17E", "17F", "18F"]
        self.link_capacity = np.array(
            [4, 6, 6, 4, 6, 4, 1, 3, 6, 6,
             3, 6, 3, 1, 6, 3, 4, 6, 6, 4,
             1, 3, 6, 6, 3, 6, 3, 1, 6, 3,
             4, 6, 6, 4, 6, 3, 4, 6, 6, 4],
            dtype=np.float32) * 100.0
        self.num_paths = [6] * len(self.controllable_sd_list)
        self.paths = self.gen_paths()
        self.paths_mask = {}
        for agent in self.controllable_sd_list:
            self.paths_mask[agent] = self.gen_path_mask(self.paths[agent])

        self.action_effective_step = 7
        self.point_count = 628
        self.flow_AD = 3 * np.array(
            [58, 52, 52, 54, 54, 43, 45, 51, 42, 42, 48, 44, 42, 42, 43, 38, 46, 48, 41, 44, 39, 37, 40, 44, 47, 39, 38,
             42, 52, 42, 39, 45, 39, 39, 42, 38, 39, 39, 41, 42, 44, 40, 38, 44, 37, 36, 41, 30, 30, 39, 42, 44, 35, 20,
             21, 23, 19, 18, 21, 18, 17, 16, 15, 16, 17, 16, 20, 25, 20, 20, 26, 19, 19, 18, 17, 18, 27, 22, 24, 32, 23,
             26, 30, 26, 23, 25, 26, 23, 26, 20, 23, 30, 28, 27, 33, 29, 28, 24, 22, 24, 30, 21, 26, 30, 25, 27, 34, 28,
             31, 30, 33, 43, 44, 40, 39, 46, 44, 49, 55, 46, 55, 61, 60, 46, 46, 36, 48, 51, 40, 39, 50, 50, 48, 50, 52,
             55, 74, 52, 61, 70, 79, 56, 61, 56, 53, 57, 62, 55, 78, 58, 53, 63, 49, 46, 61, 46, 51, 50, 51, 56, 70, 51,
             49, 51, 47, 60, 61, 52, 61, 55, 66, 60, 69, 54, 56, 86, 87, 89, 99, 82, 86, 79, 82, 86, 98, 78, 42, 50, 46,
             54, 56, 64, 53, 67, 77, 56, 72, 69, 58, 65, 57, 49, 46, 47, 57, 53, 60, 58, 55, 54, 40, 44, 36, 42, 48, 40,
             46, 38, 39, 39, 58, 44, 40, 45, 43, 39, 53, 43, 44, 42, 44, 51, 54, 46, 46, 61, 51, 53, 49, 39, 41, 40, 32,
             40, 51, 45, 46, 50, 47, 46, 52, 41, 49, 46, 53, 37, 41, 35, 48, 62, 58, 59, 64, 57, 56, 51, 51, 56, 60, 53,
             54, 62, 55, 49, 49, 49, 40, 62, 67, 70, 65, 55, 52, 60, 55, 58, 51, 46, 47, 46, 44, 48, 49, 43, 41, 48, 45,
             42, 43, 37, 35, 32, 26, 37, 42, 40, 41, 48, 39, 29, 39, 39, 36, 34, 39, 34, 46, 45, 41, 47, 37, 38, 37, 32,
             28, 28, 27, 25, 31, 23, 23, 28, 31, 29, 32, 29, 28, 28, 22, 35, 34, 10, 8, 14, 9, 9, 14, 9, 9, 11, 11, 10,
             13, 11, 8, 16, 11, 12, 15, 12, 13, 15, 15, 14, 16, 11, 16, 24, 15, 18, 19, 25, 15, 13, 13, 12, 20, 15, 13,
             23, 11, 13, 40, 15, 14, 17, 15, 16, 19, 18, 25, 38, 36, 35, 42, 33, 33, 37, 41, 34, 33, 30, 29, 42, 37, 40,
             42, 38, 47, 40, 42, 42, 51, 48, 52, 59, 54, 54, 64, 53, 54, 49, 45, 31, 43, 51, 45, 35, 49, 46, 54, 44, 42,
             57, 50, 55, 59, 60, 60, 64, 56, 48, 57, 51, 51, 48, 47, 46, 59, 53, 58, 48, 78, 82, 65, 48, 47, 52, 60, 74,
             57, 55, 52, 50, 65, 53, 83, 110, 108, 98, 91, 93, 92, 100, 107, 104, 93, 81, 91, 79, 78, 104, 78, 76, 68,
             66, 57, 57, 75, 67, 57, 52, 46, 44, 36, 42, 42, 32, 35, 36, 34, 34, 36, 31, 35, 31, 34, 28, 41, 37, 35, 25,
             27, 28, 40, 33, 32, 29, 31, 32, 38, 32, 28, 26, 32, 31, 47, 31, 28, 25, 29, 30, 36, 28, 31, 22, 24, 25, 31,
             22, 19, 19, 18, 27, 37, 33, 31, 25, 39, 38, 42, 63, 77, 46, 40, 40, 44, 38, 43, 53, 52, 48, 56, 58, 58, 57,
             45, 59, 61, 55, 53, 55, 46, 48, 55, 43, 44, 46, 41, 33, 31, 25, 29, 37, 41, 34, 40, 35, 36, 34, 27, 28, 39,
             38, 43, 33, 31, 35, 39, 30, 34, 31, 25, 32, 50, 50, 53, 48, 28, 31, 46, 33, 44, 45, 40, 40, 48, 41, 40, 39,
             37, 35, 37, 37, 34, 33, 26, 25])
        self.flow_AE = 3 * np.array(
            [67, 63, 58, 53, 54, 58, 65, 60, 62, 59, 54, 53, 59, 64, 68, 61, 67, 56, 61, 72, 62, 71, 60, 58, 64, 68, 81,
             73, 79, 78, 80, 77, 78, 78, 78, 74, 66, 68, 73, 61, 65, 61, 70, 70, 72, 75, 71, 60, 58, 60, 66, 62, 60, 57,
             76, 75, 80, 70, 60, 56, 52, 57, 73, 69, 65, 62, 59, 64, 68, 70, 60, 68, 65, 80, 79, 74, 75, 63, 67, 64, 65,
             63, 64, 55, 53, 56, 64, 59, 61, 54, 54, 58, 58, 59, 52, 47, 49, 63, 56, 47, 47, 39, 50, 47, 51, 46, 38, 38,
             44, 42, 49, 47, 52, 59, 68, 75, 77, 72, 66, 63, 64, 61, 64, 56, 59, 54, 62, 60, 66, 67, 58, 53, 64, 56, 49,
             45, 47, 42, 47, 53, 51, 49, 45, 42, 46, 45, 48, 43, 41, 37, 39, 41, 46, 49, 41, 38, 48, 46, 49, 46, 47, 43,
             45, 49, 50, 60, 54, 54, 55, 47, 54, 53, 52, 49, 44, 58, 52, 52, 50, 48, 49, 54, 59, 50, 48, 44, 47, 53, 60,
             64, 59, 51, 42, 36, 44, 42, 42, 31, 38, 45, 42, 41, 36, 41, 50, 61, 54, 45, 49, 45, 46, 45, 48, 47, 39, 45,
             48, 48, 49, 40, 42, 37, 45, 41, 44, 48, 40, 43, 40, 38, 42, 41, 41, 39, 42, 41, 44, 47, 39, 38, 45, 39, 51,
             43, 42, 34, 40, 43, 46, 42, 39, 35, 43, 42, 43, 42, 47, 41, 47, 52, 50, 43, 35, 40, 37, 34, 35, 33, 37, 47,
             51, 52, 51, 46, 46, 50, 52, 58, 65, 59, 56, 49, 50, 56, 55, 47, 46, 52, 51, 52, 59, 53, 52, 52, 55, 46, 43,
             56, 53, 57, 65, 65, 65, 58, 63, 62, 69, 69, 73, 66, 56, 58, 59, 62, 70, 66, 75, 71, 75, 85, 74, 75, 76, 74,
             70, 71, 77, 74, 86, 75, 73, 69, 69, 73, 68, 68, 65, 73, 83, 78, 92, 82, 70, 75, 74, 72, 65, 74, 73, 73, 75,
             62, 70, 74, 68, 75, 77, 70, 65, 67, 80, 76, 66, 62, 61, 54, 54, 64, 66, 58, 62, 59, 61, 62, 55, 57, 60, 54,
             56, 53, 60, 63, 48, 54, 54, 67, 64, 55, 54, 48, 51, 53, 54, 57, 48, 49, 59, 54, 56, 54, 54, 56, 61, 64, 61,
             55, 53, 57, 58, 59, 55, 49, 56, 48, 48, 53, 58, 52, 43, 53, 52, 49, 60, 57, 63, 54, 52, 56, 56, 57, 49, 57,
             63, 57, 59, 48, 50, 52, 57, 54, 56, 55, 47, 46, 49, 42, 50, 56, 62, 52, 61, 64, 63, 67, 64, 61, 56, 55, 58,
             57, 61, 61, 69, 67, 66, 64, 51, 49, 51, 47, 48, 43, 49, 45, 45, 44, 52, 56, 50, 50, 48, 45, 47, 43, 51, 46,
             48, 50, 53, 53, 55, 57, 57, 50, 58, 46, 54, 44, 45, 52, 48, 48, 45, 40, 44, 44, 47, 43, 49, 41, 54, 43, 48,
             46, 41, 46, 59, 58, 59, 49, 50, 42, 49, 54, 62, 54, 47, 50, 53, 48, 49, 41, 46, 38, 47, 53, 55, 55, 53, 53,
             56, 60, 59, 57, 62, 40, 40, 47, 48, 40, 36, 37, 40, 36, 40, 35, 42, 32, 36, 38, 38, 40, 40, 42, 48, 48, 51,
             40, 45, 44, 48, 47, 48, 50, 47, 44, 41, 41, 48, 54, 72, 50, 49, 53, 55, 54, 53, 51, 58, 54, 58, 53, 63, 61,
             58, 53, 57, 61, 57, 50, 56, 52, 59, 55, 61, 54, 63, 66, 67, 63, 58, 58, 64, 61, 57, 57, 63, 57, 63, 62, 60,
             59, 55, 56, 67, 66, 60, 61])
        self.flow_BD = 3 * np.array(
            [13, 14, 14, 13, 14, 14, 12, 15, 15, 16, 17, 18, 20, 17, 14, 15, 18, 11, 12, 11, 12, 13, 16, 14, 12, 12, 17,
             23, 18, 22, 20, 29, 21, 18, 23, 19, 22, 26, 34, 29, 43, 38, 41, 37, 37, 41, 48, 42, 39, 43, 48, 60, 47, 40,
             47, 52, 46, 45, 61, 61, 57, 54, 71, 54, 69, 45, 45, 72, 55, 59, 72, 55, 61, 76, 74, 75, 78, 89, 60, 72, 69,
             58, 81, 61, 71, 49, 59, 54, 58, 43, 44, 50, 53, 44, 61, 50, 54, 48, 40, 54, 43, 40, 57, 77, 75, 51, 58, 52,
             54, 57, 61, 60, 64, 54, 70, 82, 81, 74, 58, 58, 57, 74, 52, 63, 79, 67, 58, 64, 49, 50, 66, 58, 69, 67, 58,
             56, 64, 63, 55, 54, 56, 54, 46, 32, 35, 41, 33, 37, 45, 28, 27, 33, 24, 25, 31, 23, 31, 22, 19, 29, 32, 28,
             30, 33, 27, 25, 46, 36, 37, 39, 40, 26, 28, 21, 26, 31, 31, 24, 30, 20, 20, 25, 22, 23, 24, 23, 25, 30, 25,
             27, 44, 44, 45, 44, 40, 41, 55, 42, 35, 43, 31, 30, 36, 28, 30, 38, 31, 33, 42, 35, 41, 41, 16, 30, 40, 36,
             33, 36, 30, 29, 31, 21, 23, 30, 32, 29, 44, 41, 33, 39, 29, 33, 38, 37, 36, 40, 30, 26, 37, 26, 29, 31, 23,
             33, 54, 45, 52, 66, 56, 54, 64, 55, 40, 40, 29, 29, 30, 21, 26, 28, 26, 22, 28, 25, 20, 25, 22, 26, 26, 15,
             16, 21, 14, 12, 12, 15, 11, 10, 11, 9, 18, 12, 15, 17, 11, 12, 16, 16, 16, 18, 14, 18, 24, 16, 18, 23, 16,
             15, 18, 14, 14, 12, 11, 9, 12, 9, 14, 17, 7, 10, 19, 11, 12, 9, 11, 15, 21, 23, 25, 24, 20, 17, 22, 22, 20,
             21, 20, 20, 30, 28, 27, 32, 36, 38, 49, 42, 44, 54, 50, 52, 64, 48, 54, 59, 54, 54, 55, 45, 47, 49, 58, 44,
             43, 40, 62, 67, 62, 62, 70, 62, 59, 57, 53, 44, 64, 44, 48, 58, 56, 52, 56, 63, 87, 74, 74, 77, 87, 73, 78,
             89, 71, 73, 82, 74, 63, 63, 58, 73, 77, 68, 60, 98, 78, 86, 84, 86, 73, 76, 82, 58, 65, 61, 54, 71, 55, 62,
             69, 70, 62, 60, 58, 66, 54, 47, 44, 53, 56, 56, 53, 59, 64, 51, 45, 45, 51, 45, 30, 50, 49, 47, 49, 44, 54,
             46, 41, 38, 34, 34, 42, 40, 31, 34, 30, 34, 39, 28, 28, 50, 61, 49, 40, 57, 35, 40, 31, 25, 24, 25, 27, 29,
             35, 33, 31, 42, 42, 47, 60, 45, 45, 43, 45, 44, 44, 41, 38, 47, 43, 45, 46, 35, 39, 49, 45, 47, 55, 48, 51,
             69, 54, 55, 57, 48, 48, 51, 42, 35, 39, 37, 57, 59, 35, 56, 47, 36, 37, 34, 31, 38, 37, 51, 50, 55, 33, 36,
             45, 40, 45, 48, 37, 36, 31, 32, 34, 36, 35, 34, 38, 33, 36, 36, 35, 43, 42, 50, 42, 47, 36, 38, 39, 36, 36,
             35, 31, 35, 32, 23, 22, 29, 30, 26, 32, 36, 31, 35, 33, 31, 35, 19, 17, 19, 14, 12, 11, 12, 15, 14, 13, 15,
             29, 20, 20, 20, 12, 8, 14, 12, 13, 13, 13, 14, 17, 13, 14, 14, 15, 14, 20, 12, 12, 17, 18, 13, 16, 17, 16,
             20, 16, 14, 21, 37, 64, 19, 18, 23, 30, 23, 25, 31, 23, 52, 34, 30, 24, 33, 27, 41, 71, 29, 25, 47, 52, 45,
             50, 47, 47, 49, 43, 49])
        self.flow_BF = 3 * np.array(
            [50, 42, 34, 36, 39, 46, 59, 48, 50, 47, 46, 49, 50, 50, 45, 43, 49, 45, 50, 46, 44, 46, 49, 55, 58, 60, 56,
             56, 59, 58, 58, 50, 56, 50, 51, 57, 56, 57, 49, 47, 55, 63, 70, 60, 58, 61, 76, 84, 76, 66, 61, 64, 66, 67,
             74, 70, 71, 67, 64, 68, 68, 67, 70, 69, 64, 65, 67, 60, 54, 53, 57, 60, 61, 61, 54, 55, 56, 53, 58, 56, 57,
             64, 65, 66, 66, 65, 63, 71, 62, 53, 56, 55, 51, 51, 56, 59, 59, 71, 71, 60, 63, 60, 69, 64, 56, 56, 63, 58,
             63, 64, 62, 61, 61, 54, 58, 51, 55, 53, 51, 55, 59, 63, 65, 64, 64, 63, 58, 54, 50, 48, 45, 43, 47, 49, 44,
             40, 45, 44, 47, 43, 39, 39, 39, 38, 38, 40, 32, 40, 41, 42, 43, 42, 49, 39, 42, 43, 42, 43, 38, 38, 41, 39,
             47, 62, 65, 51, 46, 46, 52, 52, 44, 41, 46, 51, 52, 49, 56, 45, 45, 50, 53, 51, 48, 47, 52, 46, 67, 72, 73,
             66, 67, 68, 67, 57, 47, 62, 65, 55, 58, 56, 60, 59, 57, 60, 56, 47, 40, 45, 44, 37, 43, 34, 40, 39, 43, 46,
             52, 51, 52, 48, 56, 52, 53, 47, 52, 49, 47, 48, 53, 54, 49, 47, 57, 61, 54, 44, 50, 40, 54, 47, 52, 55, 49,
             46, 44, 44, 51, 48, 53, 45, 50, 55, 54, 51, 51, 47, 49, 43, 51, 44, 50, 41, 58, 56, 56, 56, 54, 61, 58, 48,
             50, 49, 59, 39, 41, 44, 45, 38, 34, 38, 39, 44, 48, 41, 49, 41, 44, 45, 50, 43, 44, 48, 43, 42, 53, 47, 51,
             52, 62, 60, 55, 54, 48, 51, 60, 57, 57, 49, 56, 57, 55, 60, 74, 72, 62, 68, 57, 54, 61, 54, 62, 48, 50, 53,
             59, 60, 56, 57, 66, 64, 66, 66, 81, 75, 78, 68, 72, 74, 70, 72, 81, 69, 71, 72, 80, 72, 77, 75, 79, 79, 79,
             77, 73, 70, 74, 74, 85, 73, 77, 82, 89, 90, 73, 69, 73, 70, 75, 52, 62, 54, 59, 58, 63, 64, 62, 62, 62, 59,
             69, 66, 64, 54, 59, 58, 62, 61, 54, 60, 63, 57, 59, 57, 64, 54, 58, 62, 51, 53, 44, 49, 53, 47, 50, 53, 61,
             51, 52, 51, 49, 52, 47, 48, 53, 51, 54, 42, 47, 40, 42, 45, 47, 48, 43, 45, 46, 41, 45, 39, 47, 36, 42, 48,
             47, 45, 42, 42, 49, 47, 50, 43, 51, 42, 46, 46, 46, 49, 42, 43, 49, 44, 46, 42, 51, 39, 43, 42, 42, 45, 43,
             42, 52, 51, 48, 49, 51, 44, 47, 51, 58, 52, 52, 49, 43, 45, 63, 56, 54, 57, 63, 71, 73, 74, 64, 54, 50, 44,
             47, 45, 48, 40, 47, 38, 37, 36, 35, 34, 33, 30, 31, 34, 40, 32, 37, 42, 43, 44, 43, 45, 44, 38, 46, 43, 45,
             44, 44, 43, 47, 46, 43, 45, 50, 50, 42, 33, 38, 39, 48, 46, 47, 44, 43, 47, 46, 45, 44, 42, 44, 43, 48, 49,
             50, 50, 40, 43, 43, 36, 41, 36, 40, 44, 42, 47, 49, 51, 45, 49, 50, 43, 43, 33, 42, 44, 45, 44, 46, 49, 41,
             42, 42, 50, 47, 48, 46, 39, 39, 40, 43, 43, 33, 33, 38, 34, 42, 34, 36, 38, 40, 39, 40, 43, 37, 41, 52, 46,
             46, 47, 46, 47, 52, 54, 54, 57, 47, 49, 52, 50, 49, 53, 59, 67, 58, 58, 61, 58, 66, 50, 55, 54, 55, 54, 54,
             47, 44, 59, 58, 54, 65, 60])
        self.flow_CE = 0.5 * (self.flow_AD + self.flow_AE)
        self.flow_CF = 0.5 * (self.flow_BD + self.flow_BF)
        self.flows = np.array([self.flow_AD, self.flow_AE, self.flow_BD, self.flow_BF, self.flow_CE, self.flow_CF])
        self.max_len = 16

        self.agent_action = {}
        for i, agent in enumerate(self.controllable_sd_list):
            self.agent_action[agent] = np.array([1 / self.num_paths[i]] * self.num_paths[i])

    def gen_paths(self):
        graph = nx.Graph()
        graph.add_nodes_from(
            ["A", "B", "C", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "D", "E", "F"])
        for link in self.link_names:
            if len(link) == 2:
                graph.add_edge(link[0], link[1])
            elif len(link) == 4:
                graph.add_edge(link[:2], link[2:])
            elif len(link) == 3:
                if "D" in link or "E" in link or "F" in link:
                    graph.add_edge(link[:2], link[2])
                else:
                    graph.add_edge(link[0], link[1:])
        path = {}
        for i, agent in enumerate(self.controllable_sd_list):
            path[agent] = self.k_shortest_paths(graph, agent[0], agent[1], self.num_paths[i])
        return path

    def k_shortest_paths(self, graph, source, target, k=1):
        A = [nx.dijkstra_path(graph, source, target, weight='weight')]
        B = []

        for i in range(1, k):
            for j in range(0, len(A[-1]) - 1):
                Gcopy = cp.deepcopy(graph)
                spurnode = A[-1][j]
                rootpath = A[-1][:j + 1]
                for path in A:
                    if rootpath == path[0:j + 1]:  # and len(path) > j?
                        if Gcopy.has_edge(path[j], path[j + 1]):
                            Gcopy.remove_edge(path[j], path[j + 1])
                        if Gcopy.has_edge(path[j + 1], path[j]):
                            Gcopy.remove_edge(path[j + 1], path[j])
                for n in rootpath:
                    if n != spurnode:
                        Gcopy.remove_node(n)
                try:
                    spurpath = nx.dijkstra_path(Gcopy, spurnode, target, weight='weight')
                    totalpath = rootpath + spurpath[1:]
                    if totalpath not in B:
                        B += [totalpath]
                except nx.NetworkXNoPath:
                    continue
            if len(B) == 0:
                break
            lenB = [len(path) for path in B]
            B = [p for _, p in sorted(zip(lenB, B))]
            A.append(B[0])
            # A_len.append(sorted(lenB)[0])
            B.remove(B[0])
        return A

    def gen_path_mask(self, paths_list):
        paths_mask = []
        for path_str in paths_list:
            path = np.zeros(self.num_links)
            for i in range(len(path_str) - 1):
                s, d = path_str[i], path_str[i + 1]
                if s + d in self.link_names:
                    idx = self.link_names.index(s + d)
                else:
                    idx = self.link_names.index(d + s)
                path[idx] = 1
            paths_mask.append(np.copy(path))
        return np.array(paths_mask)

    def sample(self, pointer):
        action_effective_step = self.action_effective_step
        if pointer < action_effective_step:
            a = self.flows[:, self.point_count - action_effective_step + pointer:]
            b = self.flows[:, :pointer + action_effective_step]
            flows = np.concatenate([a, b], axis=-1)
        elif self.point_count - pointer < action_effective_step:
            a = self.flows[:, pointer - action_effective_step:]
            b = self.flows[:, :action_effective_step - pointer]
            flows = np.concatenate([a, b], axis=-1)
        else:
            flows = self.flows[:, pointer - action_effective_step:pointer + action_effective_step]
        return flows

    def reset(self):
        for i, agent in enumerate(self.controllable_sd_list):
            self.agent_action[agent] = np.array([1 / self.num_paths[i]] * self.num_paths[i])
        self.agent_action = {}
        for i, agent in enumerate(self.controllable_sd_list):
            self.agent_action[agent] = np.array([1 / self.num_paths[i]] * self.num_paths[i])
        self.data_step = np.random.randint(self.point_count)  # set the started flow point by random
        flows = self.sample(self.data_step)
        link_state = self._get_link_features(flows)
        path_state = self._get_path_state(link_state)
        demand_state = flows[:, self.action_effective_step:self.action_effective_step + 2].flatten() / np.max(
            self.link_capacity)
        demand_state = np.tile(demand_state, [self.num_agents, 1])
        return path_state, demand_state

    def reset_eval(self):
        for i, agent in enumerate(self.controllable_sd_list):
            self.agent_action[agent] = np.array([1 / self.num_paths[i]] * self.num_paths[i])
        self.agent_action = {}
        for i, agent in enumerate(self.controllable_sd_list):
            self.agent_action[agent] = np.array([1 / self.num_paths[i]] * self.num_paths[i])
        self.data_step = self.action_effective_step  # set the started flow point by random
        flows = self.sample(self.data_step)
        link_state = self._get_link_features(flows)
        path_state = self._get_path_state(link_state)
        demand_state = flows[:, self.action_effective_step:self.action_effective_step + 2].flatten() / np.max(
            self.link_capacity)
        demand_state = np.tile(demand_state, [self.num_agents, 1])
        return path_state, demand_state

    def step(self, action_all):
        self.data_step = (self.data_step + self.action_effective_step) % self.point_count
        self._set_action(action_all)
        flows = self.sample(self.data_step)
        link_state = self._get_link_features(flows[:, :self.action_effective_step])
        avg_util = np.sum(link_state, axis=-1) / self.action_effective_step
        max_util = np.max(avg_util)
        global_reward = float(1.0 - max_util)
        if max_util > 1.0:
            done = True
            global_reward -= 1.0
        else:
            done = False
            # calculate partial reward
        # rewards = []
        # for i, agent in enumerate(self.controllable_sd_list):
        #     path_mask = np.copy(self.paths_mask[agent])
        #     m_util = np.max(path_mask * np.expand_dims(avg_util, axis=0), axis=-1)
        #     rest_bd = 1.0 - m_util
        #     rew = np.sum(rest_bd * action_all[i])
        #     rewards.append(rew)
        # rewards.append(global_reward)
        rewards = [global_reward] + [global_reward] * len(self.controllable_sd_list)
        path_state = self._get_path_state(link_state)
        demand_state = flows[:, self.action_effective_step:self.action_effective_step + 2].flatten() / np.max(
            self.link_capacity)
        demand_state = np.tile(demand_state, [self.num_agents, 1])

        info = {"link usage average": avg_util}
        return rewards, path_state, demand_state, done, info

    def step_eval(self, action_all):
        self.data_step = self.data_step + self.action_effective_step
        if self.data_step >= self.point_count:
            done = True
        else:
            done = False
        self._set_action(action_all)
        flows = self.sample(self.data_step)
        link_state = self._get_link_features(flows[:, :self.action_effective_step])
        avg_util = np.sum(link_state, axis=-1) / self.action_effective_step
        max_util = np.max(avg_util)
        global_reward = float(1.0 - max_util)
        rewards = [global_reward] + [global_reward] * len(self.controllable_sd_list)
        path_state = self._get_path_state(link_state)
        demand_state = flows[:, self.action_effective_step:self.action_effective_step + 2].flatten() / np.max(
            self.link_capacity)
        demand_state = np.tile(demand_state, [self.num_agents, 1])

        info = {"link usage average": avg_util}
        return rewards, path_state, demand_state, done, info

    def _set_action(self, action_all):
        for i, agent in enumerate(self.controllable_sd_list):
            self.agent_action[agent] = action_all[i]
        # print(self.agent_action)

    def _get_link_features(self, flows):
        features = np.zeros([self.num_links, self.action_effective_step], dtype=np.float32)
        for s in range(self.action_effective_step):
            for i, agent in enumerate(self.controllable_sd_list):
                flow = flows[i, s]
                split_flow = flow * np.array(self.agent_action[agent])
                features[:, s] += np.sum(np.expand_dims(split_flow, -1) * self.paths_mask[agent], axis=0)
        features = features / np.expand_dims(self.link_capacity, axis=-1)
        # print("features",features)
        return features

    def _get_path_state(self, link_state):
        path_states = []

        for i, agent in enumerate(self.controllable_sd_list):
            rnn_feature = np.zeros([self.num_paths[i], self.max_len, self.action_effective_step],dtype=np.float32)
            for j in range(self.num_paths[i]):
                path_str = self.paths[agent][j]
                for k in range(len(path_str) - 1):
                    s, d = path_str[k], path_str[k + 1]
                    if s + d in self.link_names:
                        idx = self.link_names.index(s + d)
                    else:
                        idx = self.link_names.index(d + s)
                    rnn_feature[j, k] = link_state[idx, :]
            path_states.append(rnn_feature)
        return path_states

if __name__ == "__main__":
    env = Environment()
    a, b = env.reset()
    print(a, b)
    actions = []
    for i in env.num_paths:
        actions.append(np.array([1 / i] * i))
    for i in range(10):
        rewards, path_state, demand_state, done, info = env.step(actions)
        print(rewards, path_state, demand_state, done, info)