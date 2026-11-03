"""YOLOX-X person architecture for the uploaded GlareMOT epoch15 checkpoint.

Inference only. Import from the TrackTrack/ByteTrack YOLOX fork, not the
Ultralytics package. Preprocessing belongs to exsort.cache, not this class.
"""
from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.test_size = (800, 1440)
        self.input_size = (800, 1440)
        self.test_conf = .001
        self.nmsthre = .65
