# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import json
import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "DetLabelEncode",
    "BaseRecLabelEncode",
    "CTCLabelEncode",
    "NRTRLabelEncode",
    "SARLabelEncode",
    "MultiLabelEncode",
]


class DetLabelEncode(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        label = data["label"]
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]["points"]
            txt = label[bno]["transcription"]
            boxes.append(box)
            txts.append(txt)
            if txt in ["*", "###"]:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool_)

        # Mark polygons with NaN coordinates as ignore to avoid downstream errors
        nan_mask = np.isnan(boxes).any(axis=(1, 2))
        txt_tags[nan_mask] = True

        data["polys"] = boxes
        data["texts"] = txts
        data["ignore_tags"] = txt_tags
        return data

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes


class BaseRecLabelEncode(object):
    """Convert between text-label and text-index"""

    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        lower=False,
    ):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower

        if character_dict_path is None:
            _LOGGER.warning(
                "The character_dict_path is None, model can only recognize number and lower letters"
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs
    ):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char
        )

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        data["length"] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data["label_ace"] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class NRTRLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs
    ):
        super(NRTRLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char
        )

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data["length"] = np.array(len(text))
        text.insert(0, 2)
        text.append(3)
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank", "<unk>", "<s>", "</s>"] + dict_character
        return dict_character


class SARLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs
    ):
        super(SARLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char
        )

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1

        return dict_character

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data["length"] = np.array(len(text))
        target = [self.start_idx] + text + [self.end_idx]
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]

        padded_text[: len(target)] = target
        data["label"] = np.array(padded_text)
        return data

    def get_ignored_tokens(self):
        return [self.padding_idx]


class MultiLabelEncode(BaseRecLabelEncode):
    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        gtc_encode=None,
        **kwargs,
    ):
        super(MultiLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char
        )

        self.ctc_encode = CTCLabelEncode(
            max_text_length, character_dict_path, use_space_char, **kwargs
        )
        self.gtc_encode_type = gtc_encode
        if gtc_encode is None:
            self.gtc_encode = SARLabelEncode(
                max_text_length, character_dict_path, use_space_char, **kwargs
            )
        else:
            self.gtc_encode = eval(gtc_encode)(
                max_text_length, character_dict_path, use_space_char, **kwargs
            )

    def __call__(self, data):
        data_ctc = copy.deepcopy(data)
        data_gtc = copy.deepcopy(data)
        data_out = dict()
        data_out["img_path"] = data.get("img_path", None)
        data_out["image"] = data["image"]
        ctc = self.ctc_encode.__call__(data_ctc)
        gtc = self.gtc_encode.__call__(data_gtc)
        if ctc is None or gtc is None:
            return None
        data_out["label_ctc"] = ctc["label"]
        if self.gtc_encode_type is not None:
            data_out["label_gtc"] = gtc["label"]
        else:
            data_out["label_sar"] = gtc["label"]
        data_out["length"] = ctc["length"]
        return data_out
