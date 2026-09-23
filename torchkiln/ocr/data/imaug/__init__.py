from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .iaa_augment import IaaAugment
from .make_border_map import MakeBorderMap
from .make_shrink_map import MakeShrinkMap
from .random_crop_data import (
    EastRandomCropData,
    RandomCrop,
    RandomCropImgMask,
    is_poly_in_rect,
    is_poly_outside_rect,
    split_regions,
    random_select,
    region_wise_random_select,
    get_min_rotated_rect_side,
    get_min_quad_side,
    clip_poly_to_rect,
    crop_area,
)
from .copy_paste import (
    CopyPaste,
    get_rotate_crop_image,
    get_union,
    get_intersection,
    get_intersection_over_union,
    rotate_bbox,
)
from .ColorJitter import ColorJitter
from .random_perspective import RandomPerspective

from .rec_img_aug import (
    RecResizeImg,
    RecAug,
    RecConAug,
    BaseDataAugmentation,
    ClsResizeImg,
    VLRecResizeImg,
    RFLRecResizeImg,
    SRNRecResizeImg,
    SARRecResizeImg,
    PRENResizeImg,
    SPINRecResizeImg,
    GrayRecResizeImg,
    ABINetRecResizeImg,
    SVTRRecResizeImg,
    RobustScannerRecResizeImg,
    ABINetRecAug,
    SVTRRecAug,
    ParseQRecAug,
)
# from .randaugment import RandAugment
from .operators import *
from .label_ops import (
    DetLabelEncode,
    BaseRecLabelEncode,
    CTCLabelEncode,
    NRTRLabelEncode,
    SARLabelEncode,
    MultiLabelEncode,
)

# from .east_process import *
# from .sast_process import *
from .gen_table_mask import *

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops