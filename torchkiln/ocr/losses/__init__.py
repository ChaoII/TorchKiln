from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["build_loss"]


def build_loss(config):
    from .det_db_loss import DBLoss

    support_dict = ["DBLoss"]
    try:
        from .rec_ctc_loss import CTCLoss
        from .rec_multi_loss import MultiLoss
        from .rec_nrtr_loss import NRTRLoss
        from .rec_sar_loss import SARLoss

        support_dict += ["CTCLoss", "MultiLoss", "NRTRLoss", "SARLoss"]
    except Exception:
        pass

    config = dict(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "loss only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class
