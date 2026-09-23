from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["build_metric"]


def build_metric(config):
    from .det_metric import DetMetric

    support_dict = ["DetMetric"]
    try:
        from .rec_metric import RecMetric

        support_dict.append("RecMetric")
    except Exception:
        pass

    config = dict(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "metric only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class
