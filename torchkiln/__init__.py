"""torchkiln: YOLO-style tasks (cls/det/obb/seg/sem/depth) on the ptcore platform."""

#: 平台版本号。与 pyproject.toml 的 `version` 保持一致——对外的 `meta.json`
#: 与 `/api/v1/models` 契约都会带上它，消费方据此判断兼容性。
__version__ = "0.1.0"