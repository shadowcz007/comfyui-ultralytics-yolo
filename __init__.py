from .nodes.detect import detectNode

NODE_CLASS_MAPPINGS = {
    "DetectByLabel": detectNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DetectByLabel": "Detect By Label",
}