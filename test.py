import os
import PySide6

# 잘못된 부분: PySide6.file
# 수정된 부분:
plugin_path = os.path.join(
    os.path.dirname(PySide6.__file__),
    "plugins",
    "platforms"
)
print(plugin_path)
