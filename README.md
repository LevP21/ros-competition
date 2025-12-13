# ros-competition
---
### Порядок запуска
В файле ros-competition/src/my_robot/robot_bringup/worlds/course.sdf пути из блока <include> заменить на
```<path_to_folder>/ros-competition/install/robot_bringup/share/robot_bringup/worlds<object_name>```

В файле ros-competition/src/autorace_core_PIVO/autorace_core_PIVO/base_node.py заменить путь до сохраняемого видео на любой подходящий.

В изолированном окружении:
```colcon build```
```source <path_to_folder>/ros-competition/install/local_setup.bash```

Для появления робота перед различными испытаниями менять координаты в файле ros-competition/src/my_robot/robot_bringup/launch/autorace_2025.launch.py в соответствии с комментариями

После этого не забыть пересобрать пакет