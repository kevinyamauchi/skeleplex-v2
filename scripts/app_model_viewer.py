"""Demo of launching the SkelePlex application."""

from qtpy.QtCore import QTimer

import skeleplex
from skeleplex.app import SkelePlexApp
from skeleplex.app._data import DataManager, SkeletonDataPaths

graph_path = "skeleton_graph_spline.json"
data_manager = DataManager(file_paths=SkeletonDataPaths(skeleton_graph=graph_path))

app = SkelePlexApp(data=data_manager)
app.show()

t = QTimer()
t.singleShot(100, app.look_at_skeleton)

skeleplex.app.run()
