"""Demo of launching the SkelePlex application."""

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from skeleplex.app.data import DataManager, SkeletonDataPaths
from skeleplex.app.model import SkelePlexApp

graph_path = "e16_skeleplex_v2.json"
data_manager = DataManager(file_paths=SkeletonDataPaths(skeleton_graph=graph_path))

qapp = QApplication.instance() or QApplication([])
app = SkelePlexApp(data=data_manager)
app.show()

t = QTimer()
t.singleShot(100, app.look_at_skeleton)

qapp.exec_()
