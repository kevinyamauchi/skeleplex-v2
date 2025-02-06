"""Demo of launching the SkelePlex application."""

from qtpy.QtWidgets import QApplication

from skeleplex.app.model import SkelePlexApp

qapp = QApplication.instance() or QApplication([])
app = SkelePlexApp()
app.show()
qapp.exec_()
