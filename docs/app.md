# SkelePlex application

## Events
### Loading data
This sequence is for loading data from the GUI. The events are connected in skeleplex.app.model.SkeleplexApp._connect_data_events()

1. Load data button clicked: skeleplex.app.qt.app_controls.AppControlsWidget.load_data_widget.called()
1. Event received by skeleplex.app.model.SkeleplexApp._load_data_clicked(). The new paths are set in the data manager (skeleplex.app.data.DataManager) and the data manager emits the DataManager.events.data() event.
1. The data() event is received by the SkeleplexApp.load_main_viewer() and the data are loaded into the viewer.