"""Module for handling data in the SkelePlex application."""

import logging

from psygnal import EventedModel, Signal, SignalGroup
from pydantic.types import FilePath

from skeleplex.graph import SkeletonGraph

log = logging.getLogger(__name__)


class SkeletonDataPaths(EventedModel):
    """A class storing the state of the skeleton dataset.

    Parameters
    ----------
    image : FilePath | None
        The path to the image file.
    segmentation : FilePath | None
        The path to the segmentation image file.
    skeleton_graph : FilePath
        The path to the skeleton graph file.
    """

    image: FilePath | None = None
    segmentation: FilePath | None = None
    skeleton_graph: FilePath | None = None


class DataEvents(SignalGroup):
    """Events for the DataManager class."""

    data = Signal()


class DataManager:
    """A class to manage data."""

    events = DataEvents()

    def __init__(
        self,
        file_paths: SkeletonDataPaths,
    ) -> None:
        self._file_paths = file_paths

        # initialize the data
        self._skeleton_graph: SkeletonGraph | None = None

    @property
    def file_paths(self) -> SkeletonDataPaths:
        """Get the file paths."""
        return self._file_paths

    @property
    def skeleton_graph(self) -> SkeletonGraph:
        """Get the skeleton graph."""
        return self._skeleton_graph

    def load(self) -> None:
        """Load data."""
        # load the skeleton graph
        if self.file_paths.skeleton_graph:
            log.info(f"Loading skeleton graph from {self.file_paths.skeleton_graph}")
            self._skeleton_graph = SkeletonGraph.from_json_file(
                self.file_paths.skeleton_graph
            )
        else:
            log.info("No skeleton graph loaded.")
            self._skeleton_graph = None

        self.events.data.emit()

    def to_dict(self) -> dict:
        """Convert to json-serializable dictionary."""
        return self._data.to_dict()
