"""Actions to be registered for the application."""

from app_model.types import Action, KeyBindingRule, KeyCode, KeyMod, MenuRule

from . import functions
from ._constants import CommandId, MenuId

ACTIONS: list[Action] = [
    # Action(
    #     id=CommandId.OPEN,
    #     title="Open",
    #     icon="fa6-solid:folder-open",
    #     callback=functions.open_file,
    #     menus=[MenuRule(id=MenuId.FILE)],
    #     keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.KeyO)],
    # ),
    Action(
        id=CommandId.CLOSE,
        title="Close",
        icon="fa-solid:window-close",
        callback=functions.close,
        menus=[MenuRule(id=MenuId.FILE)],
        keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.KeyW)],
    ),
    # Action(
    #     id=CommandId.UNDO,
    #     title="Undo",
    #     icon="fa-solid:undo",
    #     callback=functions.undo,
    #     menus=[MenuRule(id=MenuId.EDIT, group="1_undo_redo")],
    #     keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.KeyZ)],
    # ),
    # Action(
    #     id=CommandId.REDO,
    #     title="Redo",
    #     icon="fa6-solid:rotate-right",
    #     callback=functions.redo,
    #     menus=[MenuRule(id=MenuId.EDIT, group="1_undo_redo")],
    #     keybindings=[
    #         KeyBindingRule(primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ)
    #     ],
    # ),
]
