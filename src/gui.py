"""danielsinkin97@gmail.com"""

import dearpygui.dearpygui as dpg


class GUI:
    """The main user interface."""

    def __init__(self):
        self._initialize_gui()

    def _generic_button_callback(self, sender, app_data, user_data) -> None:
        print(f"Pressed button {user_data}")

    def _selection_layout(self) -> None:
        for idx in range(10):
            button_name = f"Example {idx+1}"
            dpg.add_button(
                label=button_name,
                callback=self._generic_button_callback,
                user_data=idx + 1,
            )

    def _initialize_gui(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Machine Learning Application")
        dpg.setup_dearpygui()

        with dpg.window(label="Machine Learning"):
            self._selection_layout()

        dpg.show_viewport()
        dpg.start_dearpygui()

    def __del__(self) -> None:
        print("Cleaning up GUI, deleting dgp now!")
        dpg.destroy_context()

    def _save_callback(self) -> None:
        print("Save clicked!")
