import carb
import omni.appwindow
import omni.kit.app

from carb.eventdispatcher import get_eventdispatcher, Event

class KeyboardControl:

    STOP_RECORDING_EVENT: str = "lerobot_so101_teleop.stop_recording"
    CANCEL_RECORDING_EVENT: str = "lerobot_so101_teleop.cancel_recording"

    def __init__(self):
        self.reset_world = False
        self.recording = False

        # Get the window to register keyboard callbacks
        self._window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._window.get_keyboard()

        # Register keyboard callbacks
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Keyboard event handler"""
        # Only process key press events
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_world = True
                self.stop_recording()
                print(f"[INFO]: Reset world...")
                return True

            if event.input.name == "S":
                if self.recording:
                    self.stop_recording()
                    return True

                
                self.recording = True
                print(f"[INFO]: Started recording!")
                return True

            if event.input.name == "C":
                if self.recording:
                    self.cancel_recording()
                    return True

        return False



    def cleanup(self):
        """Cleanup the keyboard interface"""
        if self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._sub_keyboard
            )
            self._sub_keyboard = None

    # This should not live in this class, but it works for now
    def stop_recording(self):
        if self.recording:
            print(f"[INFO]: Stopped recording.")
            self.recording = False

            omni.kit.app.queue_event(
                self.STOP_RECORDING_EVENT, 
                payload={}
                )
    def cancel_recording(self):
        if self.recording:
            print(f"[INFO]: Cancelled recording.")
            self.recording = False

            omni.kit.app.queue_event(
                self.CANCEL_RECORDING_EVENT, 
                payload={}
                )