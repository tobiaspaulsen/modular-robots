from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import uuid


class CustomSideChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        self.received_messages = []
        self.created_robot_module_keys = None
        self.wait_for_robot_string = True

    def on_message_received(self, msg: IncomingMessage, debug: bool = False) -> None:
        message = msg.read_string()
        self.received_messages.append(message)
        if debug:
            print(message)
        csv_mes = message.split(",")
        if csv_mes[0] == "[Unity]:[Module Information]":
            if debug:
                print("[Python]:", "received module information")
            csv_mes.pop(0)
            self.created_robot_module_keys = csv_mes
            self.wait_for_robot_string = False

    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)
