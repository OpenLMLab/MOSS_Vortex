import threading
import queue
import json
import time
try:
    import thread
except ImportError:
    import _thread as thread

from mosec import Server, Worker
import websocket
from websocket import create_connection


class Inf(Worker):
    def __init__(self) -> None:
        super().__init__()
        self.client = create_connection("ws://moss.jingyijun.xyz:12442/api/ws/response")
        self.q = queue.SimpleQueue()
        threading.Thread(target=self.sync, daemon=True).start()
        self.format = {"status":None, "uuid":None, "offset":None, "output":None }
        
        threading.Thread(target=self.interval, daemon=True).start()

    def interval(self):
        check = time.time()
        while True:
            if (time.time() - check) // 30 >= 1:
                alive = self.format
                alive["status"] = 2
                self.client.send(json.dumps(alive))
                check += 30
    def sync(self):
        while True:
            msg = self.q.get()
            self.client.send(json.dumps(msg))

    def forward(self, data):
        for i in range(5):
            self.q.put(self.format)
        return data


if __name__ == "__main__":
    server = Server()
    server.append_worker(Inf, num=1, max_batch_size=8)
    server.run()
