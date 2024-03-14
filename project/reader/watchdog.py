import os
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.observers.api import EventQueue
import time
q = queue.Queue()
class MyHandler(FileSystemEventHandler):
    @staticmethod
    def work(event):
        flag = False
        extension = ('.jpg', '.jpeg', '.png', 'JPEG')

        if not event.is_directory and event.src_path.endswith(extension):
            flag = True
        return flag

    def on_created(self, event):
        if self.work(event):

            print(f"Файл {event.src_path} добавлен в очередь")
            q.put(event.src_path)