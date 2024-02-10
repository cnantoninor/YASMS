from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging


class DataDirectoryEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            logging.info("New directory created {}, event.src_path")


class Trainer:
    def __init__(self, train_dir: str) -> None:
        self.train_dir = train_dir

    def start(self):
        # Create an observer and attach the event handle
        observer = Observer()
        observer.schedule(DataDirectoryEventHandler(), self.train_dir, recursive=False)

        # Start the observer
        observer.start()
