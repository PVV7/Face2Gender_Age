import os
import threading
import queue
import time

file_queue = queue.Queue()


# Функция для наблюдения за папкой
def watch_folder(folder_path):
    while True:
        # if not os.listdir(folder_path):
        #     continue
        lst_img = os.listdir(folder_path)
        lst_img.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

        for file_name in lst_img:
            file_path = os.path.join(folder_path, file_name)
            if file_path not in file_queue.queue:
                file_queue.put(file_path)
        time.sleep(20)


# Функция для обработки файлов из очереди
def process_files():
    while True:
        file_path = file_queue.get()
        print(f"Processing file: {file_path}")
        time.sleep(3)
        # Add your code for processing the file here
        os.remove(file_path)



# Главная функция
if __name__ == "__main__":
    folder_path = "test"

    watcher_thread = threading.Thread(target=watch_folder, args=(folder_path, ))
    watcher_thread.start()


    processor_thread = threading.Thread(target=process_files)
    processor_thread.start()


