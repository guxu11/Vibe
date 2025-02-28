# Created by guxu at 2/11/25
from .. import PROJECT_BASE_PATH
import os
import time

class FileUploadService:
    def __init__(self):
        self.upload_base_path = os.path.join(PROJECT_BASE_PATH, "uploads")

    def _get_file_path(self, user_path, file_type):
        file_type_path = {
            "audio": os.path.join(user_path, "audio"),
            "image": os.path.join(user_path, "image"),
            "pdf": os.path.join(user_path, "pdf")
        }
        return file_type_path[file_type]

    def _generate_randomized_file_name(self, file_name):
        return f"{int(time.time())}_{file_name}"

    def save_file(self, file, metadata):
        file_type = metadata["file_type"]
        file_name = metadata["file_name"]
        user_id = metadata["user_id"]
        user_path = os.path.join(self.upload_base_path, user_id)
        randomized_file_name = self._generate_randomized_file_name(file_name)
        target_directory = self._get_file_path(user_path, file_type)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        file_path = os.path.join(target_directory, randomized_file_name)
        file.save(file_path)
        return file_path

