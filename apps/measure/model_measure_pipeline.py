from abc import ABC, abstractmethod
import numpy as np
import asyncio
import time  # Import time module for task timing


class ModelMeasurePipeline(ABC):
    def __init__(self, model_name: str, **kwargs):
        """
        Initializes the model measure with the model name, data, and optional arguments like model size.
        :param model_name: The name of the model being used.
        :param data: Input data to be processed.
        :param kwargs: Additional arguments such as model size.
        """
        self.model_name = model_name
        self._has_size = False
        if "size_list" in kwargs:
            self._has_size = True
            self.size_list = kwargs["size_list"]

        self.data = None
        self.load_data()
        if self.data is None:
            raise Exception("No data loaded")

    def load_data(self):
        raise NotImplementedError("load_data needed to be implemented")
    def _default_add_noise(self):
        processed_data = self.data
        noise = np.random.normal(0, 0.1, self.data.shape)
        processed_data += noise
        return processed_data

    def preprocess(self, add_noise: bool = False):
        """
        Preprocess the input data. Optional argument to add noise to the data.
        :param add_noise: Whether to add noise to the data.
        :return: Preprocessed data.
        """
        processed_data = self.data
        if add_noise:
            processed_data = self._default_add_noise()
        return processed_data

    @abstractmethod
    async def _create_model_task(self, *args):
        """
        Abstract method for running the model task. Must be implemented by subclasses.
        :param args: Arguments passed to the model.
        :return: Output data from the model.
        """
        raise NotImplementedError("_create_model_task needs to be implemented!")

    def _default_normalizer(self, results):
        min_val = np.min(results)
        max_val = np.max(results)
        return (results - min_val) / (max_val - min_val)

    async def _normalize_results(self, results):
        """
        Normalize the results to a common scale (0 to 1).
        :param results: Raw results from the model.
        :return: Normalized results.
        """
        return self._default_normalizer(results)

    def calculate_metrics(self, predictions, ground_truth, metric_func):
        """
        Calculates the performance metrics using a custom metric function.
        :param predictions: Model predictions.
        :param ground_truth: True values.
        :param metric_func: Custom function to calculate metrics.
        :return: Calculated metrics.
        """
        return metric_func(predictions, ground_truth)

    async def _timed_model_task(self, *args):
        """
        Executes the model task and measures the execution time for each task.
        :param args: Arguments passed to the model tasks.
        :return: Tuple of (task result, time taken for task).
        """
        start_time = time.time()  # Start timing the task
        result = await self._create_model_task(*args)
        end_time = time.time()  # End timing the task
        elapsed_time = end_time - start_time
        print(f"Task completed in {elapsed_time:.4f} seconds.")  # Print or store the elapsed time
        return result, elapsed_time  # Return both the result and time taken

    async def execute_tasks(self, *args):
        """
        Execute the model tasks asynchronously, time them, and normalize the results.
        :param args: Arguments passed to the model tasks.
        :return: Normalized results from all model tasks and their respective execution times.
        """
        tasks = [self._timed_model_task(*args) for _ in range(len(args))]
        results_with_time = await asyncio.gather(*tasks)  # Gather results and times
        results = [res[0] for res in results_with_time]  # Extract results
        times = [res[1] for res in results_with_time]  # Extract time taken for each task
        normalized_results = await self._normalize_results(results)
        return normalized_results, times  # Return normalized results and timings


