import time
from collections import defaultdict

from .log_metrics import LogMetrics


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all stats."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update stats given input val and n."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Tracker(object):
    """A class to track running stats and metrics.

    Also responsible for posting metrics to the API/Dashboard

    Args:
        metrics (list): List of metrics objects
        run_id (int): The id of the current run
        rank (int): The rank of this worker node
        goal(func): A task goal to check for when logging metrics"""

    def __init__(
        self,
        metrics,
        run_id,
        rank,
        goal=None,
        communication_steps=["opt_step"],
        compute_steps=["fwd_pass", "comp_loss", "backprop"],
    ):
        self.batch_times = []
        self.validation_times = []
        self.epoch_stats = {}
        self.epoch_metrics = defaultdict(list)
        self.records = []
        self.history = []
        self.cumulative_train_time = []
        self.cumulative_compute_time = []
        self.cumulative_communication_time = []
        self.cumulative_val_time = []
        self.best_epoch = 0
        self.current_epoch = 0
        self.best_metric_value = 0
        self.is_training = True

        self.communication_steps = []
        self.compute_steps = []

        self.train_prefix = "train_"
        self.val_prefix = "val_"

        self.start_time = None
        self.metrics = metrics
        self.run_id = run_id
        self.rank = rank
        self.reset_epoch_stats()
        self.goal = goal
        self.goal_reached = False

        self.primary_metric = metrics[0]

        if not isinstance(communication_steps, list):
            communication_steps = [communication_steps]

        self.communication_steps = communication_steps

        if not isinstance(compute_steps, list):
            compute_steps = [compute_steps]

        self.compute_steps = compute_steps

    def start(self):
        """ Starts Tracking """
        if self.start_time is not None:
            raise Exception("Tracking already started")

        self.start_time = time.time()

    def train(self):
        """ Switch Tracker to training mode"""
        self.is_training = True

    def validation(self):
        """Switch Tracker to validation mode"""
        self.is_training = False

    def validation_start(self):
        """Start validation step & timer"""
        self.validation_times = [("start", time.time())]

    def validation_end(self):
        """End Validation step"""
        self.validation_times = [("end", time.time())]
        self.cumulative_val_time.append(
            self.validation_times[-1][1] - self.validation_times[0][1]
        )

    def batch_start(self):
        """Start a training batch"""
        self.batch_times = []
        self.record_batch_step("start")

    def batch_end(self):
        """End a training batch and calculate time spent"""
        self.record_batch_step("end")

        if len(self.batch_times) > 1:
            metric = "CumulativeTrainTimeEpoch"

            self.cumulative_train_time.append(
                self.batch_times[-1][1] - self.batch_times[0][1]
            )
            time_diff = self.cumulative_train_time[-1]

            # batch times
            self.batch_times.sort(key=lambda x: x[1])

            tracker_stats = list(zip(self.batch_times[:-1], self.batch_times[1:]))

            if self.compute_steps:
                compute_times = [
                    t[1][1] - t[0][1]
                    for t in tracker_stats
                    if t[1][0] in self.compute_steps
                ]
                self.cumulative_compute_time.append(sum(compute_times))

            if self.communication_steps:
                communication_times = [
                    t[1][1] - t[0][1]
                    for t in tracker_stats
                    if t[1][0] in self.communication_steps
                ]
                self.cumulative_communication_time.append(sum(communication_times))

            if len(self.epoch_metrics[metric]) < self.current_epoch + 1:
                self.epoch_metrics[metric].append(0.0)

            self.epoch_metrics[metric][self.current_epoch] += time_diff

            for (_, t1), (name, t2) in tracker_stats:
                time_diff = t2 - t1

                if len(self.epoch_metrics[name]) < self.current_epoch + 1:
                    self.epoch_metrics[name].append(0.0)

                self.epoch_metrics[name][self.current_epoch] += time_diff

    def epoch_end(self):
        """Ends a training epoch and logs epoch metrics"""
        for k, v in dict(self.epoch_metrics).items():
            LogMetrics.log(self.run_id, self.rank, self.current_epoch, k, v[-1])

        self.current_epoch += 1

        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        """ Resets all epoch stats"""
        stat_names = [k for k in ["loss"] + [m.name for m in self.metrics]]

        stat_names = [self.train_prefix + k for k in stat_names] + [
            self.val_prefix + k for k in stat_names
        ]

        self.epoch_stats = {k: AverageMeter() for k in stat_names}

    def record_batch_step(self, name):
        """Records a specific batch step for timing (e.g. "backpropagation" or "model_forward

        Args:
            name (str): The name of the step to record"""
        self.batch_times.append((name, time.time()))

    def record_stat(self, name, value, n=1, log_to_api=False):
        """Records a stat value

        Args:
            name (str): The name of the stat to record
            value (number): The stat value to record
            n (int): The number of individual samples this stat is made up of
                     in case of an average value
            log_to_api (bool): Whether to submit the stat to the Dashboard API, default:False
        """
        prefix = self.train_prefix

        if not self.is_training:
            prefix = self.val_prefix

        name = prefix + name

        if name not in self.epoch_stats:
            self.epoch_stats[name] = AverageMeter()

        self.epoch_stats[name].update(value, n)

        self.history.append((self.run_id, self.rank, name, value, time.time()))

        if log_to_api:
            LogMetrics.log(self.run_id, self.rank, self.current_epoch, name, value)

        if self.goal and self.rank == 0:
            goal_result = self.goal(name, value, self)

            if goal_result is not None and not self.goal_reached:
                self.goal_reached = True
                print("goal reached1")
                print(log_to_api)
                print(goal_result)

                if log_to_api:
                    LogMetrics.log(
                        self.run_id,
                        self.rank,
                        self.current_epoch,
                        "TaskResult",
                        goal_result,
                    )

                    LogMetrics.log(
                        self.run_id,
                        self.rank,
                        self.current_epoch,
                        "TotalCumulativeTrainTime",
                        self.get_total_train_time(),
                    )

                    metrics = dict(self.epoch_metrics).items()
                    metrics = sorted(metrics, key=lambda k: k[0])

                    for k, v in metrics:
                        LogMetrics.log(
                            self.run_id,
                            self.rank,
                            self.current_epoch,
                            "global_cum_{}".format(k),
                            sum(v),
                        )

    def record_loss(self, value, n=1, log_to_api=False):
        """Records a loss value

        Args:
            value (number): The stat value to record
            n (int): The number of individual losses this stat is made up of
                     in case of an average value (e.g. if batch loss use batch size)
            log_to_api (bool): Whether to submit the stat to the Dashboard API, default:False
        """
        self.record_stat("loss", value, n, log_to_api)

    def record_metric(self, metric, value, n=1, log_to_api=False):
        """Records a metric value

        Args:
            metric (`obj`:metric): The metric to log
            value (number): The stat value to record
            n (int): The number of individual samples this stat is made up of
                     in case of an average value
            log_to_api (bool): Whether to submit the stat to the Dashboard API, default:False
        """
        self.record_stat(metric.name, value, n, log_to_api)

        if metric.name == self.primary_metric.name and not self.is_training:
            self.update_primary_metric(value)

    def update_primary_metric(self, new_metric_value):
        """ Updates the primary (main) metric

        Args:
            new_metric_value (number): The new value of the metric
        """

        if new_metric_value > self.best_metric_value:
            self.best_metric_value = new_metric_value
            self.best_epoch = self.current_epoch

    def is_best(self):
        """ Whether the current epoch is the best epoch so far"""
        return self.current_epoch == self.best_epoch

    def get_total_train_time(self):
        return sum(self.cumulative_train_time)

    def get_total_compute_time(self):
        return sum(self.cumulative_compute_time)

    def get_total_communication_time(self):
        return sum(self.cumulative_communication_time)

    def get_total_val_time(self):
        return sum(self.cumulative_val_time)

    def __str__(self):
        """ String representation of current epoch"""
        prefix = self.train_prefix

        if not self.is_training:
            prefix = self.val_prefix

        # loss
        str_builder = ["loss={:6.2e}".format(self.epoch_stats[prefix + "loss"].avg)]

        # metrics
        for metric in self.metrics:
            str_builder.append(
                "{} {:.2e}".format(
                    metric.name, self.epoch_stats[prefix + metric.name].avg
                )
            )

        # batch times
        self.batch_times.sort(key=lambda x: x[1])

        tracker_stats = zip(self.batch_times[:-1], self.batch_times[1:])

        str_builder.append(
            " | ".join(
                "{} {:.1e}".format(name, t2 - t1)
                for (_, t1), (name, t2) in tracker_stats
            )
        )

        return " | ".join(str_builder)
