# import time
# import sys
# from functools import wraps
# import torch


# class CacheMonitor:
#     _monitoring_enabled = False

#     def __init__(self):
#         self.stats = {}

#     @classmethod
#     def enable_monitoring(cls, enable=True):
#         cls._monitoring_enabled = enable

#     @classmethod
#     def monitor(cls, func):
#         @wraps(func)
#         def wrapper(instance, *args, **kwargs):
#             if not cls._monitoring_enabled:  # Return directly when monitoring is disabled
#                 return func(instance, *args, **kwargs)
#             # Access monitor via instance
#             return instance.cache_monitor._monitor_impl(func)(instance, *args, **kwargs)

#         return wrapper

#     def _monitor_impl(self, func):
#         @wraps(func)
#         def wrapped(instance, *args, **kwargs):
#             start_time = time.time()
#             result = func(instance, *args, **kwargs)
#             exec_time = time.time() - start_time

#             cache_info = func.cache_info()

#             # Initialize stats entry
#             func_name = func.__name__
#             if func_name not in self.stats:
#                 self.stats[func_name] = {
#                     "call_count": 0,
#                     "cache_hits": 0,
#                     "cache_misses": 0,
#                     "total_time": 0.0,
#                     "max_memory": 0,
#                 }

#             # Update statistics
#             stats = self.stats[func_name]
#             stats["call_count"] += 1
#             stats["cache_hits"] += cache_info.hits
#             stats["cache_misses"] += cache_info.misses
#             stats["total_time"] += exec_time

#             # Memory estimation
#             if result is not None:
#                 entry_mem = sys.getsizeof(result)
#                 if isinstance(result, torch.Tensor):
#                     entry_mem += result.element_size() * result.nelement()
#                 current_mem = cache_info.currsize * entry_mem
#                 stats["max_memory"] = max(stats["max_memory"], current_mem)

#             return result

#         return wrapped

#     def print_stats(self):
#         """Print full cache statistics"""
#         print("\n=== Cache Performance Statistics ===")
#         total = {"calls": 0, "misses": 0, "mem": 0}

#         for func_name, stats in self.stats.items():
#             hit_rate = (
#                 stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
#                 if (stats["cache_hits"] + stats["cache_misses"])
#                 else 0
#             )
#             avg_time = (
#                 stats["total_time"] / stats["call_count"] * 1e3
#                 if stats["call_count"]
#                 else 0
#             )

#             print(f"\nFunction: {func_name}")
#             print(f"  Total calls: {stats['call_count']}")
#             print(f"  Cache hits: {stats['cache_hits']} ({hit_rate:.1%})")
#             print(f"  Cache misses: {stats['cache_misses']}")
#             print(f"  Avg time/call: {avg_time:.2f} ms")
#             print(f"  Peak memory: {stats['max_memory'] / 1024**2:.2f} MB")

#             total["calls"] += stats["call_count"]
#             total["misses"] += stats["cache_misses"]
#             total["mem"] += stats["max_memory"]

#         print("\n=== Summary ===")
#         print(f"Total function calls: {total['calls']}")
#         print(f"Total cache misses: {total['misses']}")
#         print(f"Total peak memory: {total['mem'] / 1024**2:.2f} MB")

#     def reset_stats(self):
#         """Reset statistics"""
#         self.stats = {}
import time
import sys
from functools import wraps
import torch


class CacheMonitor:
    _monitoring_enabled = False

    def __init__(self):
        self.stats = {}

    @classmethod
    def enable_monitoring(cls, enable=True):
        cls._monitoring_enabled = enable

    @classmethod
    def monitor(cls, func):
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            if (
                not cls._monitoring_enabled
            ):  # Return directly when monitoring is disabled
                return func(instance, *args, **kwargs)
            # Access monitor via instance
            return instance.cache_monitor._monitor_impl(func)(instance, *args, **kwargs)

        return wrapper

    def _monitor_impl(self, func):
        @wraps(func)
        def wrapped(instance, *args, **kwargs):
            start_time = time.time()
            result = func(instance, *args, **kwargs)
            exec_time = time.time() - start_time

            if hasattr(func, "cache_info"):
                cache_info = func.cache_info()
                hits = cache_info.hits
                misses = cache_info.misses
                currsize = cache_info.currsize
            else:
                hits = 0
                misses = 0
                currsize = 0

            # Initialize stats entry
            func_name = func.__name__
            if func_name not in self.stats:
                self.stats[func_name] = {
                    "call_count": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "total_time": 0.0,
                    "max_memory": 0,
                }

            # Update statistics
            stats = self.stats[func_name]
            stats["call_count"] += 1
            stats["cache_hits"] += hits
            stats["cache_misses"] += misses
            stats["total_time"] += exec_time

            # Memory estimation
            if result is not None:
                entry_mem = sys.getsizeof(result)
                if isinstance(result, torch.Tensor):
                    entry_mem += result.element_size() * result.nelement()
                current_mem = currsize * entry_mem
                stats["max_memory"] = max(stats["max_memory"], current_mem)

            return result

        return wrapped

    def print_stats(self):
        """Print full cache statistics"""
        print("\n=== Cache Performance Statistics ===")
        total = {"calls": 0, "misses": 0, "mem": 0}

        for func_name, stats in self.stats.items():
            hit_rate = (
                stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
                if (stats["cache_hits"] + stats["cache_misses"])
                else 0
            )
            avg_time = (
                stats["total_time"] / stats["call_count"] * 1e3
                if stats["call_count"]
                else 0
            )

            print(f"\nFunction: {func_name}")
            print(f"  Total calls: {stats['call_count']}")
            print(f"  Cache hits: {stats['cache_hits']} ({hit_rate:.1%})")
            print(f"  Cache misses: {stats['cache_misses']}")
            print(f"  Avg time/call: {avg_time:.2f} ms")
            print(f"  Peak memory: {stats['max_memory'] / 1024**2:.2f} MB")

            total["calls"] += stats["call_count"]
            total["misses"] += stats["cache_misses"]
            total["mem"] += stats["max_memory"]

        print("\n=== Summary ===")
        print(f"Total function calls: {total['calls']}")
        print(f"Total cache misses: {total['misses']}")
        print(f"Total peak memory: {total['mem'] / 1024**2:.2f} MB")

        def reset_stats(self):
            """Reset statistics"""
            self.stats = {}
