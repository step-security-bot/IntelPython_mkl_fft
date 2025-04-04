#!/usr/bin/env python
# Copyright (c) 2025, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import contextlib
import contextvars
import operator

import mkl

__all__ = ["get_workers", "set_workers"]


class _cpu_max_threads_count:
    def __init__(self):
        self.cpu_count = None
        self.max_threads_count = None

    def get_cpu_count(self):
        if self.cpu_count is None:
            max_threads = self.get_max_threads_count()
            self.cpu_count = max_threads
        return self.cpu_count

    def get_max_threads_count(self):
        if self.max_threads_count is None:
            # pylint: disable=no-member
            self.max_threads_count = mkl.get_max_threads()

        return self.max_threads_count


class _workers_data:
    def __init__(self, workers=None):
        if workers:
            self.workers_ = workers
        else:
            self.workers_ = _cpu_max_threads_count().get_cpu_count()
        self.workers_ = operator.index(self.workers_)

    @property
    def workers(self):
        return self.workers_

    @workers.setter
    def workers(self, workers_val):
        self.workerks_ = operator.index(workers_val)


_workers_global_settings = contextvars.ContextVar(
    "scipy_backend_workers", default=_workers_data()
)


def get_workers():
    """
    Gets the number of workers used by mkl_fft by default.

    For full documentation refer to `scipy.fft.get_workers`.
    """
    return _workers_global_settings.get().workers


@contextlib.contextmanager
def set_workers(n_workers):
    """
    Set the value of workers used by default, returns the previous value.

    For full documentation refer to `scipy.fft.set_workers`.
    """
    nw = operator.index(n_workers)
    token = None
    try:
        new_wd = _workers_data(nw)
        token = _workers_global_settings.set(new_wd)
        yield
    finally:
        if token:
            _workers_global_settings.reset(token)
        else:
            raise ValueError
