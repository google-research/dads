# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import gym
from gym import Wrapper
from gym.wrappers.monitoring import video_recorder

class VideoWrapper(Wrapper):

  def __init__(self, env, base_path, base_name=None, new_video_every_reset=False):
    super(VideoWrapper, self).__init__(env)

    self._base_path = base_path
    self._base_name = base_name

    self._new_video_every_reset = new_video_every_reset
    if self._new_video_every_reset:
      self._counter = 0
      self._recorder = None
    else:
      if self._base_name is not None:
        self._vid_name = os.path.join(self._base_path, self._base_name)
      else:
        self._vid_name = self._base_path
      self._recorder = video_recorder.VideoRecorder(self.env, path=self._vid_name + '.mp4')

  def reset(self):
    if self._new_video_every_reset:
      if self._recorder is not None:
        self._recorder.close()

      self._counter += 1
      if self._base_name is not None:
        self._vid_name = os.path.join(self._base_path, self._base_name + '_' + str(self._counter))
      else:
        self._vid_name = self._base_path + '_' + str(self._counter)

      self._recorder = video_recorder.VideoRecorder(self.env, path=self._vid_name + '.mp4')

    return self.env.reset()

  def step(self, action):
    self._recorder.capture_frame()
    return self.env.step(action)

  def close(self):
    self._recorder.encoder.proc.stdin.flush()
    self._recorder.close()
    return self.env.close()