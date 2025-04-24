import numpy as np

class Particula:
  def __init__(self, x, y, vx, vy, massa):
    self.x     = [x]
    self.y     = [y]
    self.vx    = [vx]
    self.vy    = [vy]
    self.massa = massa

  def newton(self, fx, fy, dt):
    if self.y[-1] >= 0:
      # calculate acceleration
      ax = fx / self.massa
      ay = fy / self.massa
      # update velocity
      self.vx.append(self.vx[-1] + ax * dt)
      self.vy.append(self.vy[-1] + ay * dt)
      # update position
      self.x.append(self.x[-1] + self.vx[-1] * dt + 0.5 * ax * dt**2)
      self.y.append(self.y[-1] + self.vy[-1] * dt + 0.5 * ax * dt**2)