from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor

import matplotlib.image

class MyApp2(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        self.scene = self.loader.loadModel("models/box")
        self.scene.reparentTo(self.render)
        
        self.params = [0.3, 0.3, 0.3,
                       0.0, 2.0, 0.0,
                       60.0, -30.0, 0.0]
        self.grads = [0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0]
        
        self.setScene()
        
        self.gold = matplotlib.image.imread('gold.jpg')
        self.gold = self.gold.astype(float) / 255.0

    def setScene(self):
        self.scene.setScale(self.params[0], self.params[1], self.params[2])
        self.scene.setPos(self.params[3], self.params[4], self.params[5])
        self.scene.setHpr(self.params[6], self.params[7], self.params[8])

    def gradientDescent(self, task):
        if task.frame == 0:
            return Task.cont
        if task.frame % 10 != 0:
            ix = task.frame % 10
            self.params[ix] += DELTA
            self.setScene()


app = MyApp2()
app.run()

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = self.loader.loadModel("models/box")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.5, 0.5, 0.5)
        self.scene.setPos(-0.25, 2, -0.25)
        self.scene.setHpr(30, 30, 0)

        self.hehe = 0
        self.taskMgr.add(self.takeScreenshot, "TakeScreenshotTask")

    def takeScreenshot(self, task):
        self.hehe += 1
        if self.hehe != 5:
            return Task.cont
        self.screenshot('hehe')
        return Task.cont

#        # Add the spinCameraTask procedure to the task manager.
#        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")
#
#    # Define a procedure to move the camera.
#    def spinCameraTask(self, task):
#        angleDegrees = task.time * 6.0
#        angleRadians = angleDegrees * (pi / 180.0)
#        self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
#        self.camera.setHpr(angleDegrees, 0, 0)
#        return Task.cont


app = MyApp()
app.run()
