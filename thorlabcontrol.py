import wx
import wx.lib.activex
import time
from util.exposure_fusion import merge_hdr_images
import thorlabs_apt as apt
import numpy as np
import cv2


#
class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(600, 400))

        # 创建 ActiveX 控件实例
        self.gd = wx.lib.activex.ActiveXCtrl(self, 'DATARAYOCX.GetDataCtrl.1')
        self.gd.ctrl.StartDriver()
        self.gd.ctrl.StartDevice()
        b1 = wx.lib.activex.ActiveXCtrl(parent=self, size=(100, 50), axID='DATARAYOCX.ButtonCtrl.1')
        b1.ctrl.ButtonID = 297
        wx.lib.activex.ActiveXCtrl(parent=self, size=(250, 250), axID='DATARAYOCX.CCDimageCtrl.1')

        time.sleep(1)
        self.Show()

    def exposure(self, time=1):
        self.gd.ctrl.SetTargetCameraExposure(0, time)

    def getWinCamdata(self, binning = 1):
        # 调用 GetWinCamDataAsVariant 方法
        rows = self.gd.ctrl.GetVerticalPixels()
        cols = self.gd.ctrl.GetHorizontalPixels()
        data = self.gd.ctrl.GetWinCamDataAsVariant()
        return np.array(data).reshape((rows//binning, cols//binning))

    def StopDevice(self):
        self.gd.ctrl.StopDevice()

    def StartDevice(self):
        self.gd.ctrl.StartDevice()


#        print(self.gd.ctrl.GetVerticalPixels(),self.gd.ctrl.GetHorizontalPixels())
# if __name__ == "__main__":
# #
#     app = wx.App(False)
#     frame = MyFrame(None, "WinCam Data Viewer")
#     time.sleep(20)
#     frame.exposure(time = 0.05)
#     print(frame.getWinCamdata())
#     app.MainLoop()
if __name__ == "__main__":

    SN1 = 27601921
    SN2 = 27601945
    apt.list_available_devices()
    stage1 = apt.Motor(SN1)
    stage2 = apt.Motor(SN2)
    stage1.set_velocity_parameters(min_vel=0, accn=1, max_vel=2)
    stage2.set_velocity_parameters(min_vel=0, accn=1, max_vel=2)
    stage1.move_home(True)
    stage2.move_home(True)
    # 扫描坐标点设置
    original = [10, 14.5] #x+探针向下，y+探针向左
    n = 5 #10
    m = 5 #10`
    step = 0.1 #0.4
    x = np.arange(original[0], original[0] + n * step , step)
    y = np.arange(original[1], original[1] + m * step , step)

    Y, X = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()

    for i in range(1, 2 * n, 2):
        Y[i * m:i * m + m] = np.flip(Y[i * m:i * m + m])
        X[i * m:i * m + m] = np.flip(X[i * m:i * m + m])

    coords = np.column_stack((X, Y))

    # 添加随机偏移量
    offset_range = step / 4
    random_offset = (2 * offset_range) * np.random.rand(coords.shape[0], coords.shape[1]) - offset_range
    coords = coords + random_offset
    print(coords.shape)
    # stage1.move_to(origon[1], blocking=True)
    # stage2.move_to(origon[0], blocking=True)
    # print('MOVE END')
    ##ccd初始化#####
    app = wx.App(False)
    cam = MyFrame(None, "WinCam Data Viewer")
    time.sleep(1)
    bin = 2  ##像素binning参数
    rows, cols = cam.getWinCamdata(binning=bin).shape[0], cam.getWinCamdata(binning=bin).shape[1]
    print(rows, cols)
    diffset = np.zeros(shape=(n * m, rows, cols))
    temp1 = np.zeros(shape=(rows, cols))
    temp2 = np.zeros(shape=(rows, cols))
    temp3 = np.zeros(shape=(rows,cols))
    temp4 = np.zeros(shape=(rows, cols))
    temp5 = np.zeros(shape=(rows, cols))
    exposuretime1 = 4
    exposuretime2 = 8
    exposuretime3 = 8
    # exposuretime4 = 40
    # exposuretime5 = 50
    exposuretime = exposuretime1 +exposuretime2
    # 扫描开始
    for i in range(m * n):
        stage1.move_to(coords[i, 0], blocking=True)
        stage2.move_to(coords[i, 1], blocking=True)
        # 采集图像
        # cam.StartDevice()
        # time.sleep(1)
        cam.exposure(time = exposuretime1)
        temp1 = cam.getWinCamdata(binning=bin)
        cam.exposure(time = exposuretime2 )
        time.sleep(0.5)
        temp2 = cam.getWinCamdata(binning=bin)
        # cam.exposure(time=exposuretime3)
        # temp3 = cam.getWinCamdata(binning=bin)

        # cam.exposure(time=exposuretime4)
        # temp4 = cam.getWinCamdata(binning=bin)
        # cam.exposure(time=exposuretime5)
        # temp5 = cam.getWinCamdata(binning=bin)
        mask1 = temp2 > 2**16*0.7
        mask2 = temp2 < 2**16*0.3
        temp2[mask1] = temp1[mask1] * (exposuretime2 / exposuretime1)
        # temp2[mask2] = temp3[mask2] * (exposuretime2 / exposuretime3)
        temp = temp1 * exposuretime1 / exposuretime + temp2 *exposuretime2/exposuretime
        # diffset[i, :, :] = temp
        # temp = merge_hdr_images([temp1,temp2,temp3],[exposuretime1,exposuretime2,exposuretime3],exposuretime2)
        diffset[i, :, :] = cv2.flip(temp, 0)  # 垂直翻转,由于ccd采集的图像并不是看到的而是翻转后的因此需要翻转回来
        diffset[i, :, :] =  cv2.rotate(diffset[i, :, :], rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)##ccd旋转之后采集的，需要矫正回去
        cv2.waitKey(0)
        print(f'第{i + 1}个点，正在采集')
        print(coords[i, 0])
        print(coords[i, 1])

    stage1.move_home(True)
    stage2.move_home(True)
    print("数据保存中.......")
    # 保存坐标信息
    np.save('positions_test.npy', coords)
    np.save('diffset_test.npy', diffset)
    print("采集结束，请关闭显示界面")
    # app.MainLoop()
