import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import math
from typing import List,Optional
from numpy.typing import NDArray
from copy import deepcopy
from functools import partial
import matplotlib.animation as animation

SEED = 729608
random.seed(SEED)
np.random.seed(SEED)
int16 = partial(int,base=16)

def Hex2RGB(hex):
    r = int16(hex[1:3])
    g = int16(hex[3:5])
    b = int16(hex[5:7])
    return [r,g,b]

class Heart:
    IMAGE_ENLARGE = 10

    def __init__(self,num_pts:int,colors=['#FFBFF0','#FA3968','#EE41FF','#FF75AD','#FABCD4'],image:NDArray = np.zeros((480,640,3),dtype=np.uint8)):
        self.point_list = set()
        self.extra_point = set()
        self.inside = set()
        self.bg_halo = set()
        self.colors = colors
        self.num_pts = num_pts
        self.image = image
        self.cw = self.image.shape[1] / 2
        self.ch = self.image.shape[0] / 2
        self.heart_point_generate()
    def __len__(self):
        return len(self.point_list)

    def __str__(self): # stdout 版本
        division = 1
        division = int(division)
        px = np.linspace(-16.,17.,32*division)
        py = np.linspace(-15.,16.,32*division)
        
        heart_str = []
        for yj in py:
            temp = ""
            for xi in px:
                if self.whether_inter_heart(xi,yj):
                    temp += "*"
                else:
                    temp += " "
            if "*" in temp:
                heart_str.append(temp)
        heart_str = "\n".join(heart_str)
        return heart_str
    
    @staticmethod
    def whether_inter_heart(x,y):
        if x<0:
            sin_t = -1*(-1*x/16) ** (1/3)
        else:
            sin_t = (x/16) ** (1/3)
        
        # 防止开根号的计算误差
        sin_t = min(sin_t,1.)
        sin_t = max(sin_t,-1.)

        # 输入一个x，计算在心线上的对应的y，看输入的y点在不在区间范围内
        t = math.asin(sin_t)
        y_1 = -1*(13.* math.cos(t)-5 * math.cos(2 *t)-2 * math.cos(3*t)-math.cos(4 * t))

        # 如果sint不是1和-1，则还有一个t使得sint为这个值
        if sin_t != 1 and sin_t != -1:
            symmetric_axi = math.pi/2 if sin_t >0 else -1*math.pi / 2
            t = 2*symmetric_axi - t
            y_2 = -1*(13.* math.cos(t)-5 * math.cos(2 *t)-2 * math.cos(3*t)-math.cos(4 * t))
        else:
            y_2 = y_1
        
        y_min = min(y_1,y_2)
        y_max = max(y_1,y_2)

        return (y>=y_min and y<=y_max)
    
    def heart_func(self,t,enlarge=None):
        px = 16*math.sin(t)**3
        py = -1*(13.* math.cos(t)-5 * math.cos(2 *t)-2 * math.cos(3*t)-math.cos(4 * t))

        # enlarge
        if enlarge == None:
            enlarge = self.IMAGE_ENLARGE
        px *= enlarge
        py *= enlarge

        # shift
        px,py = px+self.cw,py+self.ch

        return (px,py)
    
    @staticmethod
    def shrink(px,py,cx,cy,lamda,bidirectional=False):
        ratio_x,ratio_y = -1*lamda * math.log(random.random()),-1*lamda * math.log(random.random())
        direction = random.random() > 0.5
        if bidirectional and direction:
            dx = ratio_x * (px - cx)
            dy = ratio_y * (py - cy)
            return max(int(px+dx),cx*2),max((int(py+dy)),cy*2)
        else:
            ratio_x = min(ratio_x,1.)
            ratio_y = min(ratio_y,1.)
            dx = ratio_x * (px - cx)
            dy = ratio_y * (py - cy)
            return int(px-dx),int(py-dy)

    def diffusion(self,pt,lamda=0.1,num_per_pt=3,inside=False):
        px,py = pt
        for _ in range(num_per_pt):
            if not inside:
                self.extra_point.add(self.shrink(px,py,self.cw,self.ch,lamda))
            else:
                self.inside.add(self.shrink(px,py,self.cw,self.ch,lamda))

    def heart_point_generate(self):
        # 生成心形线上的点
        while len(self.point_list) < self.num_pts:
            t = random.uniform(0,2*math.pi)
            pt = self.heart_func(t)
            self.point_list.add(tuple(map(int,pt)))
        
        # 边缘扩散
        for pt in self.point_list:
            self.diffusion(pt,lamda=0.025,num_per_pt=1)
            # self.diffusion(pt,lamda=0.05,num_per_pt=1)
            # self.diffusion(pt,lamda=0.075,num_per_pt=1)

            # 内部扩散
            self.diffusion(pt,lamda=0.15,num_per_pt=5)
        
        # 光晕生成halo
        self.num_halo = 3 * self.num_pts
        while len(self.bg_halo) < self.num_halo:
            t = random.uniform(0,2*math.pi)
            halo_enlarge = random.uniform(0.75,1.25)
            pt = self.heart_func(t,halo_enlarge*self.IMAGE_ENLARGE)
            offset = random.randint(-30,30)
            self.bg_halo.add(tuple(map(int,pt)))
    
    def dance(self,ratio,pt,amplitude,out=False):
        px,py = pt
        cx,cy = self.cw,self.ch
        if (px,py) == (cx,cy):
            return px,py

        force = 1/math.sqrt((px-cx)**2 + (py-cy)**2)
        if not out:
            dx = force * ratio * amplitude * (px-cx) + random.randint(-1,1)
            dy = force * ratio * amplitude * (py-cy) + random.randint(-1,1)
            ax = px - dx
            ay = py - dy
            ax = max(min(ax,cx*2),0)
            ay = max(min(ay,cy*2),0)
            return int(ax),int(ay)
        else:
            dx = force * ratio * amplitude * (px-cx) * 3/math.log(abs(py-cy)+math.e) + random.randint(-1,1)
            dy = force * ratio * amplitude * (py-cy) * 3/math.log(abs(px-cx)+math.e) + random.randint(-1,1)
            ax = px + dx
            ay = py + dy
            ax = max(min(ax,cx*2),0)
            ay = max(min(ay,cy*2),0)
            return int(ax),int(ay)

    def get_frames(self,period=15,amplitude=5):
        frames = []
        ratio_t = lambda t:math.sin(math.pi*2 / period * t)
        for t in range(period):
            ratio = ratio_t(t)
            frames.append(self.calc_dance(ratio,amplitude))
        return frames

    def calc_dance(self,ratio,amplitude):
        all_pts = []
        # edge
        for pt in self.point_list:
            size = random.randint(2,4)
            all_pts.append(self.dance(ratio,pt,amplitude)+(size,self.colors[0]))
        
        # near the edge
        for pt in self.extra_point:
            size = random.randint(1,2)
            all_pts.append(self.dance(ratio,pt,amplitude)+(size,self.colors[1]))
        
        # inside
        for pt in self.inside:
            size = random.randint(1,2)
            all_pts.append(self.dance(ratio,pt,amplitude)+(size,self.colors[2]))
        
        # native halo
        for pt in self.bg_halo:
            size = random.randint(1,2)
            all_pts.append(self.dance(ratio,pt,amplitude)+(size,self.colors[3]))

        # global halo
        num_global_halo = self.num_halo * (ratio**2 + 1)
        global_halo = set()
        global_halo_1 = set()

        while len(global_halo_1) < 300:
            t = random.uniform(-1*math.pi/6 , 1*math.pi/6)
            pt = self.heart_func(t,self.IMAGE_ENLARGE)
            hx,hy = tuple(map(int,pt))
            # lamda = 0.15
            # global_halo.add(self.shrink(hx,hy,self.cw,self.ch,lamda=lamda,bidirectional=True))
            offset_y = int(random.randint(15,35)*(hy/180)**(3))
            global_halo_1.add((hx,hy-offset_y))
        for pt in global_halo_1:
            size = 1
            all_pts.append(self.dance(ratio**2,pt,amplitude,out=True) + (size,self.colors[4]))

        while len(global_halo) < num_global_halo:
            halo = random.choice(list(self.bg_halo))
            hx,hy = halo
            lamda = random.uniform(0.12,0.16)
            global_halo.add(self.shrink(hx,hy,self.cw,self.ch,lamda=lamda,bidirectional=True))
        for pt in global_halo:
            size = 1
            all_pts.append(self.dance(ratio**2,pt,amplitude,out=True) + (size,self.colors[4]))
            # all_pts.append(pt + (1,self.colors[3]))

        # get single frame
        frame = deepcopy(self.image)
        for px,py,ps,pc in all_pts:
            frame[py:py+ps,px:px+ps,:] = Hex2RGB(pc)
        
        return frame



    def draw(self,edge_width:int=2):
        plt.figure(figsize=(12,8))
        plt.ion()# 打开交互模式
        frames = self.get_frames()
        while True:
            for i,frame in enumerate(frames):
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.imshow(frame,vmin=0,vmax=255)
                plt.pause(0.01)
                plt.cla()
        # plt.show()
if __name__ == "__main__":
    heart = Heart(1000)
    print(len(heart))
    print(heart)

    heart.draw()