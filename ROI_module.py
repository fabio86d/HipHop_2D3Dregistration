import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

class ROIbuilder:

    def __init__(self,ax, nr, nc):

        self.ax = ax
        self.nr = nr
        self.nc = nc

        # callbacks
        self.idpress_init = self.ax.figure.canvas.mpl_connect('button_press_event', self.starting_point)
        self.idpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.button_press_callback)


    def starting_point(self,event):
        print("starting point")
        xs_init = event.xdata
        ys_init = event.ydata
        line, = self.ax.plot([xs_init], [ys_init])
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.mask = []
        self.ax.figure.canvas.mpl_disconnect(self.idpress_init)

    def get_binary_mask(self):
        print("calculating mask")
        xycrop = np.vstack((self.xs, self.ys)).T
        pth = Path(xycrop, closed=False)
        ygrid, xgrid = np.mgrid[:self.nr, :self.nc]
        xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
        mask = pth.contains_points(xypix)
        mask = mask.reshape((self.nr,self.nc))
        self.mask = mask

    def button_press_callback(self, event):
        
        if self.line == None:
            return

        if event.inaxes!=self.line.axes: return

        if event.button != 1: 
            self.line.figure.canvas.mpl_disconnect(self.idpress)
            self.get_binary_mask()
            print('disconnected. Line is', self.line)
            return

        # print 'click', event
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        # print 'the line', self.line, 'contains', self.xs, self.ys



class ROIbuilder_rect:

    def __init__(self,window_name, image):

        print("init is declared")
        self.window_name = window_name
        self.image = image.copy()
        self.refPt = [None]*2
        self.refPt_set = []
        self.mask = np.zeros(image.shape, np.bool)
        self.masks_set = []
        #self.rois_set = []

        cv2.setMouseCallback(window_name, self.draw_rect_roi)
        
        while True:
            key = cv2.waitKey(0)

            if key == ord("c"):

                cv2.destroyAllWindows()

                # display masks set
                for i in self.masks_set:

                    cv2.imshow("mask", i.astype(np.float))
                    cv2.waitKey(0)

                cv2.destroyAllWindows()

                break

    def draw_rect_roi(self, event, x, y, flags, param):
 
        #print "draw_rect_roi is called"
        #print 'refPt_set', self.refPt_set
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            
            self.refPt[0] = (x, y)
 
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt[1] = (x,y)

            self.refPt_set.append(self.refPt)

            # update mask
            self.mask[self.refPt[0][1]: self.refPt[1][1] , self.refPt[0][0]: self.refPt[1][0]] = True

            # update masks_set
            self.masks_set.append(self.mask)

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)

            self.refPt = [None]*2
            self.mask = np.zeros(self.image.shape, np.bool)



class ROIbuilder_rect_new:

    def __init__(self, image, name_window = []):

        if not name_window:
            self.window_name = "Draw rectangles from upper-right to lower-left corner. Press c to finish."
        else:
            self.window_name = name_window

        cv2.namedWindow(self.window_name)    

        print("initialization ROIbuilder_rectangle")

        self.image = image.copy()
        self.refPt = [None]*2
        self.refPt_set = []
        self.mask = np.zeros(self.image.shape, dtype=np.uint8)
        self.masks_set = []
        self.P2 = []
        self.drawing = False

        cv2.setMouseCallback(self.window_name, self.draw_rect_roi)

        cv2.imshow(self.window_name, image) 
        
        while True:

            #cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(0)

            if key == ord("c"):
                cv2.destroyAllWindows()
                break

    def draw_rect_roi(self, event, x, y, flags, param):
 
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed

        if event == cv2.EVENT_LBUTTONDOWN:
            
            self.refPt[0] = (x, y)
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:

            self.P2 = (x,y)
 
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished

            self.drawing = False
            self.refPt[1] = (x,y)

            self.refPt_set.append(self.refPt)

            # update mask
            self.mask[self.refPt[0][1]: self.refPt[1][1] , self.refPt[0][0]: self.refPt[1][0]] = 255

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)

            # update masks_set
            self.masks_set.append(self.mask)

            # restore mask
            self.refPt_set = []
            self.mask = np.zeros(self.image.shape, dtype=np.uint8)

        self.draw()


    def draw(self):

        if self.drawing:
        
            image = self.image.copy()

            cv2.rectangle(image, self.refPt[0], self.P2, (0, 255, 0), 2)   

            cv2.imshow(self.window_name, image)


    def save_masks_together(self, mask_save_name):
        
        # generate final mask
        self.final_mask = np.zeros(self.image.shape, dtype=np.uint8)

        for i in self.masks_set:

             self.final_mask += i

        cv2.namedWindow('Final mask')
        cv2.imshow('Final mask', roi.final_mask)
        cv2.waitKey(0)

        cv2.imwrite( mask_save_name, self.final_mask)

        return


class ROIbuilder_polygon:

    def __init__(self, image, name_window = []):
        
        ## load and display image
        #image = cv2.imread(file_dir, cv2.IMREAD_ANYDEPTH) # 16 bit image cv2.IMREAD_ANYDEPTH
        if not name_window:
            self.window_name = "Draw polygons. Double Click at last point. Press c to finish."
        else:
            self.window_name = name_window

        cv2.namedWindow(self.window_name)        
        
        print("initialization ROIbuilder_polygon")
        self.image = image.copy()
        self.refPt_set = []
        self.mask = np.zeros(self.image.shape, dtype=np.uint8)
        self.masks_set = []
        #self.rois_set = []
        self.ignore_mask_color = 255

        cv2.setMouseCallback(self.window_name, self.draw_poly_roi)
        
        while True:

            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1)

            if key == ord("c"):
                cv2.destroyAllWindows()
                break

    def draw_poly_roi(self, event, x, y, flags, param):

        # check to see if the left mouse button was released
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt_set.append((x,y))

        elif event == cv2.EVENT_MOUSEMOVE and len(self.refPt_set) > 1:
            cv2.line(self.image, self.refPt_set[-2], self.refPt_set[-1], 0)

        elif event == cv2.EVENT_LBUTTONDBLCLK:
        
            # draw last closing line
            cv2.line(self.image, self.refPt_set[-1], self.refPt_set[0], 0)

            # record the last point
            self.refPt_set.append((x,y))

            print(np.array([self.refPt_set], dtype=np.int32))

            # draw polygon mask
            cv2.fillPoly(self.mask, np.array([self.refPt_set], dtype=np.int32), self.ignore_mask_color)
         
            # update masks_set
            self.masks_set.append(self.mask)

            # restore mask
            self.refPt_set = []
            self.mask = np.zeros(self.image.shape, dtype=np.uint8)

    def save_masks_together(self, mask_save_name):
        
        # generate final mask
        self.final_mask = np.zeros(self.image.shape, dtype=np.uint8)

        for i in self.masks_set:

             self.final_mask += i

        cv2.namedWindow('Final mask')
        cv2.imshow('Final mask', roi.final_mask)
        cv2.waitKey(0)

        cv2.imwrite( mask_save_name, self.final_mask)

        return

    

class ROIbuilder_polygon_new:

    def __init__(self, image, name_window = []):

        if not name_window:
            self.window_name = "Draw polygons. Double Click at last point. Press c to finish."            
        else:
            self.window_name = name_window

        cv2.namedWindow(self.window_name)    

        print("initialization ROIbuilder_polygon")

        self.image = image.copy()
        self.refPt_set = []
        self.mask = np.zeros(self.image.shape, dtype=np.uint8)
        self.ignore_mask_color = 255
        self.masks_set = []
        self.drawing = False

        cv2.setMouseCallback(self.window_name, self.draw_poly_roi)

        cv2.imshow(self.window_name, image) 
        
        while True:

            #cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(0)

            if key == ord("c"):
                cv2.destroyAllWindows()
                break

    def draw_poly_roi(self, event, x, y, flags, param):
 
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed

        if event == cv2.EVENT_LBUTTONDOWN:
           
            self.refPt_set.append((x,y))
            self.drawing = True
            
            if len(self.refPt_set) > 1:

                # draw permanent line in the image
                cv2.line(self.image, self.refPt_set[-2], self.refPt_set[-1], 0)

        if event == cv2.EVENT_MOUSEMOVE:

            self.P2 = (x,y)
 
        # check to see if the left mouse button was released
        if event == cv2.EVENT_LBUTTONDBLCLK:

            # record the last point
            self.refPt_set.append((x,y))

            print(np.array([self.refPt_set], dtype=np.int32))

            # draw last closing line
            cv2.line(self.image, self.refPt_set[-1], self.refPt_set[0], 0)

            # draw polygon mask
            cv2.fillPoly(self.mask, np.array([self.refPt_set], dtype=np.int32), self.ignore_mask_color)
         
            # update masks_set
            self.masks_set.append(self.mask)

            # restore mask
            self.refPt_set = []
            self.mask = np.zeros(self.image.shape, dtype=np.uint8)

            self.drawing = False

        # must be called all the times
        self.draw()


    def draw(self):

        if self.drawing:
        
            # the current self.image possibly has already the polygons drawn before
            image = self.image.copy()

            cv2.line(image, self.refPt_set[-1], self.P2, 0)

            cv2.imshow(self.window_name, image)



    def save_masks_together(self, mask_save_name):
        
        # generate final mask
        self.final_mask = np.zeros(self.image.shape, dtype=np.uint8)

        for i in self.masks_set:

             self.final_mask += i

        cv2.namedWindow('Final mask')
        cv2.imshow('Final mask', roi.final_mask)
        cv2.waitKey(0)

        cv2.imwrite( mask_save_name, self.final_mask)

        return


class ROIbuilder_circle:

    def __init__(self, image, name_window = []):

        if not name_window:
            self.window_name = "Draw circles. Double Click at last point. Press c to finish."            
        else:
            self.window_name = name_window

        cv2.namedWindow(self.window_name)    

        print("initialization ROIbuilder_circle")

        self.image = image.copy()
        self.center = []
        self.initial_radius = 1
        self.radius = self.initial_radius
        self.mask = np.zeros(self.image.shape, dtype=np.uint8)
        self.masks_set = []
        self.drawing = False

        cv2.setMouseCallback(self.window_name, self.draw_circle_roi)

        #self.center = (100,100)
        #self.radius = np.sqrt((self.center[0] - 50)**2 +  (self.center[1] - 50)**2)
        #cv2.circle(self.image, self.center , int(self.radius), (0,255,0))

        cv2.imshow(self.window_name, self.image) 

        while True:

            #cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(0)

            if key == ord("c"):
                cv2.destroyAllWindows()
                break

    def draw_circle_roi(self, event, x, y, flags, param):
 
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed

        if event == cv2.EVENT_LBUTTONDOWN:
           
            self.center= (x,y)
            self.drawing = True
            
            #if len(self.refPt_set) > 1:

            #    # draw permanent line in the image
            #    cv2.line(self.image, self.refPt_set[-2], self.refPt_set[-1], 0)

        if event == cv2.EVENT_MOUSEMOVE:

            if self.drawing:
                self.radius = np.sqrt((self.center[0] - x)**2 +  (self.center[1] - y)**2)

        # check to see if the left mouse button was released
        if event == cv2.EVENT_LBUTTONUP:

            # draw polygon mask
            cv2.circle(self.mask, self.center, int(self.radius), 255, -1)
         
            # update masks_set
            self.masks_set.append(self.mask)

            # draw permanent circle in original image
            cv2.circle(self.image, self.center, int(self.radius), 255)

            # restore values
            self.radius = self.initial_radius
            self.center = []
            self.mask = np.zeros(self.image.shape, dtype=np.uint8)

            self.drawing = False

        # must be called all the times
        self.draw()


    def draw(self):

        if self.drawing:
        
            # the current self.image possibly has already the polygons drawn before
            image = self.image.copy()

            cv2.circle(image, self.center , int(self.radius), 255)

            cv2.imshow(self.window_name, image)



    def save_masks_together(self, mask_save_name):
        
        # generate final mask
        self.final_mask = np.zeros(self.image.shape, dtype=np.uint8)

        for i in self.masks_set:

             self.final_mask += i

        cv2.namedWindow('Final mask')
        cv2.imshow('Final mask', roi.final_mask)
        cv2.waitKey(0)

        cv2.imwrite( mask_save_name, self.final_mask)

        return


class ROI:

    def __init__(self, image, name_window = []):

        if not name_window:
            self.window_name = "Draw ROIs (c = circle, r = rectangle, p = polygon. Press f to finish."            
        else:
            self.window_name = name_window

        cv2.namedWindow(self.window_name)    

        print("initialization ROI \n Rectangular ROI ")

        self.image = image.copy()
        self.mask = np.zeros(self.image.shape, dtype=np.uint8)
        self.masks_set = []
        self.drawing = False
        self.mode = 'rect' # default

        # parameters for circle
        self.center = []
        self.initial_radius = 1
        self.radius = self.initial_radius
        
        # parameters for rectangle or polygon
        self.refPt = [None]*2     
        self.P1 = [None]*2
        self.P2 = [None]*2
        self.refPt_set = []

        # set mouse callback
        cv2.setMouseCallback(self.window_name, self.draw_roi)

        # display initial image
        cv2.imshow(self.window_name, self.image) 


        while True:

            key = cv2.waitKey(0)

            if key == ord("c"):

                print('Circle ROI selected')
                self.mode = 'circle'

            elif key == ord("r"):

                print('Rectangular ROI selected')
                self.mode = 'rect'

            elif key == ord("p"):

                print('Polygonal ROI selected')
                self.mode = 'poly'       


            if key == ord("f"):

                cv2.destroyAllWindows()

                break


    def draw_roi(self, event, x, y, flags, param):


        if event == cv2.EVENT_LBUTTONDOWN:

            if self.mode == 'circle':
                self.center= (x,y)

            elif self.mode == 'rect':
                # store 
                self.refPt[0] = (x, y)
                # for drawing
                self.P1 = (x, y)

            elif self.mode == 'poly':
                self.refPt_set.append((x,y))
                if len(self.refPt_set) > 1:

                    # draw permanent line in the image
                    cv2.line(self.image, self.refPt_set[-2], self.refPt_set[-1], 0)

            self.drawing = True
            

        if event == cv2.EVENT_MOUSEMOVE:

            if self.mode == 'circle':

                if self.drawing:
                    self.radius = np.sqrt((self.center[0] - x)**2 +  (self.center[1] - y)**2)

            elif self.mode == 'rect':
                    self.P2 = (x,y)

            elif self.mode == 'poly':
                    self.P2 = (x,y)


        if event == cv2.EVENT_LBUTTONUP:

            if self.mode == 'circle':

                # draw plain circle into mask
                cv2.circle(self.mask, self.center, int(self.radius), 255, -1)
         
                # update masks_set
                self.masks_set.append(self.mask)

                # draw permanent circle line in original image
                cv2.circle(self.image, self.center, int(self.radius), 255)

                # restore values
                self.radius = self.initial_radius
                self.center = []
                self.mask = np.zeros(self.image.shape, dtype=np.uint8)

                print('Now the drawing mode can be switched, if needed')
                self.drawing = False


            elif self.mode == 'rect':

                self.drawing = False
                self.refPt[1] = (x,y)

                self.refPt_set.append(self.refPt)

                # update mask
                self.mask[self.refPt[0][1]: self.refPt[1][1] , self.refPt[0][0]: self.refPt[1][0]] = 255

                # draw a rectangle around the region of interest
                cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)

                # update masks_set
                self.masks_set.append(self.mask)

                # restore mask
                self.refPt_set = []
                self.mask = np.zeros(self.image.shape, dtype=np.uint8)

                print('Now the drawing mode can be switched, if needed')


        if event == cv2.EVENT_LBUTTONDBLCLK:

            if self.mode == 'poly':

                # record the last point
                self.refPt_set.append((x,y))

                # draw last closing line
                cv2.line(self.image, self.refPt_set[-1], self.refPt_set[0], 0)

                # draw polygon mask
                cv2.fillPoly(self.mask, np.array([self.refPt_set], dtype=np.int32), 255)
         
                # update masks_set
                self.masks_set.append(self.mask)

                # restore mask
                self.refPt_set = []
                self.mask = np.zeros(self.image.shape, dtype=np.uint8)

                print('Now the drawing mode can be switched, if needed')

                self.drawing = False


        # draw must be called all the times
        self.draw()


    def draw(self):

        if self.drawing:
        
            # the current self.image possibly has already the polygons drawn before
            image = self.image.copy()

            if self.mode == 'circle':

                cv2.circle(image, self.center , int(self.radius), 255)

            elif self.mode == 'rect':

                cv2.rectangle(image, self.P1, self.P2, (0, 255, 0), 2)


            elif self.mode == 'poly':

                cv2.line(image, self.refPt_set[-1], self.P2, 0)

            # show image
            cv2.imshow(self.window_name, image)



    def save_masks_together(self, mask_save_name):
        
        # generate final mask
        self.final_mask = np.zeros(self.image.shape, dtype=np.uint8)

        for i in self.masks_set:

             self.final_mask += i

        cv2.namedWindow('Final mask')
        cv2.imshow('Final mask', roi.final_mask)
        cv2.waitKey(0)

        cv2.imwrite( mask_save_name, self.final_mask)

        return


class ROI_advanced:


    def __init__(self, image, 
                 name_window = 'Draw ROIs (c = circle, r = rectangle, p = polygon. Press f to finish.', 
                 InitParam = {'minImgContrast': 0, 'maxImgContrast': 2**16, 
                              'CannyThresh1': 0, 'CannyThresh2': 0,
                              'Aperture': 1}):

        #print(imagePath)
        #self.image = cv2.imread( imagePath + '.tif', cv2.IMREAD_ANYDEPTH)
        #cv2.imshow('test_original',self.image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #min_img_contrast = np.amin(image)
        #max_img_contrast = np.amax(image)
        #print(min_img_contrast, max_img_contrast)
        #print(type(image[0][0]))

        # rescale 16bit into 8bit with contrast stretching
        #img_for_canny = self.look_up_table(image, min_img_contrast, max_img_contrast)
        self.image = image
        self.processed_img = self.look_up_table(self.image, 0, 2**16-1)
        self.ChosenParam = InitParam

        #Create Trackbars
        self.window_name = name_window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Min',self.window_name,InitParam['minImgContrast'],2**16-1,self.callback)  # min Rescale Image
        cv2.createTrackbar('Max',self.window_name,InitParam['maxImgContrast'],2**16-1,self.callback)  # max Rescale Image
        cv2.createTrackbar('Canny1',self.window_name,InitParam['CannyThresh1'],10000,self.callback)  # Canny Edge Threshold 1
        cv2.createTrackbar('Canny2',self.window_name,InitParam['CannyThresh2'],10000,self.callback)  # Canny Edge Threshold 2
        cv2.createTrackbar('Aperture',self.window_name,InitParam['Aperture'],3,self.callback)  # Canny Edge Threshold 2
        cv2.createTrackbar('Int/Edge', self.window_name,0,1,self.callback)

        # set mouse callback
        cv2.setMouseCallback(self.window_name, self.draw_roi)
        self.mask = np.zeros(self.image.shape, dtype=np.uint8)
        self.inclusion_masks_set = []
        self.exclusion_masks_set = []
        self.inclusion_on = True
        self.drawing = False
        self.mode = 'poly' # default

        # parameters for circle
        self.center = []
        self.initial_radius = 1
        self.radius = self.initial_radius
        
        # parameters for rectangle or polygon
        self.refPt = [None]*2     
        self.P1 = [None]*2
        self.P2 = [None]*2
        self.refPt_set = []

        # Display image
        cv2.imshow(self.window_name,self.processed_img)

        while(1):

            if self.drawing:
                self.draw()
            else:
                cv2.imshow(self.window_name,self.processed_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # Esc button

                self.ChosenParam = self.GetChosenParam()

                break

            if key == ord("c"):

                print('Circle ROI selected')
                self.mode = 'circle'

            elif key == ord("r"):

                print('Rectangular ROI selected')
                self.mode = 'rect'

            elif key == ord("p"):

                print('Polygonal ROI selected')
                self.mode = 'poly'       

            elif key == ord("e"):

                print('Exclusion Masks On')
                self.inclusion_on = False
                #self.processed_img = self.look_up_table(self.image, 0, 2**16-1) 

    # define callback function
    def callback(self, x):

        # get current canny parameters
        minImgContrast = cv2.getTrackbarPos('Min',self.window_name)
        maxImgContrast = cv2.getTrackbarPos('Max',self.window_name)
        CannyThresh1 = cv2.getTrackbarPos('Canny1',self.window_name)
        CannyThresh2 = cv2.getTrackbarPos('Canny2',self.window_name)
        Aperture = cv2.getTrackbarPos('Aperture',self.window_name)
        switch = cv2.getTrackbarPos('Int/Edge',self.window_name)

        # Rescale 16bit into 8bit with contrast stretching
        rescaled_img = self.look_up_table(self.image, minImgContrast, maxImgContrast)

        if switch == 1 and Aperture != 0:

            # Apply canny edge to current 8bit image
            self.processed_img = cv2.Canny(rescaled_img, CannyThresh1, CannyThresh2,L2gradient=False,apertureSize= int(2*Aperture + 1)) # int(2*Aperture + 1)

        elif switch == 0:

            self.processed_img = rescaled_img



    def clip_and_rescale(self, img, min, max):

        image = np.array(img, copy = True) # just create a copy of the array
        image.clip(min,max, out = image)
        image -= min
        #image //= (max - min + 1)/256.
        image = np.divide(image,(max - min + 1.)/256.)
        return image.astype(np.uint16)

    def look_up_table(self, image, min, max):

        lut = np.arange(2**16, dtype = np.uint16)  # lut = look up table
        lut = self.clip_and_rescale(lut, min, max)
        return np.take(lut, image.astype(np.uint16)).astype(np.uint8)  # it s equivalent to lut[image] that is "fancy indexing"

    def GetChosenParam(self):

        self.ChosenParam['minImgContrast'] = cv2.getTrackbarPos('Min',self.window_name)
        self.ChosenParam['maxImgContrast'] = cv2.getTrackbarPos('Max',self.window_name)
        self.ChosenParam['CannyThresh1'] = cv2.getTrackbarPos('Canny1',self.window_name)
        self.ChosenParam['CannyThresh2'] = cv2.getTrackbarPos('Canny2',self.window_name)
        self.ChosenParam['Aperture'] = cv2.getTrackbarPos('Aperture',self.window_name)

        return self.ChosenParam


    def update_mask_set(self):

        if self.inclusion_on:
            self.inclusion_masks_set.append(self.mask)
        else:
            self.exclusion_masks_set.append(self.mask)
        

    def draw_roi(self, event, x, y, flags, param):


        if event == cv2.EVENT_LBUTTONDOWN:

            if self.mode == 'circle':
                self.center= (x,y)

            elif self.mode == 'rect':
                # store 
                self.refPt[0] = (x, y)
                # for drawing
                self.P1 = (x, y)

            elif self.mode == 'poly':
                self.refPt_set.append((x,y))
                if len(self.refPt_set) > 1:

                    # draw permanent line in the image
                    cv2.line(self.processed_img, self.refPt_set[-2], self.refPt_set[-1], 0)

            self.drawing = True
            

        if event == cv2.EVENT_MOUSEMOVE:

            if self.mode == 'circle':

                if self.drawing:
                    self.radius = np.sqrt((self.center[0] - x)**2 +  (self.center[1] - y)**2)

            elif self.mode == 'rect':
                    self.P2 = (x,y)

            elif self.mode == 'poly':
                    self.P2 = (x,y)


        if event == cv2.EVENT_LBUTTONUP:

            if self.mode == 'circle':

                # draw plain circle into mask
                cv2.circle(self.mask, self.center, int(self.radius), 255, -1)
         
                # update masks_set
                self.update_mask_set()
                    
                # draw permanent circle line in original image
                cv2.circle(self.processed_img, self.center, int(self.radius), 255)

                # restore values
                self.radius = self.initial_radius
                self.center = []
                self.mask = np.zeros(self.image.shape, dtype=np.uint8)

                #print('Now the drawing mode can be switched, if needed')
                self.drawing = False


            elif self.mode == 'rect':

                self.drawing = False
                self.refPt[1] = (x,y)

                self.refPt_set.append(self.refPt)

                # update mask
                self.mask[self.refPt[0][1]: self.refPt[1][1] , self.refPt[0][0]: self.refPt[1][0]] = 255

                # draw a rectangle around the region of interest
                cv2.rectangle(self.processed_img, self.refPt[0], self.refPt[1], (0, 255, 0), 2)

                # update masks_set
                self.update_mask_set()

                # restore mask
                self.refPt_set = []
                self.mask = np.zeros(self.image.shape, dtype=np.uint8)

                print('Now the drawing mode can be switched, if needed')


        if event == cv2.EVENT_LBUTTONDBLCLK:

            if self.mode == 'poly':

                # record the last point
                self.refPt_set.append((x,y))

                # draw last closing line
                cv2.line(self.processed_img, self.refPt_set[-1], self.refPt_set[0], 0)

                # draw polygon mask
                cv2.fillPoly(self.mask, np.array([self.refPt_set], dtype=np.int32), 255)
         
                # update masks_set
                self.update_mask_set()

                # restore mask
                self.refPt_set = []
                self.mask = np.zeros(self.image.shape, dtype=np.uint8)

                print('Now the drawing mode can be switched, if needed')

                self.drawing = False


        # draw must be called all the times
        #self.draw()


    def draw(self):

        if self.drawing:
        
            # the current self.image possibly has already the polygons drawn before
            image = self.processed_img.copy()

            if self.mode == 'circle':

                cv2.circle(image, self.center , int(self.radius), 255)

            elif self.mode == 'rect':

                cv2.rectangle(image, self.P1, self.P2, (0, 255, 0), 2)


            elif self.mode == 'poly':

                cv2.line(image, self.refPt_set[-1], self.P2, 0)

            # Show temporary image with drawings
            cv2.imshow(self.window_name, image)

        #else:
        #    # show image
        #    cv2.imshow(self.window_name, self.processed_img)


if __name__ == "__main__":
    
    ## load image
    #img = cv2.imread(sys.argv[1], 0) # 8 bit image 

    #fig = plt.figure()
    #ax = fig.add_subplot(111) # a way to get the axes
    #ax.set_title('click to build ROI')
    #plt.imshow(img,cmap = 'gray',interpolation = 'none')
    #plt.xticks([]), plt.yticks([])
    #roi = ROIbuilder(ax,img.shape)
    #plt.show()

    #plt.imshow(roi.mask,cmap = 'gray',interpolation = 'none')
    #plt.show()

    #print 'Vertices: x coordinates', roi.xs
    #print 'Vertices: y coordinates', roi.ys

    file_dir = r'C:\Users\difabio\Documents\Programming\trialDRR\DRR\DRR\inputs\Landscape\fixed_image\vicon_18d_3mm_01p\C_18d_3mm_75v_01p.tif'
    mask_save_name = r'C:\Users\difabio\Documents\Programming\trialDRR\DRR\DRR\inputs\Landscape\mask\trial_new_rect.png'

    image = cv2.imread(file_dir, cv2.IMREAD_ANYDEPTH) # 16 bit image cv2.IMREAD_ANYDEPTH

    #cv2.namedWindow("image")
    #cv2.imshow("image", image)
    #print(image.shape)
    #roi = ROIbuilder_polygon("image",image)
    name_window = []
    #roi = ROIbuilder_polygon_new(image, name_window)
    #roi = ROIbuilder_rect_new(image, name_window)
    #roi = ROIbuilder_circle(image, name_window)
    roi = ROI(image, name_window)
    #while True:
    #    key = cv2.waitKey(0)

    #    if key == ord("c"):

    #        cv2.destroyAllWindows()
    #        break

    #for i in roi.masks_set:

    #    cv2.imshow("mask", i.astype(np.float))
    #    cv2.waitKey(0)
    #print('masks' , len(roi.masks_set))

    roi.save_masks_together(mask_save_name)



    #cv2.namedWindow("mask")
    #cv2.imshow("mask", roi.mask.astype(np.float))
    
    #cv2.destroyAllWindows()