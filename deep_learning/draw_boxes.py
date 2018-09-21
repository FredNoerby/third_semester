import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
from generate_xml import write_xml

def line_select_callback(clk, rls):
	""" clk: click data
		rls: release data
	"""
	global tl_list
	global br_list
	tl_list.append((int(clk.xdata), int(clk.ydata)))
	br_list.append((int(rls.xdata), int(rls.ydata))) 
	object_list.append(obj)

def on_key_press(event):
	global object_list
	global tl_list
	global br_list
	global img
	if event.key == 'q':
		write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
		tl_list = []
		br_list = []
		object_list = []
		img = None
		plt.close()

def toggle_selector(event):
	toggle_selector.RS.set_active(True)


# Global constants
img = None
tl_list = []  # Used to store topleft mouseclicks
br_list = []  # Used to store bottomright mouseclicks
object_list = []  # Used to store all the objects in the image

# Constants
image_folder = 'images'  # Folder with image files
savedir = 'annotations'  # Where to save xml annotations
obj = 'banana'  # object marked in images

if __name__ == '__main__':
	# Goes through files in folder
	for n, image_file in enumerate(os.scandir(image_folder)):
		# Creates a plot and shows current file in plot
		img = image_file
		fig, ax = plt.subplots(1)
		mngr = plt.get_current_fig_manager()
		mngr.window.geometry("2000x2000")  # Changes the size of window
		image = cv2.imread(image_file.path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		ax.imshow(image)

		toggle_selector.RS = RectangleSelector(
			ax, line_select_callback,  # Window and function to call
			drawtype='box', useblit=True,  # Draws a box
			button=[1], minspanx=5, minspany=5,  # button[1] = left mouseclick
			spancoords='pixels', interactive=True
			)
		bbox = plt.connect('key_press_event', toggle_selector)  # Calls function on key press
		key = plt.connect('key_press_event', on_key_press)
		plt.show()
