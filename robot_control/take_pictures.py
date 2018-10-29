import numpy as np
import cv2
from project_robot import ProjectRobot 

def take_picture(counter, name, i):
	cap = cv2.VideoCapture(0)

	switch = True

	while switch:
		ret, frame = cap.read()
		cv2.imshow('Cap', frame)
		cv2.waitKey(30)
		cv2.imwrite('/home/frederik/Desktop/pictures/' + name + '_pos' + str(i) + '_'  + str(counter) + '.jpg', frame)
		print('wrote: ' + name + '_pos' + str(i) + '_'  + str(counter) + '.jpg')
		switch = False


	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


def picture(counter, rob, name):

	for i in range(12):
		rob.go_to(i)
		take_picture(counter, name, i)
		counter += 1

	rob.go_to(0)

	return counter


count = 1
robot = ProjectRobot()

while (True):
	user_in = input("Take pictures? (y/n)")
	if user_in == 'y':
		input_name = input("Type name of set:   ")
		count = picture(count, robot, input_name)
		user_in = ''


	elif user_in == 'n':
		break


robot.close()