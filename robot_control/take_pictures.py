import numpy as np
import cv2
from project_robot import ProjectRobot 

def take_picture(counter, name, i):
	converter_list = [['l', 0], ['l', 1], ['l', 2], ['l', 3], ['l', 4],
				  ['m', 0], ['m', 1], ['m', 2], ['m', 3],
				  ['h', 0], ['h', 1], ['h', 2]]
	cap = cv2.VideoCapture(0)

	switch = True

	while switch:
		ret, frame = cap.read()
		cv2.imshow('Cap', frame)
		cv2.waitKey(30)
		cv2.imwrite('/home/frederik/Desktop/pictures/' + name + '_' + converter_list[i][0] + str(converter_list[i][1]) + '_'  + str(counter) + '.jpg', frame)
		print('wrote: ' + name + '_' + converter_list[i][0] + str(converter_list[i][1]) + '_'  + str(counter) + '.jpg')
		switch = False


	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


def picture(counter, rob, name):

	for i in range(12):
		converter_list = [['l', 0], ['l', 1], ['l', 2], ['l', 3], ['l', 4],
				  ['m', 0], ['m', 1], ['m', 2], ['m', 3],
				  ['h', 0], ['h', 1], ['h', 2]]
		rob.go_to(converter_list[i][0], converter_list[i][1])
		take_picture(counter, name, i)
		counter += 1

	rob.go_to(converter_list[0][0], converter_list[0][1])

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